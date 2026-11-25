"""Turn whatever the user gives (files, folders, zips, zips inside zips) into a
flat list of course files, each identified by its real content type and hash.

- File types are detected from the bytes, not the extension, so a renamed or
  broken file is reported instead of crashing the run.
- Zips are unpacked safely: no paths outside the target folder, size limits,
  limited nesting.
- The same file given twice (e.g. loose and inside a zip) is kept once.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from examprep import config

OLE_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")
ZIP_MAGIC = b"PK\x03\x04"
COURSE_KINDS = {"pdf", "pptx", "ppt", "txt"}


@dataclass
class SourceFile:
    path: Path  # where the bytes are on disk now
    name: str  # what to show the user, e.g. "unit3.zip/slides/1.pptx"
    kind: str  # pdf | pptx | ppt | txt
    sha256: str
    size: int
    note: str = ""


@dataclass
class Skipped:
    name: str
    reason: str


@dataclass
class Collected:
    files: list[SourceFile] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def detect_kind(path: Path) -> str | None:
    """Return pdf | pptx | ppt | txt | zip, or None if not a supported file."""
    with open(path, "rb") as f:
        head = f.read(65536)
    if not head:
        return None
    if b"%PDF-" in head[:1024]:
        return "pdf"
    if head.startswith(ZIP_MAGIC) or head.startswith(b"PK\x05\x06"):
        try:
            with zipfile.ZipFile(path) as z:
                names = set(z.namelist())
        except zipfile.BadZipFile:
            return None
        if "ppt/presentation.xml" in names:
            return "pptx"
        if "[Content_Types].xml" in names:
            return None  # another Office file (docx, xlsx): not supported
        return "zip"
    if head.startswith(OLE_MAGIC):
        # Old binary Office format. Only .ppt is supported; .doc/.xls share this header.
        return "ppt" if path.suffix.lower() in {".ppt", ".pps", ".pot"} else None
    if b"\x00" in head:
        return None
    for enc in ("utf-8", "cp1252"):
        try:
            head.decode(enc)
            return "txt"
        except UnicodeDecodeError:
            continue
    return None


def _safe_member_path(name: str) -> tuple[PurePosixPath | None, str]:
    """Return (safe relative path, "") or (None, reason). Reason "" means silently ignore."""
    p = PurePosixPath(name.replace("\\", "/"))
    if not p.parts:
        return None, ""
    if p.is_absolute() or ".." in p.parts or ":" in p.parts[0]:
        return None, "unsafe path inside zip (points outside the zip folder)"
    if p.parts[0] == "__MACOSX" or any(part.startswith(".") for part in p.parts):
        return None, ""  # macOS metadata and hidden files
    return p, ""


def _extract_zip(zpath: Path, dest: Path, label: str, out: Collected, budget: list[int]) -> list[tuple[Path, str]]:
    """Extract supported members safely. budget[0] = bytes still allowed."""
    max_file = int(config.ingest()["max_file_mb"]) * 1024 * 1024
    extracted: list[tuple[Path, str]] = []
    try:
        zf = zipfile.ZipFile(zpath)
    except zipfile.BadZipFile:
        out.skipped.append(Skipped(label, "not a valid zip file"))
        return extracted
    with zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            member_label = f"{label}/{info.filename}"
            rel, reason = _safe_member_path(info.filename)
            if rel is None:
                if reason:
                    out.skipped.append(Skipped(member_label, reason))
                continue
            if info.file_size > max_file:
                out.skipped.append(Skipped(member_label, f"larger than {max_file >> 20} MB"))
                continue
            target = dest / Path(*rel.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            too_big = False
            with zf.open(info) as src, open(target, "wb") as dst:
                for block in iter(lambda: src.read(1 << 20), b""):
                    written += len(block)
                    if written > max_file or written > budget[0]:
                        too_big = True
                        break
                    dst.write(block)
            if too_big:
                target.unlink(missing_ok=True)
                out.skipped.append(Skipped(member_label, "zip unpacks to more than the allowed size"))
                continue
            budget[0] -= written
            extracted.append((target, member_label))
    return extracted


def collect(inputs: list[Path], work_dir: Path) -> Collected:
    """Expand inputs into unique course files. Zips are unpacked under work_dir."""
    cfg = config.ingest()
    max_file = int(cfg["max_file_mb"]) * 1024 * 1024
    max_depth = int(cfg.get("max_zip_depth", 2))
    out = Collected()
    seen: dict[str, str] = {}
    zip_counter = 0

    accepted = {e.lower() for e in cfg["accepted_extensions"]}

    def wanted(path: Path, label: str) -> bool:
        # Files found inside folders and zips must have a supported extension
        # (so code, images and Word files are not read as text). Files named
        # directly by the user are always checked by content.
        if path.suffix.lower() in accepted:
            return True
        out.skipped.append(Skipped(label, f"unsupported file type ({path.suffix or 'no extension'})"))
        return False

    queue: list[tuple[Path, str, int]] = []
    for p in inputs:
        p = Path(p)
        if p.is_dir():
            for f in sorted(x for x in p.rglob("*") if x.is_file()):
                rel = f.relative_to(p)
                if any(part.startswith(".") or part == "__MACOSX" for part in rel.parts):
                    continue
                label = f"{p.name}/{rel.as_posix()}"
                if wanted(f, label):
                    queue.append((f, label, 0))
        elif p.is_file():
            queue.append((p, p.name, 0))
        else:
            out.skipped.append(Skipped(str(p), "file or folder not found"))

    while queue:
        path, label, depth = queue.pop(0)
        size = path.stat().st_size
        if size == 0:
            out.skipped.append(Skipped(label, "empty file"))
            continue
        if size > max_file:
            out.skipped.append(Skipped(label, f"larger than {max_file >> 20} MB"))
            continue
        kind = detect_kind(path)
        if kind is None:
            out.skipped.append(Skipped(label, "not a PDF, PPTX, PPT, TXT or zip (checked by content)"))
            continue
        if kind == "zip":
            if depth >= max_depth:
                out.skipped.append(Skipped(label, f"zip nested deeper than {max_depth} levels"))
                continue
            zip_counter += 1
            dest = work_dir / f"zip{zip_counter}"
            budget = [int(cfg["max_zip_uncompressed_mb"]) * 1024 * 1024]
            members = _extract_zip(path, dest, label, out, budget)
            queue.extend((m, ml, depth + 1) for m, ml in sorted(members, key=lambda t: t[1]) if wanted(m, ml))
            continue
        digest = sha256_file(path)
        if digest in seen:
            out.skipped.append(Skipped(label, f"duplicate of {seen[digest]}"))
            continue
        seen[digest] = label
        note = ""
        ext = path.suffix.lower().lstrip(".")
        if ext and ext != kind and not (kind == "txt" and ext in {"md", "text"}):
            note = f"extension .{ext} but content is {kind}"
        out.files.append(SourceFile(path=path, name=label, kind=kind, sha256=digest, size=size, note=note))
    return out


def cleanup(work_dir: Path) -> None:
    shutil.rmtree(work_dir, ignore_errors=True)
