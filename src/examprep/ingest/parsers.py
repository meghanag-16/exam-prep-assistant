"""Parsers for PDF, PPTX and TXT. They read text, layout, tables, speaker notes
and image references, but do no OCR themselves: OCR jobs are collected and run
in parallel afterwards (see ocr.py and pipeline.py).

Every page, slide or text section becomes one PageData, so nothing is skipped.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

from examprep import config

MATH_CHARS = set("∑∫∮√∛≤≥≠≈≡±∓×÷∂∞∝∈∉⊂⊆∪∩∀∃∇∆Δπθλμσφωαβγδεηρτχψ→←↔⇒⇔⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉")
MATH_PATTERNS = re.compile(r"(\b\w\s*\^\s*\w|\b[a-zA-Z]\s*=\s*[-\w(]|\b(sin|cos|tan|log|ln|lim|exp)\s*\(|\d\s*/\s*\d)")


@dataclass
class ImageRef:
    hash: str
    width: int
    height: int
    locator: str  # PDF: "xref:<n>", PPTX: zip part name such as "ppt/media/image3.png"


@dataclass
class PageData:
    page_no: int
    title: str | None = None
    text: str = ""
    tables_md: str | None = None
    notes: str | None = None
    layout: list | None = None
    images: list[ImageRef] = field(default_factory=list)
    needs_page_ocr: bool = False
    has_formula_candidates: bool = False
    ocr_text: str | None = None
    hash: str = ""
    status: str = "ok"


def formula_candidate(text: str) -> bool:
    cfg = config.ingest().get("formula", {})
    if not text:
        return False
    n_sym = sum(1 for ch in text if ch in MATH_CHARS) + len(MATH_PATTERNS.findall(text))
    return n_sym >= int(cfg.get("min_math_symbols", 6)) or n_sym / max(len(text), 1) >= float(
        cfg.get("min_math_ratio", 0.02)
    )


def _finish(page: PageData) -> PageData:
    joined = "\n".join(x for x in (page.text, page.tables_md or "", page.notes or "") if x)
    page.has_formula_candidates = formula_candidate(joined)
    h = hashlib.sha256(joined.encode("utf-8", "ignore"))
    for img in page.images:
        h.update(img.hash.encode())
    page.hash = h.hexdigest()
    if page.needs_page_ocr or page.images:
        page.status = "needs_ocr"  # settled after OCR runs
    elif joined.strip():
        page.status = "ok"
    else:
        page.status = "empty"
    return page


# ---------------------------------------------------------------- PDF


def parse_pdf(path: Path) -> list[PageData]:
    import pymupdf

    cfg = config.ingest()
    min_chars = int(cfg["ocr"]["min_text_chars_for_text_layer"])
    min_px = int(cfg["ocr"]["min_image_px"])
    want_tables = bool(cfg.get("pdf", {}).get("extract_tables", True))
    pages: list[PageData] = []
    with pymupdf.open(path) as doc:
        if doc.needs_pass:
            raise ValueError("PDF is password protected")
        for i, pg in enumerate(doc):
            page = PageData(page_no=i + 1)
            layout = []
            biggest = (0.0, "")
            d = pg.get_text("dict", sort=True)
            for block in d.get("blocks", []):
                for line in block.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in spans).strip()
                    size = round(max(s["size"] for s in spans), 1)
                    bold = any(s.get("flags", 0) & 16 for s in spans)
                    layout.append([text, size, bold])
                    if size > biggest[0] and len(text) > 2:
                        biggest = (size, text)
            page.text = pg.get_text("text", sort=True).strip()
            page.layout = layout or None
            page.title = biggest[1][:200] or None
            if want_tables and page.text:
                try:
                    tabs = pg.find_tables()
                    md = [t.to_markdown() for t in tabs.tables if t.row_count > 1]
                    page.tables_md = "\n\n".join(md) or None
                except Exception:
                    page.tables_md = None
            if len(page.text) < min_chars:
                # Scanned or image-only page: OCR the whole rendered page instead
                # of individual images.
                page.needs_page_ocr = True
            else:
                seen = set()
                for img in pg.get_images(full=True):
                    xref, w, h = img[0], img[2], img[3]
                    if w < min_px or h < min_px or xref in seen:
                        continue
                    seen.add(xref)
                    try:
                        data = doc.extract_image(xref)["image"]
                    except Exception:
                        continue
                    page.images.append(ImageRef(hashlib.sha256(data).hexdigest(), w, h, f"xref:{xref}"))
            pages.append(_finish(page))
    return pages


# ---------------------------------------------------------------- PPTX


def _table_md(table) -> str:
    rows = []
    for r in table.rows:
        rows.append([c.text.replace("\n", " ").replace("|", "/").strip() for c in r.cells])
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _walk_shapes(shapes):
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    for shp in shapes:
        if shp.shape_type == MSO_SHAPE_TYPE.GROUP:
            yield from _walk_shapes(shp.shapes)
        else:
            yield shp


def parse_pptx(path: Path) -> list[PageData]:
    from pptx import Presentation

    min_px = int(config.ingest()["ocr"]["min_image_px"])
    prs = Presentation(str(path))
    pages: list[PageData] = []
    for i, slide in enumerate(prs.slides):
        page = PageData(page_no=i + 1)
        title_shape = slide.shapes.title
        page.title = title_shape.text.strip()[:200] if title_shape is not None and title_shape.has_text_frame else None
        texts, tables, layout = [], [], []
        seen_imgs = set()
        for shp in _walk_shapes(slide.shapes):
            if getattr(shp, "has_text_frame", False) and shp.has_text_frame:
                for para in shp.text_frame.paragraphs:
                    t = para.text.strip()
                    if t:
                        texts.append(("  " * para.level) + t)
                        sizes = [r.font.size.pt for r in para.runs if r.font.size is not None]
                        bold = any(bool(r.font.bold) for r in para.runs)
                        layout.append([t, max(sizes) if sizes else None, bold])
            if getattr(shp, "has_table", False) and shp.has_table:
                tables.append(_table_md(shp.table))
            image = None
            try:
                image = shp.image  # pictures and picture placeholders
            except Exception:
                image = None
            if image is not None:
                try:
                    w, h = image.size
                except Exception:
                    w = h = 0
                if w >= min_px and h >= min_px:
                    digest = hashlib.sha256(image.blob).hexdigest()
                    if digest not in seen_imgs:
                        seen_imgs.add(digest)
                        try:
                            rid = shp._element.blip_rId  # relationship id -> zip part name
                            partname = str(slide.part.related_part(rid).partname).lstrip("/")
                        except Exception:
                            continue
                        page.images.append(ImageRef(digest, w, h, partname))
        page.text = "\n".join(texts)
        page.tables_md = "\n\n".join(t for t in tables if t) or None
        if slide.has_notes_slide:
            notes = slide.notes_slide.notes_text_frame.text.strip() if slide.notes_slide.notes_text_frame else ""
            page.notes = notes or None
        page.layout = layout or None
        pages.append(_finish(page))
    return pages


# ---------------------------------------------------------------- TXT

HEADING = re.compile(
    r"^(#{1,6}\s+\S.*|(unit|chapter|module|topic|section)\s+[\w.]+.*|[A-Z][A-Z0-9 ,:&()/-]{3,80})$", re.I
)


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            if enc == "utf-16" and not raw.startswith((b"\xff\xfe", b"\xfe\xff")):
                continue
            return text.replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def parse_txt(path: Path) -> list[PageData]:
    """Split a text file into sections: at headings when there are any, and
    otherwise at blank lines, keeping each section near `section_chars`."""
    limit = int(config.ingest().get("txt", {}).get("section_chars", 2500))
    text = read_text(path)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    sections: list[tuple[str | None, list[str]]] = []
    cur_title, cur, cur_len = None, [], 0
    for para in paragraphs:
        first = para.split("\n", 1)[0].strip()
        is_heading = bool(HEADING.match(first)) and len(first) <= 80
        if cur and (is_heading or cur_len + len(para) > limit):
            sections.append((cur_title, cur))
            cur_title, cur, cur_len = None, [], 0
        if is_heading and not cur:
            cur_title = first.lstrip("#").strip()
        cur.append(para)
        cur_len += len(para)
    if cur:
        sections.append((cur_title, cur))
    pages = []
    for i, (title, paras) in enumerate(sections or [(None, [""])]):
        pages.append(_finish(PageData(page_no=i + 1, title=title, text="\n\n".join(paras))))
    return pages


def image_bytes(path: Path, kind: str, locator: str) -> bytes:
    """Fetch the raw bytes of one image again (used inside OCR workers)."""
    if kind == "pdf":
        import pymupdf

        with pymupdf.open(path) as doc:
            pix = pymupdf.Pixmap(doc, int(locator.split(":", 1)[1]))
            if pix.n - pix.alpha >= 4:  # CMYK: convert to RGB so it can be saved as PNG
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            return pix.tobytes("png")
    import zipfile

    with zipfile.ZipFile(path) as z:
        return z.read(locator)


def render_page_png(path: Path, page_no: int, dpi: int) -> bytes:
    import pymupdf

    with pymupdf.open(path) as doc:
        pix = doc[page_no - 1].get_pixmap(dpi=dpi)
        return pix.tobytes("png")


def to_pil(data: bytes):
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img
