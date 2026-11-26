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
# Fonts used for typeset maths (LaTeX Computer Modern, Symbol, Cambria Math, STIX, MathType).
MATH_FONTS = re.compile(r"(CMMI|CMSY|CMEX|CMBSY|MSBM|MSAM|Symbol|Math|STIX|MT\s*Extra|Euclid|rsfs)", re.I)
# Symbol-type fonts also carry bullets, so their runs count only with a letter, digit or maths sign.
SYMBOL_FONTS = re.compile(r"symbol", re.I)
# TeX roman: digits, = and brackets inside LaTeX maths. Counted only on a page that already has maths-font runs.
TEX_ROMAN = re.compile(r"^(?:[A-Z]{6}\+)?(CMR|LMRoman|LMMono)", re.I)
BULLET_GLYPHS = set("•∙·◦∗⋆*–—-▪▸►▶■□○●")
OMML_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
MATH_PATTERNS = re.compile(
    r"(\b\w\s*\^\s*\w|\b[a-zA-Z]\s*=\s*[-\w(]|\b(sin|cos|tan|log|ln|lim|exp)\s*\(|\d\s*/\s*\d"
    r"|\b[a-zA-Z]{1,2}\d?\s*[*+]\s*[a-zA-Z0-9]{1,3}\b)"  # short-variable arithmetic: "br + bs", "nr * bs"
)
# Characters from the Symbol font as PowerPoint and Word store them (private-use area),
# minus the codes Wingdings uses for bullets.
PUA_BULLETS = {0xA7, 0xB7, 0xD8, 0xFC, 0xA8, 0x6C, 0x6E, 0x71, 0x75, 0x76, 0x9F, 0xE0, 0xE8, 0x3E, 0x9E, 0x77}


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
    math_spans: int = 0  # text runs set in a maths font (PDF) or equation objects (PPTX)
    ocr_text: str | None = None
    hash: str = ""
    status: str = "ok"


def formula_candidate(text: str, min_symbols: int | None = None) -> bool:
    """True when the text holds enough maths. OCR text loses most symbols, so callers
    checking scanned pages pass a lower min_symbols."""
    cfg = config.ingest().get("formula", {})
    if not text:
        return False
    n_sym = sum(1 for ch in text if ch in MATH_CHARS) + len(MATH_PATTERNS.findall(text))
    n_sym += sum(1 for ch in text if 0xF020 <= ord(ch) <= 0xF0FF and ord(ch) - 0xF000 not in PUA_BULLETS)
    need = int(min_symbols if min_symbols is not None else cfg.get("min_math_symbols", 6))
    return n_sym >= need or n_sym / max(len(text), 1) >= float(cfg.get("min_math_ratio", 0.02))


def _math_span(font: str, text: str) -> str | None:
    """'math' for a text run set in a maths font, 'roman' for TeX roman, else None."""
    t = text.strip()
    if not t or set(t) <= BULLET_GLYPHS:
        return None
    if MATH_FONTS.search(font):
        if SYMBOL_FONTS.search(font) and not any(ch.isalnum() or ch in MATH_CHARS for ch in t):
            return None  # a bullet drawn from a Symbol-type font
        return "math"
    return "roman" if TEX_ROMAN.search(font) else None


def _finish(page: PageData) -> PageData:
    joined = "\n".join(x for x in (page.text, page.tables_md or "", page.notes or "") if x)
    page.has_formula_candidates = formula_candidate(joined) or page.math_spans >= 3
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
            tex_math = tex_roman = 0
            d = pg.get_text("dict", sort=True)
            for block in d.get("blocks", []):
                for line in block.get("lines", []):
                    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in spans).strip()
                    size = round(max(s["size"] for s in spans), 1)
                    bold = any(s.get("flags", 0) & 16 for s in spans)
                    for s in spans:
                        kind = _math_span(str(s.get("font", "")), s["text"])
                        tex_math += kind == "math"
                        tex_roman += kind == "roman"
                    layout.append([text, size, bold])
                    if size > biggest[0] and len(text) > 2:
                        biggest = (size, text)
            page.math_spans = tex_math + (tex_roman if tex_math else 0)
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
        # PowerPoint equations (Insert > Equation) are not part of the normal text runs:
        # read their text so formulas are not lost, and mark the slide for formula extraction.
        for eq in slide._element.iter(f"{{{OMML_NS}}}oMath"):
            eq_text = "".join(t.text or "" for t in eq.iter(f"{{{OMML_NS}}}t")).strip()
            if eq_text:
                texts.append(f"Equation: {eq_text}")
                page.math_spans += 3
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


def vector_kind(data: bytes) -> str | None:
    """'emf', 'wmf' or 'svg' for vector pictures PIL cannot open (old PowerPoint and Word
    equations are usually WMF/EMF pictures), else None."""
    head = data[:2048]
    if data[40:44] == b" EMF":
        return "emf"
    if head[:4] == b"\xd7\xcd\xc6\x9a" or (head[:2] in (b"\x01\x00", b"\x02\x00") and head[2:4] == b"\x09\x00"):
        return "wmf"
    if b"<svg" in head.lower():
        return "svg"
    return None


def vector_to_png(data: bytes, kind: str, timeout: int = 120) -> bytes:
    """Draw a WMF/EMF/SVG picture as PNG with LibreOffice (installed on Kaggle and in the
    Space), cropped to its content. Raises RuntimeError when LibreOffice is missing."""
    import shutil
    import subprocess
    import tempfile

    from PIL import Image, ImageOps

    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError(f"{kind.upper()} picture: LibreOffice is needed to read it")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"picture.{kind}"
        src.write_bytes(data)
        profile = Path(tmp, "profile").as_uri()  # own profile: several OCR workers can convert at once
        subprocess.run(
            [
                soffice,
                f"-env:UserInstallation={profile}",
                "--headless",
                "--convert-to",
                "png",
                "--outdir",
                tmp,
                str(src),
            ],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        out = Path(tmp) / "picture.png"
        if not out.exists():
            raise RuntimeError(f"{kind.upper()} picture could not be converted")
        img = Image.open(out).convert("RGB")
        box = ImageOps.invert(img.convert("L")).getbbox()  # LibreOffice draws it on a whole page
        if box:
            pad = 12
            img = img.crop((max(box[0] - pad, 0), max(box[1] - pad, 0), box[2] + pad, box[3] + pad))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        return buf.getvalue()


def open_image(data: bytes):
    """PIL image from raw bytes, drawing WMF/EMF/SVG pictures with LibreOffice first."""
    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        return img
    except (UnidentifiedImageError, OSError):
        kind = vector_kind(data)
        if not kind:
            raise
        return Image.open(io.BytesIO(vector_to_png(data, kind)))


def to_pil(data: bytes):
    img = open_image(data)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img
