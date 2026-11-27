"""Notes, revision sheets and answers as Markdown and PDF.

Markdown is the main format. The PDF is made from the same Markdown, rendered as HTML:
- with WeasyPrint where it is installed (the `pdf` extra, on Kaggle and the Space; it needs
  the Pango library, which the Docker image installs),
- otherwise with PyMuPDF (already installed for reading PDFs), so the laptop needs nothing more.
Formulas appear in the PDF as LaTeX source; rendered formulas come with the website.
"""

from __future__ import annotations

import html
import re
from pathlib import Path


def _where(page_ids: list[int], pages: dict[int, tuple[str, int]]) -> str:
    from examprep.search import search as searching

    refs = [{"name": pages[i][0], "page_no": pages[i][1]} for i in page_ids if i in pages]
    return searching.where(refs)


def notes_markdown(subject: str, notes: list[dict], report: dict | None = None) -> str:
    lines = [f"# Notes: {subject}", ""]
    if report:
        lines += [
            f"Coverage: {report['covered_pages']} of {report['content_pages']} content pages "
            f"({report['coverage']:.0%}) across {report['topics']} topics.",
            "",
        ]
    for n in notes:
        c = n["coverage"]
        lines += [
            f"## {n['topic']}",
            "",
            f"*Pages covered: {n['pages'] or '-'} ({c['covered']} of {c['content_pages']})*",
        ]
        if c["uncovered"]:
            missing = ", ".join(
                f"{u['file'].replace(chr(92), '/').rsplit('/', 1)[-1]} p.{u['page_no']}" for u in c["uncovered"]
            )
            lines.append(f"*Not covered: {missing}*")
        lines.append("")
        for sec in n["sections"]:
            if sec["heading"] and sec["heading"] != n["topic"] and len(n["sections"]) > 1:
                lines += [f"### {sec['heading']}", ""]
                if sec.get("pages"):
                    lines += [f"*Pages: {sec['pages']}*", ""]
            lines += [f"- {p['text']}" for p in sec["points"]]
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def revision_markdown(subject: str, sheets: list[dict], title: str | None = None) -> str:
    lines = [f"# Quick revision: {title or subject}", ""]
    for s in sheets:
        lines += [f"## {s['topic']}", ""]
        if s.get("keywords"):
            lines += ["### Keywords", ""] + [f"- **{k['term']}:** {k['meaning']}" for k in s["keywords"]] + [""]
        if s.get("short_points"):
            lines += ["### Short points", ""] + [f"- {p['text']}" for p in s["short_points"]] + [""]
        for t in s.get("comparisons") or []:
            lines += [f"### {t['title'] or 'Comparison'}", ""]
            lines.append("| " + " | ".join(_cell(c) for c in t["columns"]) + " |")
            lines.append("|" + "---|" * len(t["columns"]))
            lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in t["rows"]]
            lines.append("")
        if s.get("formulas"):
            lines += ["### Formulas", ""]
            for f in s["formulas"]:
                head = f["name"] or "Formula"
                lines += [f"**{head}** ({f['source']})", "", f"$$ {f['latex']} $$", ""]
                if f.get("meaning"):
                    lines.append(f"- Meaning: {f['meaning']}")
                syms = [x for x in f.get("symbols") or [] if isinstance(x, dict) and x.get("symbol")]
                if syms:
                    lines.append("- Symbols: " + "; ".join(f"${x['symbol']}$ = {x.get('meaning', '')}" for x in syms))
                if f.get("units"):
                    lines.append(f"- Units (generated): {f['units']}")
                if f.get("when_to_use"):
                    lines.append(f"- When to use (generated): {f['when_to_use']}")
                ex = f.get("example")
                if ex:
                    given = ", ".join(f"{k} = {v}" for k, v in ex["given"].items())
                    lines.append(
                        f"- Worked example (computed with SymPy): {ex['story']} Given {given}: "
                        f"`{ex['equation']}` gives {ex['find']} = {ex['result']}."
                    )
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ------------------------------------------------------------ PDF

CSS = """
body { font-family: sans-serif; font-size: 10pt; line-height: 1.35; }
h1 { font-size: 18pt; } h2 { font-size: 14pt; margin-top: 14pt; } h3 { font-size: 11.5pt; margin-top: 8pt; }
code { font-family: monospace; font-size: 9pt; }
table { border-collapse: collapse; margin: 4pt 0; } td, th { border: 1px solid #999; padding: 3px 5px; }
th { background-color: #eeeeee; } .muted { color: #555555; font-style: italic; }
"""


def _inline(text: str) -> str:
    out = html.escape(text, quote=False)
    out = re.sub(r"\$\$(.+?)\$\$", r"<code>\1</code>", out)
    out = re.sub(r"\$(.+?)\$", r"<code>\1</code>", out)
    out = re.sub(r"`(.+?)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", out)
    out = re.sub(r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<i>\1</i>", out)
    return re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', out)


def markdown_to_html(md: str) -> str:
    """HTML for the Markdown this package writes: headings, lists, tables, paragraphs."""
    parts: list[str] = []
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if m := re.match(r"(#{1,6})\s+(.*)", line):
            level = len(m.group(1))
            parts.append(f"<h{level}>{_inline(m.group(2))}</h{level}>")
            i += 1
        elif line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", lines[i].strip())[1:-1]]
                if not all(set(c) <= set("-: ") for c in cells):
                    rows.append(cells)
                i += 1
            if rows:
                head = "".join(f"<th>{_inline(c)}</th>" for c in rows[0])
                body = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
                parts.append(f"<table><tr>{head}</tr>{body}</table>")
        elif re.match(r"\s*([-*]|\d+\.)\s+", line):
            ordered = bool(re.match(r"\s*\d+\.", line))
            items = []
            while i < len(lines) and re.match(r"\s*([-*]|\d+\.)\s+", lines[i]):
                items.append(re.sub(r"^\s*([-*]|\d+\.)\s+", "", lines[i]))
                i += 1
            tag = "ol" if ordered else "ul"
            parts.append(f"<{tag}>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + f"</{tag}>")
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not re.match(r"(#{1,6}\s|\s*\||\s*([-*]|\d+\.)\s)", lines[i]):
                para.append(lines[i].strip())
                i += 1
            text = " ".join(para)
            if text.startswith("*") and text.endswith("*") and not text.startswith("**"):
                parts.append(f'<p class="muted">{_inline(text[1:-1])}</p>')
            else:
                parts.append(f"<p>{_inline(text)}</p>")
    return "\n".join(parts)


def _weasyprint(html_body: str, path: Path) -> bool:
    try:
        from weasyprint import HTML
    except (ImportError, OSError):  # not installed, or the Pango library is missing (Windows)
        return False
    page = f"<html><head><meta charset='utf-8'><style>@page {{ size: A4; margin: 17mm }} {CSS}</style></head>"
    HTML(string=page + f"<body>{html_body}</body></html>").write_pdf(str(path))
    return True


def to_pdf(md: str, path: Path, engine: str = "auto") -> Path:
    """engine: auto (WeasyPrint if available, else PyMuPDF), weasyprint or pymupdf."""
    import pymupdf

    if engine in ("auto", "weasyprint") and _weasyprint(markdown_to_html(md), path):
        return path
    if engine == "weasyprint":
        raise RuntimeError("WeasyPrint is not available (pip install -e .[pdf]; it needs the Pango library)")

    story = pymupdf.Story(html=markdown_to_html(md), user_css=CSS)
    writer = pymupdf.DocumentWriter(str(path))
    page = pymupdf.paper_rect("a4")
    area = page + (48, 48, -48, -48)
    more = 1
    while more:
        device = writer.begin_page(page)
        more, _ = story.place(area)
        story.draw(device)
        writer.end_page()
    writer.close()
    # Rewrite once with only the characters used from each font, compressed (about a quarter of the size).
    doc = pymupdf.open(str(path))
    try:
        doc.subset_fonts()
    except Exception:  # noqa: BLE001 (subsetting only makes the file smaller)
        pass
    data = doc.tobytes(garbage=4, deflate=True, deflate_fonts=True)
    doc.close()
    path.write_bytes(data)
    return path


def write(md: str, base: Path, formats: list[str]) -> list[Path]:
    """Write base.md and/or base.pdf. Returns the files written."""
    base.parent.mkdir(parents=True, exist_ok=True)
    out = []
    if "md" in formats:
        p = base.with_suffix(".md")
        p.write_text(md, encoding="utf-8")
        out.append(p)
    if "pdf" in formats:
        out.append(to_pdf(md, base.with_suffix(".pdf")))
    return out
