"""Formula extraction: detection, LaTeX checks, LLM extraction (faked), image
batches, resumable runs and the review file. No network is used."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from examprep import llm
from examprep.formulas import extract, validate
from examprep.ingest import ledger, parsers, pipeline


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EXAMPREP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(llm, "available_providers", lambda: [])


def _png(w: int = 300, h: int = 120) -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(img).text((10, 40), "x = (-b +- sqrt(b^2 - 4ac)) / 2a", fill="black")
    out = io.BytesIO()
    img.save(out, "PNG")
    return out.getvalue()


def _maths_pdf(path: Path) -> None:
    import pymupdf

    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((72, 72), "Quadratic equations", fontsize=20)
    for k, line in enumerate(
        [
            "The roots are x = (-b + sqrt(d)) / 2a when a is not 0.",
            "Discriminant: d = b^2 - 4ac; if d >= 0 the roots are real.",
            "Matrix form: A = [[a, b], [c, d]], det(A) = ad - bc.",
            "Sum of roots s = -b/a, product p = c/a, and y = sin(x) + log(x).",
        ]
    ):
        p.insert_text((72, 120 + 24 * k), line, fontsize=11, fontname="helv")
    p.insert_image(pymupdf.Rect(72, 260, 372, 380), stream=_png())
    q = doc.new_page()
    q.insert_text((72, 72), "History of databases", fontsize=20)
    q.insert_text((72, 120), "Relational databases were proposed in 1970 and became popular later.", fontsize=11)
    doc.save(path)


def test_check_latex():
    ok = validate.check(r"$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$")
    assert ok.valid and ok.status == "ok" and not ok.latex.startswith("$")
    m = validate.check(r"A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}")
    assert m.valid and m.status == "ok"
    bad = validate.check(r"\frac{a}{b")
    assert not bad.valid and "unbalanced braces" in bad.issues
    env = validate.check(r"\begin{bmatrix} a \end{pmatrix}")
    assert not env.valid
    odd = validate.check(r"\foo{x} + y")
    assert odd.valid and odd.status == "needs_review"
    prose = validate.check("this sentence describes relational databases without any maths at all")
    assert prose.status == "needs_review"
    assert validate.normalise(r"a + b") == validate.normalise(r"a+b")


def test_powerpoint_equations_are_read(tmp_path: Path):
    from lxml import etree
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.shapes.add_textbox(Inches(1), Inches(1), Inches(6), Inches(1)).text_frame.text = "Energy"
    m = "http://schemas.openxmlformats.org/officeDocument/2006/math"
    eq = etree.SubElement(s.shapes._spTree, f"{{{m}}}oMath")
    for t in ("E", "=", "m", "c", "2"):
        etree.SubElement(etree.SubElement(eq, f"{{{m}}}r"), f"{{{m}}}t").text = t
    path = tmp_path / "eq.pptx"
    prs.save(path)
    page = parsers.parse_pptx(path)[0]
    assert "Equation: E=mc2" in page.text
    assert page.has_formula_candidates


def _fake_text_reply(seen: list):
    def fake(messages, temperature=0.1, max_tokens=4096, providers=None):
        seen.append((messages, providers))
        content = messages[-1]["content"]
        if isinstance(content, list):  # image request
            ids = [c["text"].split("id: ")[1] for c in content if c["type"] == "text" and c["text"].startswith("id: ")]
            return {
                "items": [
                    {
                        "id": i,
                        "formulas": [
                            {"latex": r"x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}", "kind": "formula", "name": "Roots"}
                        ],
                    }
                    for i in ids
                ]
            }
        material = json.loads(content.split("<material>")[1].split("</material>")[0])
        return {
            "items": [
                {
                    "id": item["id"],
                    "formulas": [
                        {
                            "latex": r"\Delta = b^2 - 4ac",
                            "kind": "equation",
                            "name": "Discriminant",
                            "meaning": "Decides whether the roots are real.",
                            "symbols": [{"symbol": r"\Delta", "meaning": "discriminant"}],
                        },
                        {"latex": r"A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}", "kind": "matrix"},
                        {"latex": r"\frac{a}{b", "kind": "expression"},
                    ],
                }
                for item in material
            ]
        }

    return fake


def test_extract_text_and_images_and_resume(tmp_path: Path, monkeypatch):
    pdf = tmp_path / "maths.pdf"
    _maths_pdf(pdf)
    pipeline.ingest([pdf], "maths", use_ocr=False, log=lambda m: None)
    seen: list = []
    monkeypatch.setattr(llm, "available_providers", lambda: ["gemini"])
    monkeypatch.setattr(llm, "vision_providers", lambda: ["gemini"])
    monkeypatch.setattr(llm, "chat_json", _fake_text_reply(seen))

    # Without the course files the image waits; page text is still read.
    rep = extract.extract("maths", log=lambda m: None)
    assert rep["this_run"]["waiting_for_files"] == 1
    assert rep["by_kind"] == {"equation": 1, "matrix": 1, "expression": 1}
    assert rep["invalid_latex"] == 1 and rep["needs_review"] == 1
    assert len(seen) == 1  # only the maths page was sent, not the history page

    # With the files, the image goes to the image model; the text is not sent again.
    seen.clear()
    rep = extract.extract("maths", [pdf], log=lambda m: None)
    assert len(seen) == 1 and seen[0][1] == ["gemini"]
    parts = seen[0][0][-1]["content"]
    assert any(p["type"] == "image_url" and p["image_url"]["url"].startswith("data:image/jpeg") for p in parts)
    assert rep["by_kind"]["formula"] == 1 and rep["still_to_do"] == {"text": 0, "image": 0, "page": 0}

    # Nothing left: no calls.
    seen.clear()
    extract.extract("maths", [pdf], log=lambda m: None)
    assert seen == []

    rows = extract.formulas("maths")
    disc = next(r for r in rows if r["name"] == "Discriminant")
    assert disc["page_no"] == 1 and disc["symbols"][0]["meaning"] == "discriminant" and disc["valid"]
    review = (ledger.subject_dir("maths") / "formulas.md").read_text(encoding="utf-8")
    assert review.index("## Needs review") < review.index("Discriminant")
    assert "unbalanced braces" in review


def test_no_key_sends_nothing(tmp_path: Path):
    pdf = tmp_path / "maths.pdf"
    _maths_pdf(pdf)
    pipeline.ingest([pdf], "maths", use_ocr=False, log=lambda m: None)
    rep = extract.extract("maths", [pdf], log=lambda m: None)
    assert rep["stopped"] and rep["formulas"] == 0
    assert rep["still_to_do"]["text"] == 1


def test_scanned_pages_need_fewer_maths_matches():
    ocr = (
        "Dual space-time representation. Each moving object is drawn as a line in the plane and the "
        "query becomes a region; the line is y = vt + a for each object, and the example uses n = 7 points."
    )
    assert extract.looks_like_maths(ocr, scanned=True)
    assert not extract.looks_like_maths(ocr)
    assert not extract.looks_like_maths("Relational databases were proposed in 1970.", scanned=True)


def test_double_escaped_latex_is_repaired():
    fixed = validate.check(r"\\begin{bmatrix}1 & 2 \\ 3 & 4\\end{bmatrix}")
    assert fixed.latex == r"\begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}" and fixed.status == "ok"


def test_latex_slide_fonts_count_as_maths():
    # Beamer/LaTeX slides: symbols in CMMI/CMSY/CMEX, digits and = in CMR, bullets never count.
    assert parsers._math_span("CMMI10", "x") == "math"
    assert parsers._math_span("CMEX10", "") == "math"
    assert parsers._math_span("ABCDEF+CMR10", "= 2") == "roman"
    assert parsers._math_span("CMSY10", "•") is None
    assert parsers._math_span("OpenSymbol", "●") is None
    assert parsers._math_span("NimbusRomNo9L-Regu", "Definition") is None


def test_unescaped_latex_in_json_reply_is_kept():
    # A model that forgets to escape backslashes: \b, \f, \n and \t would become control characters.
    reply = r'{"latex": "\begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix} \neq \frac{a}{b} \times \sigma", "note": "a\nb"}'
    data = llm.parse_json(reply)
    assert data["latex"] == r"\begin{bmatrix}1 & 2 \ 3 & 4\end{bmatrix} \neq \frac{a}{b} \times \sigma"
    assert data["note"] == "a\nb"
    good = llm.parse_json(r'{"latex": "\\begin{bmatrix}1 \\\\ 2\\end{bmatrix}"}')
    assert good["latex"] == r"\begin{bmatrix}1 \\ 2\end{bmatrix}"
    # If a broken one still gets through, the check rejects it.
    assert not validate.check("\x08egin{bmatrix}1 & 2\\end{bmatrix}").valid
    assert not validate.check("a \x0crac{1}{2}").valid
