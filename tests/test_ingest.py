"""Ingestion tests: file detection, safe zip handling, parsing, OCR flow and the page ledger.
All sample files are generated on the fly, so no course material is stored in the repo."""

from __future__ import annotations

import io
import shutil
import sqlite3
import zipfile
from pathlib import Path

import pytest

from examprep.ingest import ledger, ocr, parsers, pipeline, sources


def _text_png(text: str) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (900, 250), "white")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 48)
    except OSError:
        font = ImageFont.load_default(size=48)
    ImageDraw.Draw(img).text((20, 80), text, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def samples(tmp_path: Path) -> Path:
    import pymupdf
    from pptx import Presentation
    from pptx.util import Inches

    d = tmp_path / "course"
    d.mkdir()
    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((72, 72), "Deadlocks", fontsize=24)
    p.insert_text(
        (72, 110), "A deadlock needs mutual exclusion, hold and wait, no preemption, circular wait.", fontsize=10
    )
    p.insert_image(pymupdf.Rect(72, 150, 472, 260), stream=_text_png("Bankers algorithm"))
    p = doc.new_page()  # scanned: image only
    p.insert_image(pymupdf.Rect(50, 50, 550, 200), stream=_text_png("Scanned page about paging"))
    doc.new_page()  # blank
    doc.save(d / "unit_3.pdf")

    prs = Presentation()
    s = prs.slides.add_slide(prs.slide_layouts[1])
    s.shapes.title.text = "Process vs Thread"
    s.placeholders[1].text_frame.text = "Threads share memory"
    s.notes_slide.notes_text_frame.text = "Explain context switch cost"
    s = prs.slides.add_slide(prs.slide_layouts[5])
    s.shapes.title.text = "Comparison"
    t = s.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(4), Inches(1)).table
    for (r, c), v in {(0, 0): "Process", (0, 1): "Thread", (1, 0): "Heavy", (1, 1): "Light"}.items():
        t.cell(r, c).text = v
    s = prs.slides.add_slide(prs.slide_layouts[6])
    png = tmp_path / "pic.png"
    png.write_bytes(_text_png("Semaphore wait signal"))
    s.shapes.add_picture(str(png), Inches(1), Inches(1))
    prs.save(d / "1.pptx")

    (d / "notes.txt").write_text(
        "UNIT 1 INTRODUCTION\n\nAn operating system manages hardware.\n\n"
        "SCHEDULING\n\nRound robin uses a time quantum.\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(tmp_path / "bundle.zip", "w") as z:
        z.write(d / "1.pptx", "slides/1.pptx")  # duplicate of the loose file
        z.writestr("../evil.txt", "zip slip attempt")
        z.writestr("__MACOSX/._junk", "mac metadata")
        inner = io.BytesIO()
        with zipfile.ZipFile(inner, "w") as zi:
            zi.writestr("inner/extra.txt", "Paging divides memory into frames.")
        z.writestr("nested.zip", inner.getvalue())
    return tmp_path


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch) -> Path:
    d = tmp_path / "data"
    monkeypatch.setenv("EXAMPREP_DATA_DIR", str(d))
    return d


def _rows(subject: str, sql: str, *args):
    con = sqlite3.connect(ledger.db_path(subject))
    con.row_factory = sqlite3.Row
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


# ------------------------------------------------------------------ detection


def test_detect_kind_by_content(samples: Path, tmp_path: Path):
    c = samples / "course"
    assert sources.detect_kind(c / "unit_3.pdf") == "pdf"
    assert sources.detect_kind(c / "1.pptx") == "pptx"
    assert sources.detect_kind(c / "notes.txt") == "txt"
    assert sources.detect_kind(samples / "bundle.zip") == "zip"
    renamed = tmp_path / "slides.pdf"
    shutil.copy(c / "1.pptx", renamed)
    assert sources.detect_kind(renamed) == "pptx"
    junk = tmp_path / "junk.pdf"
    junk.write_bytes(bytes(range(256)) * 4)
    assert sources.detect_kind(junk) is None
    docx = tmp_path / "a.docx"
    with zipfile.ZipFile(docx, "w") as z:
        z.writestr("[Content_Types].xml", "<x/>")
        z.writestr("word/document.xml", "<x/>")
    assert sources.detect_kind(docx) is None


def test_collect_dedups_and_unpacks_safely(samples: Path, tmp_path: Path):
    got = sources.collect([samples / "course", samples / "bundle.zip"], tmp_path / "work")
    names = sorted(f.name for f in got.files)
    assert names == ["bundle.zip/nested.zip/inner/extra.txt", "course/1.pptx", "course/notes.txt", "course/unit_3.pdf"]
    reasons = {s.name: s.reason for s in got.skipped}
    assert reasons["bundle.zip/slides/1.pptx"].startswith("duplicate of")
    assert "unsafe path" in reasons["bundle.zip/../evil.txt"]
    assert not any("__MACOSX" in n for n in reasons)
    assert not (tmp_path / "evil.txt").exists()


# ------------------------------------------------------------------ parsers


def test_pptx_parser_reads_title_notes_tables_images(samples: Path):
    pages = parsers.parse_pptx(samples / "course" / "1.pptx")
    assert [p.page_no for p in pages] == [1, 2, 3]
    assert pages[0].title == "Process vs Thread"
    assert pages[0].notes == "Explain context switch cost"
    assert "| Process | Thread |" in pages[1].tables_md
    assert len(pages[2].images) == 1 and pages[2].images[0].locator.startswith("ppt/media/")


def test_pdf_parser_flags_scanned_pages(samples: Path):
    pages = parsers.parse_pdf(samples / "course" / "unit_3.pdf")
    assert len(pages) == 3
    assert pages[0].title == "Deadlocks" and len(pages[0].images) == 1 and not pages[0].needs_page_ocr
    assert pages[1].needs_page_ocr and pages[2].needs_page_ocr


def test_txt_split_at_headings(samples: Path):
    pages = parsers.parse_txt(samples / "course" / "notes.txt")
    assert [p.title for p in pages] == ["UNIT 1 INTRODUCTION", "SCHEDULING"]


def test_formula_candidate():
    assert parsers.formula_candidate("x = a^2 + b^2, y = sin(t), z = 3/4, E = mc^2, f = log(x), k = n/2")
    assert not parsers.formula_candidate("Threads share memory with their process.")


# ------------------------------------------------------------------ pipeline


def test_ingest_without_ocr_then_fill_ocr_later(samples: Path, monkeypatch):
    inputs = [samples / "course", samples / "bundle.zip"]
    report = pipeline.ingest(inputs, "Operating Systems", use_ocr=False, log=lambda m: None)
    assert report["subject"] == "operating_systems"
    cov = report["coverage"]
    assert cov["pages_total"] == 3 + 3 + 2 + 1  # every page/slide/section has a row
    assert cov["pages_needing_ocr"] == 4  # pdf p1 (image), p2 (scan), p3 (blank, treated as scan), slide 3 (image)
    statuses = {
        (r["name"], r["page_no"]): r["status"]
        for r in _rows(
            "operating_systems", "SELECT f.name, p.page_no, p.status FROM pages p JOIN files f ON f.id = p.file_id"
        )
    }
    assert statuses[("course/unit_3.pdf", 2)] == "needs_ocr"
    assert statuses[("course/1.pptx", 3)] == "needs_ocr"
    assert statuses[("course/1.pptx", 1)] == "ok"

    # Second run with OCR "available": only pending work is done, nothing is re-parsed.
    calls = []

    def fake_run_jobs(jobs, progress=None):
        calls.extend(jobs)
        return {j.key: ("" if j.locator == "3" else "ocr words", None) for j in jobs}

    monkeypatch.setattr(ocr, "tesseract_available", lambda: True)
    monkeypatch.setattr(ocr, "run_jobs", fake_run_jobs)
    report = pipeline.ingest(inputs, "operating_systems", log=lambda m: None)
    assert all(r["status"] in {"already ingested", "skipped"} for r in report["this_run"])
    assert {j.what for j in calls} == {"image", "page"}
    cov = report["coverage"]
    assert cov["pages_needing_ocr"] == 0
    assert cov["empty_pages"] == ["course/unit_3.pdf p.3"]

    # Third run: nothing left to do.
    calls.clear()
    pipeline.ingest(inputs, "operating_systems", log=lambda m: None)
    assert calls == []


@pytest.mark.skipif(not ocr.tesseract_available(), reason="Tesseract not installed (expected on the laptop)")
def test_real_ocr(samples: Path):
    report = pipeline.ingest([samples / "course" / "unit_3.pdf"], "os", log=lambda m: None)
    assert report["ocr"]["failed"] == 0
    page2 = _rows("os", "SELECT ocr_text FROM pages WHERE page_no = 2")[0]["ocr_text"]
    assert "paging" in page2.lower()
    imgs = [r["ocr_text"] for r in _rows("os", "SELECT ocr_text FROM images")]
    assert any("bankers" in (t or "").lower() for t in imgs)


def test_broken_file_is_reported_not_fatal(tmp_path: Path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4\nthis is not really a pdf")
    good = tmp_path / "ok.txt"
    good.write_text("Some notes", encoding="utf-8")
    report = pipeline.ingest([bad, good], "s", use_ocr=False, log=lambda m: None)
    status = {r["file"]: r["status"] for r in report["this_run"]}
    assert status["ok.txt"] == "ingested"
    assert status["broken.pdf"] == "failed"


# ------------------------------------------------------------------ API and CLI


def test_api_ingest_and_coverage(samples: Path):
    from fastapi.testclient import TestClient

    from examprep.api.main import app

    client = TestClient(app)
    c = samples / "course"
    files = [
        ("files", ("notes.txt", (c / "notes.txt").read_bytes(), "text/plain")),
        ("files", ("bundle.zip", (samples / "bundle.zip").read_bytes(), "application/zip")),
    ]
    r = client.post("/ingest", data={"subject": "dbms"}, files=files)
    assert r.status_code == 200, r.text
    cov = client.get("/coverage/dbms").json()
    assert cov["pages_total"] == 2 + 3 + 1
    assert client.get("/coverage/nothing_here").status_code == 404


def test_cli_ingest_and_report(samples: Path, capsys):
    from examprep import cli

    cli.ingest_main([str(samples / "course" / "notes.txt"), "--subject", "cn", "--no-ocr"])
    out = capsys.readouterr().out
    assert "[ingested] notes.txt" in out
    cli.report_main(["--subject", "cn"])
    assert "2 page(s)" in capsys.readouterr().out


def test_folder_and_zip_skip_unsupported_types(tmp_path: Path):
    d = tmp_path / "mixed"
    d.mkdir()
    (d / "notes.txt").write_text("Paging", encoding="utf-8")
    (d / "script.py").write_text("print('hi')", encoding="utf-8")
    with zipfile.ZipFile(d / "z.zip", "w") as z:
        z.writestr("README.md", "# readme")
        z.writestr("unit.txt", "Segmentation")
    got = sources.collect([d], tmp_path / "work")
    assert sorted(f.name for f in got.files) == ["mixed/notes.txt", "mixed/z.zip/unit.txt"]
    assert {s.name for s in got.skipped} == {"mixed/script.py", "mixed/z.zip/README.md"}
