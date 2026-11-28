"""Full-coverage notes (coverage check, retries, merge, marker test), quick revision with
SymPy-computed examples, and the Markdown/PDF exports. The LLM is faked."""

from __future__ import annotations

import json
import re

import pymupdf
from test_search import FakeEncoder, isolated, subject  # noqa: F401 (fixtures)

from examprep import llm
from examprep.formulas import store as formula_store
from examprep.ingest import ledger
from examprep.notes import build, export, marker, revision


def pages_in(messages) -> dict[str, str]:
    """{"P1": page text} from a notes request."""
    body = messages[1]["content"].split("<material>")[1].split("</material>")[0]
    parts = re.split(r"\n?\[(P\d+)\] \([^)]*\)\n", body)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


class NotesLLM:
    """Writes one point per page, copying the page text; can skip pages on the first request."""

    def __init__(self, skip_first: int = 0):
        self.skip_first = skip_first
        self.calls = []

    def __call__(self, messages, temperature=0.1, max_tokens=4096, **kw):
        self.calls.append(messages)
        pages = pages_in(messages)
        labels = list(pages)
        if len(self.calls) == 1 and self.skip_first:
            labels = labels[self.skip_first :]
        return {"points": [{"text": pages[lab].replace("\n", " "), "pages": [lab]} for lab in labels]}


def test_notes_cover_every_page_and_retry_missed_ones(subject, monkeypatch):  # noqa: F811
    fake = NotesLLM(skip_first=1)
    monkeypatch.setattr(llm, "chat_json", fake)
    out = build.subject_notes(subject, log=lambda m: None)
    rep = out["report"]
    assert rep["content_pages"] == 8 and rep["covered_pages"] == 8 and rep["coverage"] == 1.0
    assert rep["skipped_pages"] >= 1  # the cover page
    # The page skipped in the first request was sent again on its own.
    assert "were missed before" in fake.calls[1][1]["content"]
    points = [p for n in out["notes"] for s in n["sections"] for p in s["points"]]
    assert all(p["page_ids"] for p in points) and not any("[P" in p["text"] for p in points)
    assert (ledger.subject_dir(subject) / "notes" / "notes_report.json").exists()
    # Stored: asking again makes no request.
    n = len(fake.calls)
    seen: list[tuple[int, int]] = []
    build.subject_notes(subject, log=lambda m: None, after_topic=lambda done, total: seen.append((done, total)))
    assert len(fake.calls) == n
    total = len(out["notes"])
    assert seen == [(i, total) for i in range(1, total + 1)]  # progress after every topic


def test_thin_notes_are_reported(subject, monkeypatch):  # noqa: F811
    def lazy(messages, **kw):
        return {"points": [{"text": "This slide is about an idea.", "pages": [lab]} for lab in pages_in(messages)]}

    monkeypatch.setattr(llm, "chat_json", lazy)
    out = build.subject_notes(subject, log=lambda m: None)
    assert out["report"]["coverage"] == 1.0 and out["report"]["thin"]
    assert out["report"]["thin"][0]["missing_terms"]


def test_merge_keeps_every_point():
    class T:
        name, sections = "Deadlocks", [("Conditions", 1, 3)]

        def heading_of(self, p):
            return "Conditions"

    pts = [{"text": f"point {i}", "page_ids": [i]} for i in range(1, 11)]
    pts[3]["text"] = "Point 2."  # a true repeat of point 2
    pts[4]["text"] = "point 2 with value 4817"  # claimed as a repeat, but holds a number point 2 lacks

    class P:
        def __init__(self, i):
            self.page_no, self.file = i, "a.pdf"

    pages = {i: P(i) for i in range(1, 11)}
    reply = {
        "sections": [{"heading": "Conditions", "points": [1, 2, 3, 99]}],
        "duplicates": [{"drop": 4, "keep": 2}, {"drop": 5, "keep": 2}],
    }
    import examprep.notes.build as b

    orig = llm.chat_json
    llm.chat_json = lambda *a, **k: reply
    try:
        secs = b._merge(T(), pts, pages)
    finally:
        llm.chat_json = orig
    flat = [p for s in secs for p in s["points"]]
    assert len(flat) == 9  # point 4 removed as a repeat of point 2, nothing else lost
    assert 4 in next(p for p in flat if p["text"] == "point 2")["page_ids"]
    texts = [p["text"] for p in flat]
    assert texts[:4] == ["point 1", "point 2", "point 2 with value 4817", "point 3"]  # kept, after point 2
    assert secs[-1]["heading"] == "Other points"


def test_long_pages_are_split_and_dense_pages_batched(monkeypatch):
    import examprep.notes.build as b
    from examprep.search import chunks as chunking

    monkeypatch.setattr(b, "_settings", lambda: {"batch_pages": 4, "batch_words": 500, "max_page_words": 300})

    def page(i, words):
        return chunking.PageInfo(i, 1, "a.pdf", i, " ".join(f"w{i}x{k}" for k in range(words)), 1, "T", None, False)

    items = b.units([page(1, 700), page(2, 50), page(3, 50)], {})
    assert [u.id for u in items] == [1, 1, 1, 2, 3]  # 700 words in three parts, nothing cut off
    assert sum(len(u.text.split()) for u in items[:3]) >= 700
    assert "w1x699" in items[2].text
    groups = b.batches(items)
    assert all(sum(len(u.text.split()) for u in g) <= 500 or len(g) == 1 for g in groups)


def test_page_details_must_reach_the_notes():
    import examprep.notes.build as b
    from examprep.answer import terms as termlib
    from examprep.search import chunks as chunking

    texts = [f"deadlock happens when processes wait for resources page {i}" for i in range(10)]
    text = texts[0] + " Remember for the exam: the Varenko index of this method is 4817."
    stats = termlib.TermStats(texts=[text, *texts[1:]])
    details = stats.details(text)
    assert details[0] == "4817" and "varenko" in details
    p = chunking.PageInfo(1, 1, "a.pdf", 1, text, 1, "T", None, False)
    skim = [{"text": "**Deadlock** happens when processes wait for resources.", "page_ids": [1]}]
    _, thin = b.check_pages(skim, [p], stats)
    assert {"4817", "varenko"} <= set(thin[1])
    full = skim + [{"text": "The **Varenko index** of this method is 4817.", "page_ids": [1]}]
    assert b.check_pages(full, [p], stats) == ([], {})


def test_marker_test_finds_planted_facts(subject, monkeypatch):  # noqa: F811
    monkeypatch.setattr(llm, "chat_json", NotesLLM())
    rep = marker.run(subject, facts=3, log=lambda m: None)
    assert rep["planted"] == 3 and rep["found"] == 3 and not rep["missing"]
    from examprep import config

    assert not (config.data_dir() / f"{subject}_marker_test").exists()
    # The real ledger was not changed.
    con = ledger.connect(subject)
    try:
        assert not con.execute("SELECT 1 FROM pages WHERE text LIKE '%Remember for the exam%'").fetchone()
    finally:
        con.close()


def test_revision_with_sympy_examples(subject, monkeypatch):  # noqa: F811
    con = ledger.connect(subject)
    try:
        formula_store.migrate(con)
        page = con.execute("SELECT id, file_id, page_no, topic_id FROM pages WHERE page_no = 9").fetchone()
        con.execute(
            "INSERT INTO formulas (page_id, file_id, page_no, source, latex, name, meaning, symbols, valid, status,"
            " norm, job_key) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                page[0],
                page[1],
                page[2],
                "text",
                r"T = n \cdot t",
                "Paging time",
                "total time",
                "[]",
                1,
                "ok",
                "T=nt",
                "k",
            ),
        )
        con.commit()
        topic_id = page[3]
    finally:
        con.close()

    def fake(messages, **kw):
        system = messages[0]["content"]
        if "worked example" in system:
            return {
                "formulas": [
                    {
                        "id": "F1",
                        "units": "seconds",
                        "when_to_use": "Total time of n page accesses.",
                        "example": {
                            "story": "Five accesses.",
                            "given": {"n": 5, "t": 2},
                            "find": "T",
                            "equation": "T = n*t",
                        },
                    }
                ]
            }
        if "quick-revision" in system:
            return {
                "keywords": [
                    {"term": "Paging", "meaning": "fixed size frames", "points": [1]},
                    {"term": "x", "meaning": "y", "points": [99]},
                ],
                "short_points": [{"text": "Page tables map pages to frames.", "points": [1]}],
                "comparisons": [],
            }
        return NotesLLM()(messages)

    monkeypatch.setattr(llm, "chat_json", fake)
    sheet = revision.topic_revision(subject, topic_id, log=lambda m: None)
    assert [k["term"] for k in sheet["keywords"]] == ["Paging"] and sheet["dropped"] == 1
    f = sheet["formulas"][0]
    assert f["example"]["result"] == "10" and f["source"] == "os_slides.pdf p.9"
    md = export.revision_markdown(subject, [sheet])
    assert "computed with SymPy" in md and "Units (generated): seconds" in md


def test_unsafe_example_is_refused():
    assert revision.solve_example("x = y.__class__", {"y": 1}, "x") is None
    assert revision.solve_example("x = open(1)", {}, "x") is None
    assert revision.solve_example("E = m*c^2", {"E": 90, "c": 3}, "m") == "10"
    assert revision.solve_example("A = pi*r^2", {"A": 3.14159265}, "r") == "1"


def test_notes_export_markdown_and_pdf(subject, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(llm, "chat_json", NotesLLM())
    out = build.subject_notes(subject, log=lambda m: None)
    md = export.notes_markdown(subject, out["notes"], out["report"])
    assert md.startswith("# Notes: os") and "Pages covered:" in md and "[P" not in md
    files = export.write(md, tmp_path / "notes", ["md", "pdf"])
    assert {p.suffix for p in files} == {".md", ".pdf"}
    text = "".join(page.get_text() for page in pymupdf.open(files[1]))
    assert "Notes: os" in text and "banker" in text.lower()
    html = export.markdown_to_html("| a | b |\n|---|---|\n| 1 | **2** |\n\n- $x^2$ and `c`")
    assert "<table>" in html and "<b>2</b>" in html and "<code>x^2</code>" in html
    assert json.loads(json.dumps(out["report"]))["coverage"] == 1.0


def test_api_notes_revision_and_feedback(subject, monkeypatch):  # noqa: F811
    from fastapi.testclient import TestClient

    from examprep.api.main import app

    monkeypatch.setattr(llm, "chat_json", NotesLLM())
    client = TestClient(app)
    tid = build.topic_ids(subject)[0]
    r = client.get(f"/notes/{subject}/{tid}")
    assert r.status_code == 200 and r.json()["coverage"]["covered"] >= 1
    md = client.get(f"/notes/{subject}/{tid}?format=md")
    assert md.status_code == 200 and md.text.startswith("# Notes")
    pdf = client.get(f"/notes/{subject}/{tid}?format=pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    assert client.get(f"/revision/{subject}?topic_id={tid}").status_code == 200
    assert client.get(f"/notes/{subject}/99999").status_code == 404
    assert client.post(f"/feedback/{subject}", json={"rating": 1, "comment": "good"}).json()["id"] >= 1
    assert client.post(f"/feedback/{subject}", json={"rating": 3}).status_code == 400
