"""Topic-map tests: duplicates, deck boundaries, topic segments, naming, edits,
syllabus mapping and the LLM client. Sample files are generated on the fly."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from examprep import llm
from examprep.ingest import pipeline
from examprep.structure import analysis, build, embed
from examprep.structure.store import Page

DECKS = [
    (
        "Process Management",
        "OS Unit 1",
        [
            ("Processes", "A process is a program in execution with its own address space and process control block."),
            ("Process States", "New ready running waiting terminated states form the process lifecycle diagram."),
            ("Context Switch", "The kernel saves registers into the process control block during a context switch."),
            (
                "CPU Scheduling",
                "Scheduling algorithms include first come first served shortest job first and round robin.",
            ),
            ("Round Robin", "Round robin scheduling uses a time quantum and a ready queue for fairness."),
            (
                "Scheduling Criteria",
                "Throughput turnaround time waiting time and response time are scheduling criteria.",
            ),
        ],
    ),
    (
        "Deadlocks",
        "OS Unit 3",
        [
            ("Deadlock Conditions", "Mutual exclusion hold and wait no preemption and circular wait cause deadlock."),
            ("Resource Allocation Graph", "A cycle in the resource allocation graph may indicate deadlock."),
            ("Deadlock Prevention", "Prevention breaks one of the four necessary conditions for deadlock."),
            ("Bankers Algorithm", "The banker's algorithm checks for a safe state using available max and need."),
            ("Safe State", "A safe sequence exists when every process can finish with available resources."),
            ("Deadlock Recovery", "Recovery terminates processes or preempts resources to break the deadlock."),
        ],
    ),
    (
        "Memory Management",
        "OS Unit 4",
        [
            ("Paging", "Paging divides memory into fixed size frames and pages with a page table."),
            ("Page Table", "The page table maps page numbers to frame numbers with a TLB cache."),
            ("Segmentation", "Segmentation divides memory into variable size segments like code stack and heap."),
            ("Virtual Memory", "Virtual memory lets programs larger than physical memory run using demand paging."),
            ("Page Replacement", "FIFO LRU and optimal are page replacement algorithms with Belady's anomaly."),
            ("Thrashing", "Thrashing happens when processes spend more time paging than executing."),
        ],
    ),
]


def _slide(doc, heading: str, body: str, footer: str, number: int) -> None:
    p = doc.new_page()
    p.insert_text((72, 72), heading, fontsize=22)
    y = 120
    for k in range(4):
        p.insert_text((72, y), body if k == 0 else f"Point {k}: " + body.lower(), fontsize=11)
        y += 22
    p.insert_text((72, 760), footer, fontsize=8)
    p.insert_text((540, 780), str(number), fontsize=8)


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EXAMPREP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(embed, "backend_name", lambda: "tfidf")
    monkeypatch.setattr(llm, "available_providers", lambda: [])  # no network in tests unless a test enables it


@pytest.fixture
def combined_pdf(tmp_path: Path) -> Path:
    import pymupdf

    doc = pymupdf.open()
    for title, footer, slides in DECKS:
        p = doc.new_page()
        p.insert_text((72, 300), title, fontsize=36)
        p.insert_text((72, 760), footer, fontsize=8)
        p.insert_text((540, 780), "1", fontsize=8)
        for n, (h, body) in enumerate(slides, start=2):
            _slide(doc, h, body, footer, n)
    _slide(doc, "Thrashing", DECKS[2][2][5][1], "OS Unit 4", 8)  # repeated slide
    path = tmp_path / "combined_all_units.pdf"
    doc.save(path)
    return path


def _ingest(path: Path, subject: str = "os") -> None:
    pipeline.ingest([path], subject, use_ocr=False, log=lambda m: None)


# ------------------------------------------------------------ pure functions


def _page(i: int, text: str) -> Page:
    return Page(id=i, file_id=1, file_name="f.pdf", kind="pdf", page_no=i, title=None, text=text)


def test_near_duplicates_keeps_fuller_slide():
    base = "deadlock needs mutual exclusion hold and wait no preemption and circular wait conditions"
    pages = [
        _page(1, base),
        _page(2, base + " plus one extra bullet about recovery"),
        _page(3, "paging and frames " * 3),
    ]
    dups = analysis.near_duplicates(pages)
    assert dups == {1: 2}


def test_segment_is_contiguous_and_respects_sizes():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=8), rng.normal(size=8)
    vecs = np.vstack(
        [a + rng.normal(scale=0.05, size=8) for _ in range(10)]
        + [b + rng.normal(scale=0.05, size=8) for _ in range(10)]
    )
    segs = analysis.segment(vecs, [None] * 20)
    assert segs[0][0] == 0 and segs[-1][1] == 19
    assert all(segs[i][1] + 1 == segs[i + 1][0] for i in range(len(segs) - 1))
    assert all(e - s + 1 >= 2 for s, e in segs)
    assert any(s == 10 for s, _ in segs)  # the real topic change is found


def test_boundary_accuracy():
    acc = analysis.boundary_accuracy({"a.pdf": [1, 8, 20]}, {"a.pdf": [1, 8, 15]})
    assert acc["true_positive"] == 1 and acc["false_positive"] == 1 and acc["missed"] == 1


# ------------------------------------------------------------ end to end


def test_build_finds_decks_topics_and_duplicates(combined_pdf: Path):
    _ingest(combined_pdf)
    rep = build.build("os", log=lambda m: None)
    starts = [d["pages"].split("-")[0] for d in rep["decks"]]
    assert starts == ["1", "8", "15"]
    assert rep["duplicate_pages"] == 1
    assert {t["named_by"] for t in rep["topics"]} == {"heading"}
    names = [t["name"] for t in rep["topics"]]
    assert not any("OS Unit" in n for n in names)  # repeated footer lines never become topic names
    assert (Path(build.ledger.subject_dir("os")) / "topics.yaml").exists()
    acc = analysis.boundary_accuracy(build.deck_starts_by_file("os"), {"combined_all_units.pdf": [1, 8, 15]})
    assert acc["f1"] == 1.0


def test_llm_names_and_user_edits_survive_rebuild(combined_pdf: Path, monkeypatch):
    _ingest(combined_pdf)
    monkeypatch.setattr(llm, "available_providers", lambda: ["gemini"])

    def fake_chat(messages, *args, **kw):
        material = messages[-1]["content"].split("<material>", 1)[1].rsplit("</material>", 1)[0]
        items = json.loads(material)
        return json.dumps({"names": [{"id": it["id"], "name": f"LLM topic {it['id']}"} for it in items]})

    monkeypatch.setattr(llm, "chat", fake_chat)
    rep = build.build("os", log=lambda m: None)
    assert all(t["named_by"] == "llm" for t in rep["topics"])

    first = build.topic_map("os")[0]
    build.rename_topic("os", first["id"], "My own name")
    rep = build.build("os", log=lambda m: None)
    assert rep["topics"][0]["name"] == "My own name" and rep["topics"][0]["named_by"] == "user"


def test_yaml_edits(combined_pdf: Path):
    import yaml

    _ingest(combined_pdf)
    build.build("os", log=lambda m: None)
    path = build.ledger.subject_dir("os") / "topics.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert all("sections" in t for t in data)  # headings and points are listed for reference
    data[1]["name"] = "Deadlock Handling"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert build.apply_edits("os") == 1
    assert build.topic_map("os")[1]["name"] == "Deadlock Handling"


def test_syllabus_mapping(combined_pdf: Path, tmp_path: Path):
    _ingest(combined_pdf)
    build.build("os", log=lambda m: None)
    syl = tmp_path / "syllabus.txt"
    syl.write_text(
        "UNIT I: Round robin scheduling\nUNIT III: Banker's algorithm\nUNIT V: File systems and disk scheduling\n",
        encoding="utf-8",
    )
    assert build.parse_syllabus(syl)[0] == ("Unit I", "Round robin scheduling")
    rep = build.map_syllabus("os", syl, log=lambda m: None)
    status = {i["point"]: i["covered"] for i in rep["items"]}
    assert status["Round robin scheduling"] and status["Banker's algorithm"]
    assert not status["File systems and disk scheduling"]


def test_sentence_transformer_vectors_are_cached(monkeypatch, combined_pdf: Path):
    calls = []

    class FakeModel:
        def __init__(self, name):
            pass

        def encode(self, texts, batch_size=64, normalize_embeddings=True):
            calls.append(len(texts))
            return np.array([[len(t) % 7 + 1, len(t) % 5 + 1, 1.0] for t in texts], dtype=np.float32)

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel))
    monkeypatch.setattr(embed, "backend_name", lambda: "sentence-transformers")
    embed._MODEL_CACHE.clear()
    _ingest(combined_pdf)
    build.build("os", log=lambda m: None)
    first = len(calls)
    build.build("os", log=lambda m: None)
    assert len(calls) == first  # second build reuses cached vectors


# ------------------------------------------------------------ LLM client


def test_parse_json_variants():
    assert llm.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert llm.parse_json('Sure! {"a": 2} hope this helps') == {"a": 2}
    with pytest.raises(llm.LLMError):
        llm.parse_json("no json here")


def test_chat_falls_back_to_next_provider(monkeypatch):
    monkeypatch.setattr(llm, "available_providers", lambda: ["gemini", "groq"])
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm.config, "secret", lambda name: "test-key")
    seen = []

    class Resp:
        def __init__(self, code, payload=None):
            self.status_code, self._payload = code, payload

        def json(self):
            return self._payload

    def fake_post(url, json=None, headers=None, timeout=None):
        seen.append(url)
        if "googleapis" in url:
            return Resp(429)
        return Resp(200, {"choices": [{"message": {"content": "ok from groq"}}]})

    monkeypatch.setattr(llm.httpx, "post", fake_post)
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok from groq"
    assert any("groq" in u for u in seen)


def _deck(path: Path, slides: list[tuple[str, str, str]]) -> None:
    """slides: (heading, sub-heading, body). Every deck shares the same cover and closing slide
    (course name, instructor), like real college decks."""
    from pptx import Presentation
    from pptx.util import Inches

    prs = Presentation()

    def add(lines: list[str]) -> None:
        s = prs.slides.add_slide(prs.slide_layouts[6])
        tf = s.shapes.add_textbox(Inches(0.5), Inches(0.5), Inches(9), Inches(6)).text_frame
        tf.text = lines[0]
        for ln in lines[1:]:
            tf.add_paragraph().text = ln

    add(["DATABASE TECHNOLOGIES", "Parallel and Distributed Databases", "Prof. Example Name", "Department of CSE"])
    for head, sub, body in slides:
        add([head, "DATABASE TECHNOLOGIES", sub, body])
    add(["prof@example.edu", "THANK YOU", "Prof. Example Name", "Department of CSE"])
    prs.save(path)


def test_repeated_course_and_instructor_lines_are_ignored(tmp_path: Path):
    d = tmp_path / "unit"
    d.mkdir()
    body = "Each processor works on its own part of the data so queries finish faster on large tables. " * 2
    _deck(
        d / "1.pptx",
        [("Models of Parallelism", "Shared-Memory Machines", body)] * 3
        + [("Query Parallelism", "Inter-Query Parallelism", body.replace("processor", "query"))] * 3,
    )
    _deck(d / "2.pptx", [("Distributed Locking", "Primary-Copy Locking", body.replace("data", "locks"))] * 4)
    _deck(d / "3.pptx", [("Commit Protocols", "Atomic Commit", body.replace("tables", "sites"))] * 4)
    pipeline.ingest([d], "dbms", use_ocr=False, log=lambda m: None)
    rep = build.build("dbms", log=lambda m: None)
    names = [t["name"] for t in rep["topics"]]
    assert names == ["Models of Parallelism", "Query Parallelism", "Distributed Locking", "Commit Protocols"]
    assert rep["topics"][0]["sections"][0]["points"] == ["Shared-Memory Machines"]
    assert rep["filler_pages"] == 6  # cover and closing slides of 3 decks
    assert not any("Example Name" in n or "DATABASE" in n for n in names)


def test_chat_json_skips_provider_with_cut_off_reply(monkeypatch):
    monkeypatch.setattr(llm, "available_providers", lambda: ["gemini", "groq"])
    replies = {"gemini": '{"names": [{"id": 0, "na', "groq": '{"names": []}'}
    monkeypatch.setattr(llm, "chat", lambda messages, *a, providers=None, **k: replies[providers[0]])
    assert llm.chat_json([{"role": "user", "content": "x"}]) == {"names": []}


# ------------------------------------------------------------ other colleges' formats


def _notes_pdf(path: Path) -> None:
    """Typed notes: a running header on every other page, chapter titles in a big font,
    numbered section headings, bold labels as points, long paragraphs."""
    import pymupdf

    para = (
        "This paragraph explains the idea in full sentences the way typed lecture notes do, with enough "
        "words on every page that the file reads as notes and not as slides. "
    ) * 7
    chapters = [
        ("DATABASE DESIGN", "1", ["Keys", "Functional Dependency", "Normal Forms", "Decomposition"]),
        ("ER MODEL", "2", ["Entity Sets", "Participation", "Weak Entity Sets", "Mapping to Tables"]),
        ("TRANSACTIONS AND", "3", ["ACID Properties", "Schedules", "Serializability", "Two Phase Locking"]),
    ]
    doc = pymupdf.open()
    page_no = 0
    for title, num, secs in chapters:
        for k, sec in enumerate(secs):
            p = doc.new_page()
            page_no += 1
            p.insert_text((72, 40), "Data Base Management System" if page_no % 2 else "GATE Handbook", fontsize=9)
            y = 90
            if k == 0:
                p.insert_text((72, y), title, fontsize=30, fontname="hebo")
                y += 36
                if title.endswith("AND"):
                    p.insert_text((72, y), "CONCURRENCY CONTROL", fontsize=24, fontname="hebo")
                    y += 30
            p.insert_text((72, y), f"{num}.{k + 1} {sec}", fontsize=14, fontname="hebo")
            y += 24
            p.insert_text((72, y), "Example:", fontsize=11, fontname="hebo")
            y += 18
            text = para.replace("the idea", f"{sec.lower()} (part {page_no})")
            for line in range(0, len(text), 95):
                p.insert_text((72, y), text[line : line + 95], fontsize=11)
                y += 14
    doc.save(path)


def test_notes_use_chapters_and_numbered_sections(tmp_path: Path):
    path = tmp_path / "notes.pdf"
    _notes_pdf(path)
    _ingest(path, "notes")
    rep = build.build("notes", log=lambda m: None)
    assert rep["decks"][0]["topics_from"] == "chapter headings"
    names = [t["name"] for t in rep["topics"]]
    assert names == ["Database Design", "ER Model", "Transactions and Concurrency Control"]
    first = rep["topics"][0]["sections"]
    assert [s["heading"] for s in first][1:] == ["1.2 Functional Dependency", "1.3 Normal Forms", "1.4 Decomposition"]
    headers = json.dumps(rep["topics"])
    assert "Data Base Management System" not in headers and "GATE Handbook" not in headers  # alternating headers
    assert "Example:" not in [s["heading"] for t in rep["topics"] for s in t["sections"]]


def test_continued_slides_and_split_headings():
    from examprep.structure import sections

    heads = [
        "Hash-Join",
        "Hash-Join (Cont.)",
        "Other Operations",
        "Other Operations : Aggregation",
        "Chapter 15: Query Processing",
        "Chapter 15: Query Processing",
        "Example: External Sorting",
    ]
    assert sections.base_headings(heads) == [
        "Hash-Join",
        "Hash-Join",
        "Other Operations",
        "Other Operations",
        "Chapter 15: Query Processing",
        "Chapter 15: Query Processing",
        "Example: External Sorting",
    ]
    secs = sections.build_sections(heads[:4], [None] * 4, [False] * 4)
    assert [(s.heading, s.size) for s in secs] == [("Hash-Join", 2), ("Other Operations", 2)]
    assert secs[1].subheadings == ["Aggregation"]

    style = sections.FileStyle(body_size=20.0, body_bold=False, notes=False)
    page = Page(
        1,
        1,
        "f.pdf",
        "pdf",
        1,
        None,
        "",
        layout=[
            ["DATABASES AND", 40, False],
            ["DATABASE USERS", 40, False],
            ["Body text that explains the slide in a sentence.", 20, False],
        ],
    )
    head, _, _ = sections.headings(page, sections.page_lines(page), style, set())
    assert head == "DATABASES AND DATABASE USERS"


def test_scanned_page_layout_from_ocr_is_stored(tmp_path: Path, monkeypatch):
    import pymupdf

    from examprep.ingest import ocr

    path = tmp_path / "scan.pdf"
    doc = pymupdf.open()
    for _ in range(2):
        doc.new_page()  # no text layer: needs whole-page OCR
    doc.save(path)
    layout = [["2.3 Dual Space-Time Representation", 28.0, None], ["Query is transformed into a polygon.", 18.0, None]]
    monkeypatch.setattr(ocr, "tesseract_available", lambda: True)
    monkeypatch.setattr(
        ocr,
        "run_jobs",
        lambda jobs, progress=None: {j.key: ("\n".join(x[0] for x in layout), None, layout) for j in jobs},
    )
    pipeline.ingest([path], "scan", log=lambda m: None)
    con = build.ledger.connect("scan")
    rows = con.execute("SELECT ocr_text, layout_json FROM pages").fetchall()
    con.close()
    assert all("\n" in r["ocr_text"] and json.loads(r["layout_json"]) == layout for r in rows)
