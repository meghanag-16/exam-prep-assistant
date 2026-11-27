"""Chunking, the Qdrant index, hybrid search with reranking, the evaluation set and the
retrieval evaluation. Models are faked (no downloads, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from examprep import llm
from examprep.ingest import ledger, pipeline
from examprep.search import chunks, evalset, evaluate, index
from examprep.search import search as searching
from examprep.structure import build, embed

SLIDES = [
    ("Processes", "A process is a program in execution with its own address space and a process control block."),
    ("Context Switch", "The kernel saves registers into the process control block during a context switch."),
    ("Round Robin", "Round robin scheduling gives each process a time quantum taken from the ready queue."),
    ("Deadlock Conditions", "Mutual exclusion, hold and wait, no preemption and circular wait cause deadlock."),
    ("Bankers Algorithm", "The banker's algorithm checks for a safe state using available, max and need matrices."),
    ("Deadlock Recovery", "Recovery terminates processes or preempts resources to break the deadlock cycle."),
    ("Paging", "Paging divides memory into fixed size frames and pages, mapped by a page table."),
    ("Thrashing", "Thrashing happens when processes spend more time paging than executing instructions."),
]


class FakeEncoder:
    """Bag-of-words vectors over a fixed vocabulary: similar words, similar vectors."""

    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.hv = HashingVectorizer(n_features=512, alternate_sign=False, norm="l2", stop_words="english")
        self.name = "fake-encoder"
        self.calls = 0

    def docs(self, texts, con):
        self.calls += len(texts)
        return self.hv.transform(texts).toarray().astype(np.float32)

    def query(self, text):
        return self.hv.transform([text]).toarray()[0].astype(np.float32)


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("EXAMPREP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(embed, "backend_name", lambda: "tfidf")
    monkeypatch.setattr(llm, "available_providers", lambda: [])
    enc = FakeEncoder()
    monkeypatch.setattr(index, "encoder", lambda: index.Encoder(enc.name, enc.docs, enc.query))
    monkeypatch.setattr(index, "reranker", lambda: None)
    yield enc
    index.close()


@pytest.fixture
def subject(tmp_path: Path) -> str:
    import pymupdf

    doc = pymupdf.open()
    p = doc.new_page()
    p.insert_text((72, 300), "Operating Systems", fontsize=36)
    for n, (head, body) in enumerate(SLIDES, start=2):
        p = doc.new_page()
        p.insert_text((72, 72), head, fontsize=22)
        y = 120
        for k in range(4):
            p.insert_text((72, y), body if k == 0 else f"Point {k}: " + body.lower(), fontsize=11)
            y += 22
        p.insert_text((540, 780), str(n), fontsize=8)
    path = tmp_path / "os_slides.pdf"
    doc.save(path)
    pipeline.ingest([path], "os", use_ocr=False, log=lambda m: None)
    build.build("os", log=lambda m: None)
    return "os"


def test_keyword_tokens_and_bm25():
    assert index.tokens("The Banker's algorithms use C++ and B+ trees") == [
        "banker",
        "algorithm",
        "use",
        "c++",
        "b+",
        "tree",
    ]
    vecs = index.bm25_vectors(["deadlock deadlock safe state", "paging frames", "deadlock recovery"])
    weight = dict(zip(*vecs[0], strict=True))
    rare, common = index._term_id("safe"), index._term_id("deadlock")
    assert weight[common] > 0 and weight[rare] > 0
    # "safe" is in one chunk, "deadlock" in two: the rarer term weighs more per occurrence.
    assert weight[rare] > weight[common] / 2


def test_three_chunking_methods_keep_pages_and_topics(subject: str):
    con = ledger.connect(subject)
    try:
        chunks.migrate(con)
        pages = chunks.load_pages(con)
        by_page = chunks.make_chunks(con, "page", pages)
        window = chunks.make_chunks(con, "window", pages)
        heading = chunks.make_chunks(con, "heading", pages)
    finally:
        con.close()
    content = [p for p in pages if not p.skip]
    assert len(by_page) == len(content) and all(len(c.page_ids) == 1 for c in by_page)
    assert all(c.topic_id for c in by_page)
    # Windows run over page breaks but never mix topics.
    assert any(len(c.page_ids) > 1 for c in window)
    topic_of = {p.id: p.topic_id for p in pages}
    assert all(len({topic_of[i] for i in c.page_ids}) == 1 for c in window)
    assert {i for c in heading for i in c.page_ids} == {p.id for p in content}
    # The cover page is filler: never a chunk.
    assert pages[0].skip and all(pages[0].id not in c.page_ids for c in by_page + window + heading)


def test_index_and_all_search_modes(subject: str, isolated):
    rep = index.build(subject, log=lambda m: None)
    assert set(rep["methods"]) == {"page", "window", "heading"} and rep["model"] == "fake-encoder"
    first_calls = isolated.calls
    assert not index.is_stale(subject)

    for mode in ("keyword", "vector", "hybrid"):
        out = searching.search(subject, "what is the bankers algorithm safe state", "page", mode, 3)
        assert out["mode"] == mode and out["results"]
        assert out["results"][0]["pages"][0]["page_no"] == 6, mode
    # Without a reranker the reranking mode says so and falls back to hybrid.
    out = searching.search(subject, "bankers algorithm", "page", "hybrid_rerank", 3)
    assert out["mode"] == "hybrid" and out["notes"]

    # Embeddings are cached in the ledger: rebuilding embeds nothing again.
    index.build(subject, log=lambda m: None)
    assert isolated.calls == first_calls


def test_reranker_reorders_results(subject: str, monkeypatch):
    index.build(subject, ["page"], log=lambda m: None)

    def rerank(query, texts):
        return [10.0 if "Thrashing" in t else 0.0 for t in texts]

    s = searching.Searcher(subject, reranker=rerank)
    try:
        hits, info = s.search("time spent paging memory frames", "page", "hybrid_rerank", 3)
    finally:
        s.close()
    assert info["mode"] == "hybrid_rerank"
    assert hits[0].pages[0]["page_no"] == 9 and hits[0].rerank_score == 10.0


def test_keyword_only_index_without_embedding_model(subject: str, monkeypatch):
    monkeypatch.setattr(index, "encoder", lambda: None)
    rep = index.build(subject, ["page"], log=lambda m: None)
    assert rep["keyword_only"]
    out = searching.search(subject, "round robin quantum", "page", "hybrid", 2)
    assert out["mode"] == "keyword" and out["results"][0]["pages"][0]["page_no"] == 4


def test_evalset_draft_and_evaluation(subject: str, monkeypatch):
    def fake_chat_json(messages, **kw):
        content = messages[-1]["content"]
        if "<material>" in content:
            material = json.loads(content.split("<material>")[1].split("</material>")[0])
            asked = []
            for item in material:
                head = next((h for h, b in SLIDES if b[:30] in item["text"]), None)
                asked.append({"id": item["id"], "question": f"Explain {head.lower()}" if head else ""})
            return {"items": asked}
        return {"questions": ["What is a binary search tree?", "Explain photosynthesis."]}

    monkeypatch.setattr(llm, "available_providers", lambda: ["gemini"])
    monkeypatch.setattr(llm, "chat_json", fake_chat_json)
    rep = evalset.generate(subject, count=6, outside=2, log=lambda m: None)
    assert rep["questions"] == 6 and rep["outside"] == 2
    data = evalset.load(subject)
    assert all(q["pages"][0]["file"] == "os_slides.pdf" for q in data["questions"])

    # A hand-written question with a folder in the file name, and one with a page that does not exist.
    data["questions"].append({"id": "h1", "question": "thrashing", "pages": [{"file": "x/os_slides.pdf", "page": 9}]})
    data["questions"].append({"id": "h2", "question": "nothing", "pages": [{"file": "os_slides.pdf", "page": 99}]})
    evalset.save(subject, data)
    assert yaml.safe_load(evalset.path(subject).read_text(encoding="utf-8"))["questions"][-1]["id"] == "h2"

    index.build(subject, log=lambda m: None)
    report = evaluate.run(subject, save_best=True, log=lambda m: None)
    assert report["questions"] == 7 and report["skipped"] == ["h2"]
    assert len(report["results"]) == 3 * 4
    vector = next(r for r in report["results"] if (r["chunking"], r["mode"]) == ("page", "vector"))
    assert vector["recall@5"] == 1.0 and vector["mrr"] > 0.5
    rerank = next(r for r in report["results"] if r["mode"] == "hybrid_rerank")
    assert rerank["ran_as"] == "hybrid"  # no reranker in tests: not chosen as best
    assert report["best"]["mode"] != "hybrid_rerank"
    saved = json.loads(searching.settings_path(subject).read_text(encoding="utf-8"))
    assert (saved["chunking"], saved["mode"]) == (report["best"]["chunking"], report["best"]["mode"])
    assert searching.default_settings(subject)["mode"] == report["best"]["mode"]
    md = (ledger.subject_dir(subject) / "eval" / "retrieval_report.md").read_text(encoding="utf-8")
    assert "| page | vector |" in md


def test_refusal_threshold():
    t = evaluate.threshold([0.9, 0.8, 0.85, 0.4], [0.2, 0.3, 0.5])
    assert t["accuracy"] >= 0.85 and 0.5 <= t["threshold"] <= 0.8
    assert evaluate.threshold([0.5], []) is None


def test_search_api(subject: str):
    from fastapi.testclient import TestClient

    from examprep.api.main import app

    client = TestClient(app)
    assert client.get("/search/os", params={"q": "paging"}).status_code == 404  # no index yet
    index.build(subject, ["heading"], log=lambda m: None)
    r = client.get("/search/os", params={"q": "page table frames", "chunking": "heading", "mode": "keyword"})
    assert r.status_code == 200 and r.json()["results"][0]["where"].startswith("os_slides.pdf p.")
    assert client.get("/search/os", params={"q": "x", "chunking": "page"}).status_code == 400
