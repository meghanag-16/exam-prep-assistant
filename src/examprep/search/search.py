"""Search a subject's index: keyword, vector, hybrid (merged with reciprocal rank
fusion) or hybrid plus reranking. Every result carries its file and page numbers.

The chunking and mode used by default are the best ones found by examprep-eval
(data/<subject>/search_settings.json), otherwise the defaults in config/search.yaml.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field

from examprep import config, tracing
from examprep.ingest import ledger
from examprep.search import chunks as chunking
from examprep.search import index


def settings_path(subject: str):
    return ledger.subject_dir(subject) / "search_settings.json"


def default_settings(subject: str) -> dict:
    cfg = config.search()["search"]
    out = {"chunking": cfg.get("default_chunking", "heading"), "mode": cfg.get("default_mode", "hybrid_rerank")}
    p = settings_path(subject)
    if p.exists():
        saved = json.loads(p.read_text(encoding="utf-8"))
        out.update({k: saved[k] for k in ("chunking", "mode") if saved.get(k)})
    return out


@dataclass
class Hit:
    chunk_id: int
    score: float
    text: str
    pages: list[dict]  # [{"page_id", "file", "page_no"}]
    topic: str | None
    heading: str | None
    formula_ids: list[int] = field(default_factory=list)
    rerank_score: float | None = None

    def as_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 4),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4),
            "topic": self.topic,
            "heading": self.heading,
            "pages": self.pages,
            "where": where(self.pages),
            "text": self.text,
            "formula_ids": self.formula_ids,
        }


def _ranges(nums: list[int]) -> str:
    """[4, 5, 6, 9] -> "4-6, 9"."""
    out: list[str] = []
    start: int | None = None
    prev = 0
    for n in sorted(set(nums)):
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if prev != start else str(start))
            start = prev = n
    if start is not None:
        out.append(f"{start}-{prev}" if prev != start else str(start))
    return ", ".join(out)


def where(pages: list[dict]) -> str:
    """'1parallel.pptx p.4-6' (file name without its folder, page ranges)."""
    by_file: dict[str, list[int]] = {}
    for p in pages:
        by_file.setdefault(p["name"], []).append(p["page_no"])
    return "; ".join(f"{name} p.{_ranges(nums)}" for name, nums in by_file.items())


def _join_text(a: str, b: str) -> str:
    """Join two chunk texts, dropping the words they share (overlapping windows) and a
    repeated first line (the topic and heading line)."""
    a_lines, b_lines = a.split("\n"), b.split("\n")
    if len(b_lines) > 1 and b_lines[0] == a_lines[0]:
        b = "\n".join(b_lines[1:])
    aw, bw = a.split(), b.split()
    for m in range(min(len(aw), len(bw), 200), 2, -1):
        if aw[-m:] == bw[:m]:
            return a + " " + " ".join(bw[m:])
    if b in a:
        return a
    return a + "\n...\n" + b


def merge_overlapping(hits: list[Hit], k: int, max_pages: int = 4) -> list[Hit]:
    """Results that share a page, or are neighbouring pages of the same topic, become one
    result covering all their pages, at the rank and score of the best of them
    (overlapping windows, a section split in two, an explanation over three slides). Only
    results ranked in the first 2k are joined, and a result never grows past max_pages
    pages; a lower overlapping result is simply left out, so it is not shown twice."""

    def touches(h: Hit, g: Hit) -> bool:
        if {p["page_id"] for p in h.pages} & {p["page_id"] for p in g.pages}:
            return True
        # Neighbouring pages of the same file and topic (pages 4, 5 and 6 of one explanation).
        return h.topic == g.topic and any(
            a["file"] == b["file"] and abs(a["page_no"] - b["page_no"]) == 1 for a in h.pages for b in g.pages
        )

    groups: list[Hit] = []
    for rank, h in enumerate(hits):
        ids = {p["page_id"] for p in h.pages}
        target = next((g for g in groups if touches(h, g)), None)
        if target is not None:
            shared = bool(ids & {p["page_id"] for p in target.pages})
            pages_after = len(ids | {p["page_id"] for p in target.pages})
            if shared and pages_after == len({p["page_id"] for p in target.pages}):
                continue  # adds no page: already shown
            if rank >= 2 * k or pages_after > max_pages:
                if shared:
                    continue  # same pages again: not shown twice
                target = None  # only a neighbour: shown as its own result
        if target is None:
            groups.append(
                Hit(h.chunk_id, h.score, h.text, list(h.pages), h.topic, h.heading, list(h.formula_ids), h.rerank_score)
            )
            continue
        known = {p["page_id"] for p in target.pages}
        target.pages = sorted(
            target.pages + [p for p in h.pages if p["page_id"] not in known], key=lambda p: (p["file"], p["page_no"])
        )
        # Keep the text in page order: the chunk that starts earlier goes first.
        first_new = min((p["page_no"] for p in h.pages), default=0)
        first_old = min((p["page_no"] for p in target.pages if p["page_id"] in known), default=0)
        target.text = _join_text(h.text, target.text) if first_new < first_old else _join_text(target.text, h.text)
        target.formula_ids += [f for f in h.formula_ids if f not in target.formula_ids]
    return groups


def min_rerank_score(subject: str) -> float | None:
    """Reranker scores below this are dropped: the threshold examprep-eval found for this
    subject, else config/search.yaml min_rerank_score."""
    p = settings_path(subject)
    if p.exists():
        saved = json.loads(p.read_text(encoding="utf-8")).get("refusal_threshold")
        if saved is not None:
            return float(saved)
    value = config.search()["search"].get("min_rerank_score")
    return None if value is None else float(value)


def relative_cut(subject: str) -> float | None:
    """Results whose reranker score is below this share of the best result's score are
    dropped (they share words with the question but answer something else): the value
    examprep-eval measured for this subject, else config/search.yaml min_relative_score."""
    cfg = config.search()["search"]
    p = settings_path(subject)
    if p.exists():
        saved = json.loads(p.read_text(encoding="utf-8")).get("relative_cut")
        if saved is not None:
            # Results almost as good as the best are always kept, whatever was measured.
            return min(float(saved), float(cfg.get("relative_cut_max", 0.8)))
    value = cfg.get("min_relative_score")
    return None if value is None else float(value)


class Searcher:
    """Holds the index, models and chunk table open, so many questions are cheap (evaluation)."""

    def __init__(self, subject: str, encoder=None, reranker=None):
        self.subject = ledger.safe_subject(subject)
        if not ledger.db_path(self.subject).exists():
            raise FileNotFoundError(f"No ledger for '{self.subject}'. Run examprep-ingest first.")
        meta = index.read_meta(self.subject)
        if not meta:
            raise FileNotFoundError(f"No search index for '{self.subject}'. Run examprep-index first.")
        self.meta = meta
        self.encoder = encoder if encoder is not None else (None if meta.get("keyword_only") else index.encoder())
        self._reranker = reranker
        self.con = ledger.connect(self.subject)
        chunking.migrate(self.con)
        self.qc = index.client(self.subject)
        self.pages = {
            r["id"]: {
                "page_id": r["id"],
                "file": r["name"],
                "name": r["name"].replace("\\", "/").rsplit("/", 1)[-1],
                "page_no": r["page_no"],
            }
            for r in self.con.execute("SELECT p.id, p.page_no, f.name FROM pages p JOIN files f ON f.id = p.file_id")
        }
        self.topics = {r[0]: r[1] for r in self.con.execute("SELECT id, name FROM topics")}
        self._rows: dict[int, sqlite3.Row] = {}

    def close(self) -> None:
        self.con.close()

    def reranker(self):
        if self._reranker is None:
            self._reranker = index.reranker() or False
        return self._reranker or None

    def _row(self, cid: int) -> sqlite3.Row:
        if cid not in self._rows:
            self._rows[cid] = self.con.execute("SELECT * FROM chunks WHERE id = ?", (cid,)).fetchone()
        return self._rows[cid]

    def _hit(self, cid: int, score: float) -> Hit:
        r = self._row(cid)
        return Hit(
            chunk_id=cid,
            score=float(score),
            text=r["text"],
            pages=[self.pages[p] for p in json.loads(r["page_ids"]) if p in self.pages],
            topic=self.topics.get(r["topic_id"]),
            heading=r["heading"],
            formula_ids=json.loads(r["formula_ids"] or "[]"),
        )

    def search(
        self,
        query: str,
        chunking_method: str | None = None,
        mode: str | None = None,
        k: int | None = None,
        cut: bool = True,
    ):
        """Returns (hits, info). info says which mode really ran and how long it took.
        Results sharing a page are merged into one. With cut (and the reranker), results
        scoring below the subject's threshold are dropped, so weak matches are not shown."""
        from qdrant_client import models as qm

        defaults = default_settings(self.subject)
        method = chunking_method or defaults["chunking"]
        mode = mode or defaults["mode"]
        if mode not in index.MODES:
            raise ValueError(f"unknown mode '{mode}' (use one of {', '.join(index.MODES)})")
        if method not in self.meta.get("methods", {}):
            raise ValueError(f"chunking '{method}' is not in the index (built: {', '.join(self.meta['methods'])})")
        cfg = config.search()["search"]
        k = int(k or cfg.get("top_k", 5))
        cand = max(k, int(cfg.get("candidates", 30)))
        notes = []
        ran = mode
        if mode != "keyword" and self.encoder is None:
            notes.append("No embedding model here (sentence-transformers not installed): keyword search was used.")
            ran = "keyword"
        started = time.perf_counter()
        name = index.collection(method)
        idx, vals = index.query_vector(query)
        sparse = qm.SparseVector(indices=idx, values=vals)
        if ran == "keyword":
            pts = self.qc.query_points(name, query=sparse, using="bm25", limit=cand).points if idx else []
        else:
            assert self.encoder is not None  # ran is keyword without an encoder
            dense = self.encoder.query(query).tolist()
            if ran == "vector":
                pts = self.qc.query_points(name, query=dense, using="dense", limit=cand).points
            else:
                prefetch = [qm.Prefetch(query=dense, using="dense", limit=cand)]
                if idx:
                    prefetch.append(qm.Prefetch(query=sparse, using="bm25", limit=cand))
                pts = self.qc.query_points(
                    name, prefetch=prefetch, query=qm.FusionQuery(fusion=qm.Fusion.RRF), limit=cand
                ).points
        hits = [self._hit(int(p.id), p.score) for p in pts]
        if ran == "hybrid_rerank" and hits:
            score = self.reranker()
            if score is None:
                notes.append("No reranker here (sentence-transformers not installed): hybrid results not reranked.")
                ran = "hybrid"
            else:
                top = hits[: int(cfg.get("rerank_top", 20))]
                for h, s in zip(top, score(query, [h.text for h in top]), strict=True):
                    h.rerank_score = s
                hits = sorted(top, key=lambda h: h.rerank_score or 0.0, reverse=True) + hits[len(top) :]
                if cut:
                    floor = min_rerank_score(self.subject)
                    share = relative_cut(self.subject)
                    best = hits[0].rerank_score if hits and hits[0].rerank_score is not None else 0.0
                    need = max(floor or 0.0, (share or 0.0) * best)
                    if floor is not None or share is not None:
                        kept = [h for h in hits if h.rerank_score is not None and h.rerank_score >= need]
                        if len(kept) < len(hits):
                            notes.append(f"Weaker matches (reranker score below {round(need, 4)}) were left out.")
                        hits = kept
        hits = merge_overlapping(hits, k, int(cfg.get("merge_max_pages", 4)))
        with tracing.observe("search-material", as_type="retriever", input=query) as obs:
            tracing.update(
                obs,
                output=[
                    {
                        "where": where(h.pages),
                        "topic": h.topic,
                        "score": h.rerank_score if h.rerank_score is not None else h.score,
                        "text": h.text,
                    }
                    for h in hits[:k]
                ],
                metadata={
                    "subject": self.subject,
                    "mode": ran,
                    "chunking": method,
                    "results": len(hits[:k]),
                    "ms": round((time.perf_counter() - started) * 1000, 1),
                },
            )
        info = {
            "chunking": method,
            "mode": ran,
            "asked_mode": mode,
            "ms": round((time.perf_counter() - started) * 1000, 1),
            "notes": notes,
        }
        return hits[:k], info


def search(subject: str, query: str, chunking_method: str | None = None, mode: str | None = None, k: int | None = None):
    """One search: {"results": [...], "chunking", "mode", "ms", "notes"}."""
    s = Searcher(subject)
    try:
        hits, info = s.search(query, chunking_method, mode, k)
        return {**info, "query": query, "results": [h.as_dict() for h in hits]}
    finally:
        s.close()
