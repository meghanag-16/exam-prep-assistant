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

from examprep import config
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
            "where": ", ".join(f"{p['file']} p.{p['page_no']}" for p in self.pages),
            "text": self.text,
            "formula_ids": self.formula_ids,
        }


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
            r["id"]: {"page_id": r["id"], "file": r["name"], "page_no": r["page_no"]}
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

    def search(self, query: str, chunking_method: str | None = None, mode: str | None = None, k: int | None = None):
        """Returns (hits, info). info says which mode really ran and how long it took."""
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
                hits = sorted(top, key=lambda h: h.rerank_score, reverse=True) + hits[len(top) :]
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
