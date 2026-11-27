"""The search index: an embedded Qdrant database in data/<subject>/index/qdrant.

One collection per chunking method, each point with two vectors:
- "dense":  the chunk's embedding (bge-small, config/models.yaml). Embeddings are
            cached in the ledger by text hash, so unchanged chunks are never embedded again.
- "bm25":   a sparse keyword vector with BM25 weights, for exact terms (names, acronyms,
            symbols) that embeddings miss.

Search modes:
- keyword:        BM25 only.
- vector:         embeddings only.
- hybrid:         both lists merged with reciprocal rank fusion (Qdrant's RRF).
- hybrid_rerank:  hybrid, then the top results re-scored by a cross-encoder
                  (bge-reranker-base), which reads the question and passage together.

Embeddings and the reranker need sentence-transformers (the `ml` extra, installed on
Kaggle and the Space). Without it (the laptop) the index is keyword-only and the
other modes fall back to keyword search, which the results say.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import math
import re
import shutil
import sqlite3
import time
import zlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from examprep import config
from examprep.ingest import ledger
from examprep.search import chunks as chunking

MODES = ("keyword", "vector", "hybrid", "hybrid_rerank")
_CLIENTS: dict[str, Any] = {}

# scikit-learn's English stop words (BSD licence), copied here so that a search does not
# have to import scikit-learn, which takes a second or more on a laptop.
STOP_WORDS = frozenset(
    (
        "a about above across after afterwards again against all almost alone along already also although "
        "always am among amongst amoungst amount an and another any anyhow anyone anything anyway anywhere "
        "are around as at back be became because become becomes becoming been before beforehand behind being "
        "below beside besides between beyond bill both bottom but by call can cannot cant co con could "
        "couldnt cry de describe detail do done down due during each eg eight either eleven else elsewhere "
        "empty enough etc even ever every everyone everything everywhere except few fifteen fifty fill find "
        "fire first five for former formerly forty found four from front full further get give go had has "
        "hasnt have he hence her here hereafter hereby herein hereupon hers herself him himself his how "
        "however hundred i ie if in inc indeed interest into is it its itself keep last latter latterly least "
        "less ltd made many may me meanwhile might mill mine more moreover most mostly move much must my "
        "myself name namely neither never nevertheless next nine no nobody none noone nor not nothing now "
        "nowhere of off often on once one only onto or other others otherwise our ours ourselves out over own "
        "part per perhaps please put rather re same see seem seemed seeming seems serious several she should "
        "show side since sincere six sixty so some somehow someone something sometime sometimes somewhere "
        "still such system take ten than that the their them themselves then thence there thereafter thereby "
        "therefore therein thereupon these they thick thin third this those though three through throughout "
        "thru thus to together too top toward towards twelve twenty two un under until up upon us very via "
        "was we well were what whatever when whence whenever where whereafter whereas whereby wherein "
        "whereupon wherever whether which while whither who whoever whole whom whose why will with within "
        "without would yet you your yours yourself yourselves "
    ).split()
)


def _stopwords() -> frozenset[str]:
    return STOP_WORDS


def tokens(text: str) -> list[str]:
    """Lower-case words and numbers; stop words dropped; a plural 's' removed."""
    out = []
    for t in re.findall(r"[a-z0-9]+(?:[+#][a-z0-9+#]*)?", text.lower()):
        if t in _stopwords() or (len(t) == 1 and not t.isdigit()):
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


def _term_id(term: str) -> int:
    return zlib.crc32(term.encode()) & 0x7FFFFFFF


def bm25_vectors(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    """BM25 weights per chunk (IDF over this collection), as sparse vectors."""
    cfg = config.search()["search"]
    k1, b = float(cfg.get("bm25_k1", 1.2)), float(cfg.get("bm25_b", 0.75))
    docs = [Counter(tokens(t)) for t in texts]
    n = len(docs)
    avg = sum(sum(d.values()) for d in docs) / max(n, 1) or 1.0
    df = Counter(term for d in docs for term in d)
    out = []
    for d in docs:
        length = sum(d.values())
        weights: dict[int, float] = {}
        for term, tf in d.items():
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            w = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg))
            tid = _term_id(term)
            weights[tid] = weights.get(tid, 0.0) + w
        out.append((list(weights), list(weights.values())))
    return out


def query_vector(text: str) -> tuple[list[int], list[float]]:
    counts = Counter(_term_id(t) for t in tokens(text))
    return list(counts), [float(c) for c in counts.values()]


# ------------------------------------------------------------ models


@dataclass
class Encoder:
    name: str
    docs: Callable[[list[str], sqlite3.Connection | None], np.ndarray]
    query: Callable[[str], np.ndarray]


def encoder() -> Encoder | None:
    """The embedding model, or None when sentence-transformers is not installed."""
    from examprep.structure import embed

    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return None
    prefix = str(config.search()["search"].get("query_prefix", ""))
    return Encoder(
        name=config.models()["embeddings"]["model"],
        docs=lambda texts, con: embed._st_embed(texts, None),
        query=lambda q: embed._st_embed([prefix + q], None)[0],
    )


def cached_vectors(enc: Encoder, texts: list[str], con: sqlite3.Connection) -> np.ndarray:
    """Chunk embeddings, cached in the ledger by text hash and model: only new text is embedded."""
    keys = [hashlib.sha256(t.encode("utf-8", "ignore")).hexdigest() for t in texts]
    have = {
        k: np.frombuffer(blob, dtype=np.float32)
        for k, blob in con.execute("SELECT hash, vector FROM embeddings WHERE model = ?", (enc.name,))
    }
    todo = [i for i, k in enumerate(keys) if k not in have]
    if todo:
        new = np.asarray(enc.docs([texts[i] for i in todo], None), dtype=np.float32)
        for i, v in zip(todo, new, strict=True):
            have[keys[i]] = v
            con.execute(
                "INSERT OR REPLACE INTO embeddings (hash, model, vector) VALUES (?,?,?)",
                (keys[i], enc.name, v.tobytes()),
            )
        con.commit()
    return np.vstack([have[k] for k in keys])


_RERANKER: dict[str, Any] = {}


def reranker() -> Callable[[str, list[str]], list[float]] | None:
    """Cross-encoder scores for (question, passage) pairs, or None without sentence-transformers."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return None
    name = config.models()["reranker"]["model"]

    def score(query: str, texts: list[str]) -> list[float]:
        if name not in _RERANKER:
            _RERANKER[name] = CrossEncoder(name, max_length=512)
        model = _RERANKER[name]
        return [float(s) for s in model.predict([(query, t) for t in texts])]

    return score


# ------------------------------------------------------------ storage


def index_dir(subject: str) -> Path:
    return ledger.subject_dir(subject) / "index"


def meta_path(subject: str) -> Path:
    return index_dir(subject) / "meta.json"


def read_meta(subject: str) -> dict | None:
    p = meta_path(subject)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def client(subject: str):
    """The subject's embedded Qdrant (one per process: Qdrant locks its folder)."""
    from qdrant_client import QdrantClient

    key = str(index_dir(subject) / "qdrant")
    if key not in _CLIENTS:
        Path(key).mkdir(parents=True, exist_ok=True)
        _CLIENTS[key] = QdrantClient(path=key)
    return _CLIENTS[key]


def close(subject: str | None = None) -> None:
    for key in list(_CLIENTS):
        if subject is None or key == str(index_dir(subject) / "qdrant"):
            try:
                _CLIENTS.pop(key).close()
            except Exception:
                pass


atexit.register(close)


def collection(method: str) -> str:
    return f"chunks_{method}"


def fingerprint(con: sqlite3.Connection, model: str | None) -> str:
    """Changes when pages, the topic map, formulas, chunk settings or the model change."""
    h = hashlib.sha256()
    for (x,) in con.execute("SELECT IFNULL(p.hash, '') FROM pages p ORDER BY p.id"):
        h.update(x.encode())
    for row in con.execute("SELECT heading, start_page, end_page FROM sections ORDER BY id"):
        h.update(repr(tuple(row)).encode())
    if con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'formulas'").fetchone():
        h.update(repr(con.execute("SELECT COUNT(*), IFNULL(MAX(id), 0) FROM formulas").fetchone()[:]).encode())
    h.update(json.dumps(config.search()["chunking"], sort_keys=True).encode())
    h.update((model or "keyword-only").encode())
    return h.hexdigest()[:24]


def build(subject: str, methods: list[str] | None = None, log: Callable[[str], None] = print) -> dict:
    """Chunk the subject's pages with each method and (re)build the index."""
    from qdrant_client import models as qm

    started = time.time()
    subject = ledger.safe_subject(subject)
    if not ledger.db_path(subject).exists():
        raise FileNotFoundError(f"No ledger for '{subject}'. Run examprep-ingest first.")
    methods = list(methods or config.search()["chunking"].get("methods", chunking.METHODS))
    enc = encoder()
    con = ledger.connect(subject)
    close(subject)
    shutil.rmtree(index_dir(subject) / "qdrant", ignore_errors=True)
    report: dict = {"subject": subject, "model": enc.name if enc else None, "methods": {}}
    try:
        chunking.migrate(con)
        pages = chunking.load_pages(con)
        if not con.execute("SELECT 1 FROM sections LIMIT 1").fetchone():
            log("No topic map yet (run examprep-topics): heading chunks fall back to one per page.")
        qc = client(subject)
        for method in methods:
            t0 = time.time()
            found = chunking.make_chunks(con, method, pages)
            ids = chunking.save(con, method, found)
            texts = [c.text for c in found]
            dense = cached_vectors(enc, texts, con) if enc and texts else None
            sparse = bm25_vectors(texts)
            vectors_config = (
                {"dense": qm.VectorParams(size=int(dense.shape[1]), distance=qm.Distance.COSINE)}
                if dense is not None
                else {}
            )
            qc.create_collection(
                collection(method),
                vectors_config=vectors_config,
                sparse_vectors_config={"bm25": qm.SparseVectorParams()},
            )
            points = []
            for n, (cid, c) in enumerate(zip(ids, found, strict=True)):
                vec: dict = {"bm25": qm.SparseVector(indices=sparse[n][0], values=sparse[n][1])}
                if dense is not None:
                    vec["dense"] = dense[n].tolist()
                points.append(qm.PointStruct(id=cid, vector=vec, payload={"topic_id": c.topic_id}))
            for i in range(0, len(points), 256):
                qc.upsert(collection(method), points[i : i + 256])
            words = [c.words for c in found]
            report["methods"][method] = {
                "chunks": len(found),
                "avg_words": round(sum(words) / len(words), 1) if words else 0,
                "avg_pages": round(sum(len(c.page_ids) for c in found) / len(found), 2) if found else 0,
                "seconds": round(time.time() - t0, 1),
            }
            log(f"  {method}: {len(found)} chunks")
        report["fingerprint"] = fingerprint(con, report["model"])
        report["built_at"] = ledger.now()
        report["seconds"] = round(time.time() - started, 1)
        report["keyword_only"] = enc is None
        meta_path(subject).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        con.close()


def is_stale(subject: str) -> bool:
    meta = read_meta(subject)
    if not meta:
        return True
    con = ledger.connect(subject)
    try:
        chunking.migrate(con)
        return meta.get("fingerprint") != fingerprint(con, meta.get("model"))
    finally:
        con.close()
