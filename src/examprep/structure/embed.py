"""Turn page text into vectors for similarity.

Two backends:
- sentence-transformers with the model in config/models.yaml (better quality).
  Installed on Kaggle and the Space via the `ml` extra; vectors are cached in
  the ledger by text hash, so each page is embedded once.
- TF-IDF + LSA from scikit-learn (no model download, runs anywhere). Used on
  the laptop, where nothing heavy is installed.
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Any

import numpy as np

from examprep import config


def backend_name() -> str:
    want = config.structure()["embeddings"].get("backend", "auto")
    if want == "tfidf":
        return "tfidf"
    try:
        import sentence_transformers  # noqa: F401

        return "sentence-transformers"
    except ImportError:
        if want == "sentence-transformers":
            raise RuntimeError(
                "sentence-transformers is not installed. Install the ml extra (pip install -e .[ml]) "
                "or set embeddings.backend to auto in config/structure.yaml."
            ) from None
        return "tfidf"


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


_MODEL_CACHE: dict[str, Any] = {}


def _st_embed(texts: list[str], con: sqlite3.Connection | None) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    name = config.models()["embeddings"]["model"]
    batch = int(config.models()["embeddings"].get("batch_size", 64))
    keys = [hashlib.sha256(t.encode("utf-8", "ignore")).hexdigest() for t in texts]
    cached: dict[str, np.ndarray] = {}
    if con is not None:
        for k, blob in con.execute("SELECT hash, vector FROM embeddings WHERE model = ?", (name,)):
            cached[k] = np.frombuffer(blob, dtype=np.float32)
    todo = [i for i, k in enumerate(keys) if k not in cached]
    if todo:
        if name not in _MODEL_CACHE:
            _MODEL_CACHE[name] = SentenceTransformer(name)
        model = _MODEL_CACHE[name]
        vecs = model.encode([texts[i] or " " for i in todo], batch_size=batch, normalize_embeddings=True)
        for i, v in zip(todo, vecs, strict=True):
            v = np.asarray(v, dtype=np.float32)
            cached[keys[i]] = v
            if con is not None:
                con.execute(
                    "INSERT OR REPLACE INTO embeddings (hash, model, vector) VALUES (?,?,?)",
                    (keys[i], name, v.tobytes()),
                )
        if con is not None:
            con.commit()
    return _normalize(np.vstack([cached[k] for k in keys]))


def _tfidf_embed(texts: list[str], extra: list[str] | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = [t or " " for t in texts] + list(extra or [])
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=1, max_features=20000)
    try:
        m = vec.fit_transform(corpus)
    except ValueError:  # empty vocabulary
        z = np.zeros((len(corpus), 1), dtype=np.float32)
        return z[: len(texts)], (z[len(texts) :] if extra else None)
    k = min(128, m.shape[1] - 1, m.shape[0] - 1)
    dense = TruncatedSVD(n_components=k, random_state=0).fit_transform(m) if k >= 2 else m.toarray()
    dense = _normalize(np.asarray(dense, dtype=np.float32))
    return dense[: len(texts)], (dense[len(texts) :] if extra else None)


def embed(texts: list[str], con: sqlite3.Connection | None = None) -> np.ndarray:
    """Unit-length vectors, one row per text."""
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    if backend_name() == "sentence-transformers":
        return _st_embed(texts, con)
    return _tfidf_embed(texts)[0]


def embed_pair(
    texts: list[str], queries: list[str], con: sqlite3.Connection | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Embed two groups in the same vector space (needed for TF-IDF, which is fitted per corpus)."""
    if backend_name() == "sentence-transformers":
        return _st_embed(texts, con), _st_embed(queries, None)
    a, b = _tfidf_embed(texts, queries)
    return a, b if b is not None else np.zeros((0, a.shape[1]), dtype=np.float32)


def keywords(texts: list[str], groups: list[list[int]], top_n: int = 6) -> list[list[str]]:
    """Top TF-IDF terms for each group of text indices (used for topic names and fallbacks)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [" ".join(texts[i] for i in g) or " " for g in groups]
    try:
        vec = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#-]{2,}\b",  # noqa: S106 (regex, not a password)
        )
        m = vec.fit_transform(docs)
    except ValueError:
        return [[] for _ in groups]
    terms = np.array(vec.get_feature_names_out())
    out = []
    for row in m:
        arr = row.toarray().ravel()
        idx = arr.argsort()[::-1]
        picked: list[str] = []
        for i in idx:
            if arr[i] <= 0 or len(picked) >= top_n:
                break
            t = terms[i]
            if any(t in p or p in t for p in picked):
                continue
            picked.append(t)
        out.append(picked)
    return out
