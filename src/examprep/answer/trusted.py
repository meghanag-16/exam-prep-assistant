"""Passages from trusted sites for the "Beyond your slides" part of an answer.

1. Search: the providers in config/trusted_sites.yaml search_providers, in order, restricted
   to the allowlisted sites (the default list plus the subject's own list). By default:
   Tavily (include_domains, free monthly credit), then Exa (includeDomains, free monthly
   credit), then DuckDuckGo through the ddgs library (site: operators, no key; unofficial
   and rate-limited, so it is the last resort). A provider without a key is skipped. Brave
   Search is available too but its free credit now needs a card on file. (Google's Custom
   Search JSON API is closed to new customers and shuts down on 1 January 2027.)
2. Every result URL is checked against the same allowlist in code, before and after
   redirects, so a domain outside the list is never used even if a search returns it.
3. The main text of each page is extracted with trafilatura (Tavily's snippet is used
   when a page cannot be fetched) and cut into passages.
4. Relevance gate: a passage is kept only if it is about the question (reranker score,
   where the reranker is installed) and close to what the slides say (similarity with the
   slide passages retrieved for the question). The rest are counted as rejected.

Searches and fetched pages are cached in data/web_cache.sqlite, so a repeated question
costs no Tavily credit and no download.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import numpy as np

from examprep import config, tracing

TAVILY_URL = "https://api.tavily.com/search"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"
EXA_URL = "https://api.exa.ai/search"
USER_AGENT = "examprep/0.1 (study tool; reads public pages from an allowlist)"


class TrustedError(RuntimeError):
    pass


@dataclass
class WebPassage:
    url: str
    title: str
    site: str
    text: str
    score: float | None = None  # reranker score against the question (None without a reranker)
    slide_similarity: float | None = None
    label: str = ""

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "url": self.url,
            "title": self.title,
            "site": self.site,
            "text": self.text,
            "score": None if self.score is None else round(self.score, 4),
            "slide_similarity": None if self.slide_similarity is None else round(self.slide_similarity, 4),
        }


def site_of(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    return host.removeprefix("www.")


def allowed(url: str, domains: list[str]) -> bool:
    """True only for http(s) URLs on an allowlisted domain or one of its subdomains."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = (parsed.hostname or "").lower()
    return any(host == d or host.endswith("." + d) for d in (x.lower().strip() for x in domains) if d)


# ------------------------------------------------------------ cache


def cache_path():
    return config.data_dir() / "web_cache.sqlite"


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(cache_path(), timeout=30)
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS searches (key TEXT PRIMARY KEY, query TEXT, results TEXT NOT NULL, created REAL);
        CREATE TABLE IF NOT EXISTS pages (url TEXT PRIMARY KEY, text TEXT, created REAL);
        """
    )
    return con


def _fresh() -> float:
    return time.time() - float(config.answer()["trusted"].get("cache_days", 30)) * 86400


def _cached(sql: str, key: str):
    try:
        con = _db()
        try:
            row = con.execute(sql, (key, _fresh())).fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    return row


def _store(sql: str, values: tuple) -> None:
    try:
        con = _db()
        try:
            con.execute(sql, values)
            con.commit()
        finally:
            con.close()
    except sqlite3.Error:
        pass


# ------------------------------------------------------------ search and fetch


def _tavily(query: str, domains: list[str], cfg: dict) -> list[dict]:
    token = config.secret("TAVILY_API_KEY")
    if not token:
        raise TrustedError("TAVILY_API_KEY is not set")
    try:
        r = httpx.post(
            TAVILY_URL,
            json={"query": query, "max_results": int(cfg.get("max_results", 5)), "include_domains": domains},
            headers={"Authorization": f"Bearer {token}"},
            timeout=float(cfg.get("fetch_timeout", 15)),
        )
    except httpx.HTTPError as e:
        raise TrustedError(f"could not reach Tavily: {e.__class__.__name__}") from e
    if r.status_code != 200:
        raise TrustedError(f"HTTP {r.status_code} from Tavily")
    return [
        {"url": x.get("url", ""), "title": x.get("title") or "", "content": x.get("content") or ""}
        for x in (r.json().get("results") or [])
        if isinstance(x, dict)
    ]


def _brave(query: str, domains: list[str], cfg: dict) -> list[dict]:
    token = config.secret("BRAVE_API_KEY")
    if not token:
        raise TrustedError("BRAVE_API_KEY is not set")
    sites = " OR ".join(f"site:{d}" for d in domains)
    q = f"{query[: max(40, 380 - len(sites))]} ({sites})"
    try:
        r = httpx.get(
            BRAVE_URL,
            params={"q": q, "count": min(20, int(cfg.get("max_results", 5)) * 2)},
            headers={"Accept": "application/json", "X-Subscription-Token": token},
            timeout=float(cfg.get("fetch_timeout", 15)),
        )
    except httpx.HTTPError as e:
        raise TrustedError(f"could not reach Brave Search: {e.__class__.__name__}") from e
    if r.status_code != 200:
        raise TrustedError(f"HTTP {r.status_code} from Brave Search")
    web = (r.json().get("web") or {}).get("results") or []
    return [
        {
            "url": x.get("url", ""),
            "title": re.sub(r"<[^>]+>", "", x.get("title") or ""),
            "content": re.sub(r"<[^>]+>", "", x.get("description") or ""),
        }
        for x in web
        if isinstance(x, dict)
    ]


def _exa(query: str, domains: list[str], cfg: dict) -> list[dict]:
    token = config.secret("EXA_API_KEY")
    if not token:
        raise TrustedError("EXA_API_KEY is not set")
    try:
        r = httpx.post(
            EXA_URL,
            json={
                "query": query,
                "numResults": min(10, int(cfg.get("max_results", 5))),
                "includeDomains": domains,
                "contents": {"highlights": True},
            },
            headers={"x-api-key": token},
            timeout=float(cfg.get("fetch_timeout", 15)),
        )
    except httpx.HTTPError as e:
        raise TrustedError(f"could not reach Exa: {e.__class__.__name__}") from e
    if r.status_code != 200:
        raise TrustedError(f"HTTP {r.status_code} from Exa")
    return [
        {
            "url": x.get("url", ""),
            "title": x.get("title") or "",
            "content": " ".join(h for h in (x.get("highlights") or []) if isinstance(h, str)) or x.get("text") or "",
        }
        for x in (r.json().get("results") or [])
        if isinstance(x, dict)
    ]


def _duckduckgo(query: str, domains: list[str], cfg: dict) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError as e:
        raise TrustedError("the ddgs package is not installed") from e
    sites = " OR ".join(f"site:{d}" for d in domains)
    try:
        rows = DDGS(timeout=int(cfg.get("fetch_timeout", 15))).text(
            f"{query} ({sites})", max_results=min(20, int(cfg.get("max_results", 5)) * 2), backend="duckduckgo"
        )
    except Exception as e:  # noqa: BLE001 (ddgs raises its own errors for rate limits and timeouts)
        raise TrustedError(f"DuckDuckGo search failed: {e.__class__.__name__}") from e
    return [
        {"url": x.get("href", ""), "title": x.get("title") or "", "content": x.get("body") or ""}
        for x in rows or []
        if isinstance(x, dict)
    ]


PROVIDERS = {"tavily": _tavily, "exa": _exa, "duckduckgo": _duckduckgo, "brave": _brave}
DEFAULT_ORDER = ["tavily", "exa", "duckduckgo"]


def search(query: str, domains: list[str]) -> list[dict]:
    """[{"url", "title", "content", "provider"}] from the first search provider that finds
    something on the allowlisted domains. Raises TrustedError when every provider failed."""
    if not domains:
        return []
    cfg = config.answer()["trusted"]
    order = [p for p in (config.trusted_sites().get("search_providers") or DEFAULT_ORDER) if p in PROVIDERS]
    key = hashlib.sha256(json.dumps([query, sorted(domains), order]).encode()).hexdigest()
    row = _cached("SELECT results FROM searches WHERE key = ? AND created >= ?", key)
    if row:
        return [x for x in json.loads(row[0]) if allowed(x["url"], domains)]
    errors = []
    for name in order:
        with tracing.observe(
            "search-web", as_type="retriever", input=query, metadata={"provider": name, "sites": len(domains)}
        ) as obs:
            try:
                found = [
                    dict(x, provider=name) for x in PROVIDERS[name](query, domains, cfg) if allowed(x["url"], domains)
                ]
            except TrustedError as e:
                tracing.update(obs, level="WARNING", status_message=str(e)[:200])
                errors.append(f"{name}: {e}")
                continue
            tracing.update(obs, output=[{"url": x["url"], "title": x["title"]} for x in found])
        if found:
            _store("INSERT OR REPLACE INTO searches VALUES (?,?,?,?)", (key, query, json.dumps(found), time.time()))
            return found
        errors.append(f"{name}: nothing on the allowed sites")
    if errors and all("not set" in e for e in errors):
        raise TrustedError("no search provider is available (set TAVILY_API_KEY or EXA_API_KEY)")
    if errors and not any("nothing on the allowed sites" in e for e in errors):
        raise TrustedError("; ".join(errors))
    return []


def fetch_text(url: str, domains: list[str]) -> str | None:
    """Main text of an allowlisted page (trafilatura), or None when it cannot be read."""
    if not allowed(url, domains):
        return None
    row = _cached("SELECT text FROM pages WHERE url = ? AND created >= ?", url)
    if row:
        return row[0]
    import trafilatura

    text = None
    try:
        r = httpx.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=float(config.answer()["trusted"].get("fetch_timeout", 15)),
            follow_redirects=True,
        )
        if r.status_code == 200 and allowed(str(r.url), domains):
            text = trafilatura.extract(r.text, include_comments=False, include_tables=True)
    except httpx.HTTPError:
        text = None
    if text:
        _store("INSERT OR REPLACE INTO pages VALUES (?,?,?)", (url, text, time.time()))
    return text


def split(text: str, words: int) -> list[str]:
    """Passages of about `words` words, cut at paragraph ends where possible."""
    out: list[str] = []
    cur: list[str] = []
    for para in (p.strip() for p in text.split("\n")):
        if not para:
            continue
        w = para.split()
        if cur and len(cur) + len(w) > words:
            out.append(" ".join(cur))
            cur = []
        while len(w) > words:
            out.append(" ".join(w[:words]))
            w = w[words:]
        cur += w
    if cur:
        out.append(" ".join(cur))
    return [p for p in out if len(p.split()) >= 25]


# ------------------------------------------------------------ relevance gate


def _slide_similarity(passages: list[str], slides: list[str]) -> tuple[list[float], str]:
    from examprep.structure import embed

    backend = embed.backend_name()
    if not passages or not slides:
        return [0.0] * len(passages), backend
    a, b = embed.embed_pair(passages, slides)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    return [float(x) for x in (a @ b.T).max(axis=1)], backend


def gate(
    question: str, slides: list[str], candidates: list[WebPassage], reranker=None
) -> tuple[list[WebPassage], list[WebPassage]]:
    """(kept, rejected). Kept passages are sorted best first. With slide passages, a passage must
    be close to them; without (a question your slides do not cover), it must be close to the
    question itself. Where the reranker is installed, its score must also pass."""
    cfg = config.trusted_sites()
    texts = [c.text for c in candidates]
    if slides:
        sims, backend = _slide_similarity(texts, slides)
        need_sim = float((cfg.get("slide_similarity") or {}).get(backend, 0.0))
    else:
        sims, backend = _slide_similarity(texts, [question])
        need_sim = float((cfg.get("question_similarity") or {}).get(backend, 0.0))
    scores = reranker(question, texts) if reranker and candidates else None
    need_score = float(cfg.get("relevance_threshold", 0.5))
    kept: list[WebPassage] = []
    rejected: list[WebPassage] = []
    for i, c in enumerate(candidates):
        c.slide_similarity = sims[i]
        c.score = None if scores is None else float(scores[i])
        ok = c.slide_similarity >= need_sim and (c.score is None or c.score >= need_score)
        (kept if ok else rejected).append(c)
    kept.sort(key=lambda c: (c.score if c.score is not None else 0.0, c.slide_similarity), reverse=True)
    return kept, rejected


def passages(question: str, subject: str, slides: list[str], reranker=None) -> dict:
    """Trusted passages for a question: {"kept": [...], "rejected": n, "candidates": n, "error"}."""
    with tracing.observe("find-trusted-passages", as_type="chain", input=question) as obs:
        res = _passages(question, subject, slides, reranker)
        tracing.update(
            obs,
            output=[{"label": p.label, "url": p.url, "text": p.text} for p in res["kept"]],
            metadata={k: res[k] for k in ("candidates", "rejected", "provider", "error")},
            level="WARNING" if res["error"] else None,
        )
    return res


def _passages(question: str, subject: str, slides: list[str], reranker=None) -> dict:
    cfg = config.answer()["trusted"]
    domains = config.allowed_domains(subject)
    try:
        results = search(question, domains)
    except TrustedError as e:
        return {"kept": [], "rejected": 0, "candidates": 0, "error": str(e), "provider": None}
    cands: list[WebPassage] = []
    results = [r for r in results if allowed(r["url"], domains)]  # checked again here, whatever the search returned
    # Pages are downloaded at the same time, not one after another (each can take up to fetch_timeout).
    with ThreadPoolExecutor(max_workers=max(1, min(8, len(results)))) as pool:
        texts = list(pool.map(lambda r: fetch_text(r["url"], domains), results))
    for res, page_text in zip(results, texts, strict=True):
        text = page_text or res["content"]
        for part in split(text or "", int(cfg.get("passage_words", 180))):
            cands.append(WebPassage(res["url"], res["title"], site_of(res["url"]), part))
    kept, rejected = gate(question, slides, cands, reranker)
    kept = kept[: int(cfg.get("max_passages", 4))]
    for n, p in enumerate(kept, start=1):
        p.label = f"W{n}"
    provider = results[0].get("provider") if results else None
    return {"kept": kept, "rejected": len(rejected), "candidates": len(cands), "error": None, "provider": provider}
