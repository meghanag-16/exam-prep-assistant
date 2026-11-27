"""Cut a subject's pages into searchable chunks, in three ways (config/search.yaml):

- page:    one chunk per slide or page.
- window:  a fixed number of words with overlap, kept inside one topic, so a chunk
           can run over a page break but never mixes two topics.
- heading: one chunk per section of the topic map (pages under the same heading),
           split at page breaks when it grows too long.

Every chunk keeps the ids of its pages, its topic and the formulas on its pages,
so search results always point to a file and page. Repeated lines (course name,
instructor, ...) found by the topic step are left out, and so are near-duplicate
pages and cover/contents pages.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from examprep import config
from examprep.structure import sections, store

METHODS = ("page", "window", "heading")

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,
    method    TEXT NOT NULL,          -- page | window | heading
    text      TEXT NOT NULL,          -- what is searched and shown
    page_ids  TEXT NOT NULL,          -- JSON list of pages.id
    topic_id  INTEGER,
    heading   TEXT,
    formula_ids TEXT,                 -- JSON list of formulas.id on those pages
    words     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_method ON chunks(method);
"""


@dataclass
class PageInfo:
    id: int
    file_id: int
    file: str
    page_no: int
    text: str
    topic_id: int | None
    topic: str | None
    heading: str | None
    skip: bool  # near-duplicate or cover/contents page


@dataclass
class Chunk:
    method: str
    text: str
    page_ids: list[int]
    topic_id: int | None = None
    heading: str | None = None
    formula_ids: list[int] = field(default_factory=list)

    @property
    def words(self) -> int:
        return len(self.text.split())


def migrate(con: sqlite3.Connection) -> None:
    store.migrate(con)
    con.executescript(SCHEMA)
    con.commit()


def load_pages(con: sqlite3.Connection) -> list[PageInfo]:
    """Pages in file and page order, with repeated lines removed and their topic and heading."""
    store.migrate(con)
    pages = store.load_pages(con)
    extra = {
        r["id"]: r
        for r in con.execute(
            "SELECT p.id, p.topic_id, p.heading, p.is_filler, p.duplicate_of, p.deck_id, t.name AS topic"
            " FROM pages p LEFT JOIN topics t ON t.id = p.topic_id"
        )
    }
    units: dict[object, list[store.Page]] = {}
    for p in pages:
        deck = extra[p.id]["deck_id"] if p.id in extra else None
        units.setdefault(("deck", deck) if deck else ("file", p.file_id), []).append(p)
    boiler = sections.boilerplate(list(units.values()))
    out = []
    for p in pages:
        e = extra.get(p.id)
        text = "\n".join(sections.clean_lines(p, boiler))
        out.append(
            PageInfo(
                id=p.id,
                file_id=p.file_id,
                file=p.file_name,
                page_no=p.page_no,
                text=text,
                topic_id=e["topic_id"] if e else None,
                topic=e["topic"] if e else None,
                heading=e["heading"] if e else None,
                skip=bool(e and (e["duplicate_of"] or e["is_filler"])),
            )
        )
    return out


def _formulas(con: sqlite3.Connection) -> dict[int, list[tuple[int, str]]]:
    """page id -> [(formula id, "name: latex")], when formulas were extracted."""
    if not con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'formulas'").fetchone():
        return {}
    out: dict[int, list[tuple[int, str]]] = {}
    seen: set[tuple[int, str]] = set()
    for r in con.execute("SELECT id, page_id, name, latex, norm FROM formulas WHERE valid = 1 ORDER BY id"):
        key = (r["page_id"], r["norm"] or r["latex"])
        if key in seen:
            continue
        seen.add(key)
        label = f"{r['name']}: {r['latex']}" if r["name"] else r["latex"]
        out.setdefault(r["page_id"], []).append((r["id"], label))
    return out


def _context(topic: str | None, heading: str | None) -> str:
    parts = [x for x in (topic, heading) if x]
    if len(parts) == 2 and sections.norm(parts[0]) == sections.norm(parts[1]):
        parts = parts[:1]
    return " - ".join(parts)


def _finish(chunk: Chunk, pages: dict[int, PageInfo], formulas: dict[int, list[tuple[int, str]]]) -> Chunk:
    cfg = config.search()["chunking"]
    lines = []
    first = pages[chunk.page_ids[0]]
    if cfg.get("add_context", True):
        ctx = _context(first.topic, chunk.heading if chunk.method == "heading" else first.heading)
        if ctx:
            lines.append(ctx)
    lines.append(chunk.text)
    found = [f for pid in chunk.page_ids for f in formulas.get(pid, [])]
    chunk.formula_ids = [fid for fid, _ in found]
    if cfg.get("add_formulas", True) and found:
        lines.append("Formulas: " + "; ".join(label for _, label in found))
    chunk.text = "\n".join(lines)
    chunk.topic_id = first.topic_id
    return chunk


def _by_page(pages: list[PageInfo]) -> list[Chunk]:
    return [Chunk("page", p.text, [p.id], heading=p.heading) for p in pages if not p.skip]


def _windows(pages: list[PageInfo], size: int, overlap: int) -> list[Chunk]:
    """Word windows inside each topic (or file when there is no topic map)."""
    step = max(1, size - overlap)
    groups: list[list[PageInfo]] = []
    for p in pages:
        if p.skip:
            continue
        key = (p.file_id, p.topic_id)
        if groups and (groups[-1][0].file_id, groups[-1][0].topic_id) == key:
            groups[-1].append(p)
        else:
            groups.append([p])
    out = []
    for group in groups:
        words: list[tuple[str, int]] = [(w, p.id) for p in group for w in p.text.split()]
        start = 0
        while start < len(words):
            part = words[start : start + size]
            ids = list(dict.fromkeys(pid for _, pid in part))
            out.append(Chunk("window", " ".join(w for w, _ in part), ids))
            if start + size >= len(words):
                break
            start += step
    return out


def _sections(con: sqlite3.Connection, pages: list[PageInfo], max_words: int) -> list[Chunk]:
    """One chunk per section of the topic map; long sections are split at page breaks."""
    rows = con.execute(
        "SELECT s.heading, s.start_page, s.end_page, t.file_id, t.id AS topic_id FROM sections s"
        " JOIN topics t ON t.id = s.topic_id ORDER BY t.topic_no, s.start_page"
    ).fetchall()
    if not rows:
        return [Chunk("heading", c.text, c.page_ids, heading=c.heading) for c in _by_page(pages)]
    by_file: dict[int, dict[int, PageInfo]] = {}
    for p in pages:
        by_file.setdefault(p.file_id, {})[p.page_no] = p
    out: list[Chunk] = []
    used: set[int] = set()
    for r in rows:
        found = [by_file.get(r["file_id"], {}).get(n) for n in range(r["start_page"], r["end_page"] + 1)]
        members = [p for p in found if p and not p.skip and p.id not in used]
        cur: list[PageInfo] = []
        for p in members:
            used.add(p.id)
            if cur and sum(len(x.text.split()) for x in cur) + len(p.text.split()) > max_words:
                out.extend(_split_section(cur, r["heading"], max_words))
                cur = []
            cur.append(p)
        if cur:
            out.extend(_split_section(cur, r["heading"], max_words))
    # Pages outside every section (should be rare) still get a chunk.
    out += [Chunk("heading", p.text, [p.id], heading=p.heading) for p in pages if not p.skip and p.id not in used]
    return out


def _split_section(group: list[PageInfo], heading: str | None, max_words: int) -> list[Chunk]:
    text = "\n".join(p.text for p in group)
    words = text.split()
    if len(group) > 1 or len(words) <= max_words:
        return [Chunk("heading", text, [p.id for p in group], heading=heading)]
    # One very long page: cut it into pieces of max_words.
    return [
        Chunk("heading", " ".join(words[i : i + max_words]), [group[0].id], heading=heading)
        for i in range(0, len(words), max_words)
    ]


def make_chunks(con: sqlite3.Connection, method: str, pages: list[PageInfo] | None = None) -> list[Chunk]:
    if method not in METHODS:
        raise ValueError(f"unknown chunking method '{method}' (use one of {', '.join(METHODS)})")
    cfg = config.search()["chunking"]
    pages = pages if pages is not None else load_pages(con)
    if method == "page":
        raw = _by_page(pages)
    elif method == "window":
        raw = _windows(pages, int(cfg.get("window_words", 220)), int(cfg.get("window_overlap", 60)))
    else:
        raw = _sections(con, pages, int(cfg.get("heading_max_words", 450)))
    by_id = {p.id: p for p in pages}
    formulas = _formulas(con)
    min_words = int(cfg.get("min_words", 5))
    return [_finish(c, by_id, formulas) for c in raw if len(c.text.split()) >= min_words]


def save(con: sqlite3.Connection, method: str, chunks: list[Chunk]) -> list[int]:
    """Replace the stored chunks of one method. Returns the new chunk ids in order."""
    migrate(con)
    con.execute("DELETE FROM chunks WHERE method = ?", (method,))
    ids = []
    for c in chunks:
        cur = con.execute(
            "INSERT INTO chunks (method, text, page_ids, topic_id, heading, formula_ids, words) VALUES (?,?,?,?,?,?,?)",
            (c.method, c.text, json.dumps(c.page_ids), c.topic_id, c.heading, json.dumps(c.formula_ids), c.words),
        )
        ids.append(int(cur.lastrowid or 0))
    con.commit()
    return ids
