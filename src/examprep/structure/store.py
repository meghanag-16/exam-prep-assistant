"""Topic-map tables (stored in the same ledger.sqlite as the pages) and helpers
to load page text for analysis."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

SCHEMA = """
CREATE TABLE IF NOT EXISTS decks (
    id          INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    deck_no     INTEGER NOT NULL,
    start_page  INTEGER NOT NULL,
    end_page    INTEGER NOT NULL,
    title       TEXT,
    evidence    TEXT                     -- JSON: why this deck starts here
);
CREATE TABLE IF NOT EXISTS topics (
    id          INTEGER PRIMARY KEY,
    deck_id     INTEGER NOT NULL REFERENCES decks(id) ON DELETE CASCADE,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    topic_no    INTEGER NOT NULL,        -- order within the subject
    name        TEXT NOT NULL,
    name_source TEXT NOT NULL,           -- llm | heading | keywords | user
    start_page  INTEGER NOT NULL,
    end_page    INTEGER NOT NULL,
    keywords    TEXT                     -- JSON list
);
CREATE TABLE IF NOT EXISTS sections (
    id          INTEGER PRIMARY KEY,
    topic_id    INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    heading     TEXT,
    start_page  INTEGER NOT NULL,
    end_page    INTEGER NOT NULL,
    subheadings TEXT                     -- JSON list: the points under this section
);
CREATE TABLE IF NOT EXISTS syllabus_points (
    id         INTEGER PRIMARY KEY,
    unit       TEXT,
    text       TEXT NOT NULL,
    topic_id   INTEGER REFERENCES topics(id) ON DELETE SET NULL,
    score      REAL,
    covered    INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS embeddings (
    hash    TEXT NOT NULL,
    model   TEXT NOT NULL,
    vector  BLOB NOT NULL,
    PRIMARY KEY (hash, model)
);
"""

PAGE_COLUMNS = {
    "heading": "TEXT",
    "subheading": "TEXT",
    "is_filler": "INTEGER",
    "duplicate_of": "INTEGER",
    "deck_id": "INTEGER",
    "topic_id": "INTEGER",
}


def migrate(con: sqlite3.Connection) -> None:
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(pages)")}
    for col, typ in PAGE_COLUMNS.items():
        if col not in have:
            con.execute(f"ALTER TABLE pages ADD COLUMN {col} {typ}")
    con.commit()


@dataclass
class Page:
    id: int
    file_id: int
    file_name: str
    kind: str
    page_no: int
    title: str | None
    text: str  # everything readable on the page: text, tables, notes, OCR
    layout: list = field(default_factory=list)
    hash: str = ""


def load_pages(con: sqlite3.Connection) -> list[Page]:
    """All pages of all successfully read files, in file and page order."""
    # Text read from an image that appears on many pages of a file (a college
    # logo or banner) says nothing about the page, so it is left out here.
    file_pages = {r[0]: r[1] for r in con.execute("SELECT file_id, COUNT(*) FROM pages GROUP BY file_id")}
    img_text: dict[int, list[str]] = {}
    for r in con.execute(
        "SELECT pi.page_id, i.ocr_text, p.file_id,"
        " (SELECT COUNT(*) FROM page_images x JOIN pages y ON y.id = x.page_id"
        "  WHERE x.image_hash = pi.image_hash AND y.file_id = p.file_id) AS uses"
        " FROM page_images pi JOIN images i ON i.hash = pi.image_hash JOIN pages p ON p.id = pi.page_id"
        " WHERE i.ocr_text IS NOT NULL AND i.ocr_text != ''"
    ):
        if r[3] >= 3 and r[3] >= 0.2 * file_pages.get(r[2], 0):
            continue
        img_text.setdefault(r[0], []).append(r[1])
    pages = []
    for r in con.execute(
        "SELECT p.*, f.name AS file_name, f.kind AS kind FROM pages p JOIN files f ON f.id = p.file_id"
        " WHERE f.status = 'done' ORDER BY f.name, p.page_no"
    ):
        parts = [r["text"], r["tables_md"], r["notes"], r["ocr_text"], *img_text.get(r["id"], [])]
        text = "\n".join(p for p in parts if p and p.strip())
        pages.append(
            Page(
                id=r["id"],
                file_id=r["file_id"],
                file_name=r["file_name"],
                kind=r["kind"],
                page_no=r["page_no"],
                title=r["title"],
                text=text,
                layout=json.loads(r["layout_json"]) if r["layout_json"] else [],
                hash=r["hash"] or "",
            )
        )
    return pages
