"""Formula tables, kept in the same ledger.sqlite as the pages."""

from __future__ import annotations

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS formulas (
    id          INTEGER PRIMARY KEY,
    page_id     INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    file_id     INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    page_no     INTEGER NOT NULL,
    source      TEXT NOT NULL,          -- text | image | page (scanned page)
    image_hash  TEXT,
    kind        TEXT,                   -- equation | formula | matrix | expression | inequality | relational | other
    latex       TEXT NOT NULL,
    name        TEXT,
    meaning     TEXT,
    symbols     TEXT,                   -- JSON list of {"symbol", "meaning"}
    valid       INTEGER NOT NULL,       -- LaTeX is well formed
    issues      TEXT,                   -- JSON list of problems found by the check
    status      TEXT NOT NULL,          -- ok | needs_review
    norm        TEXT NOT NULL,          -- normalised LaTeX, to spot the same formula on several pages
    job_key     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS formulas_page ON formulas(page_id);
CREATE INDEX IF NOT EXISTS formulas_job ON formulas(job_key);
CREATE TABLE IF NOT EXISTS formula_jobs (
    key         TEXT PRIMARY KEY,       -- what was sent: page text, image or scanned page (by content hash)
    kind        TEXT NOT NULL,          -- text | image | page | caption
    status      TEXT NOT NULL,          -- done | failed
    error       TEXT,
    found       INTEGER DEFAULT 0,
    done_at     TEXT
);
"""


def migrate(con: sqlite3.Connection) -> None:
    from examprep.structure import store as topic_store

    topic_store.migrate(con)  # formulas are shown with their topic, so the topic tables must exist
    con.executescript(SCHEMA)
    have = {r[1] for r in con.execute("PRAGMA table_info(pages)")}
    if "diagram_caption" not in have:
        con.execute("ALTER TABLE pages ADD COLUMN diagram_caption TEXT")
    con.commit()
