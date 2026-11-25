"""Page ledger: one SQLite database per subject with one row per page or slide.

This is the proof of full coverage: every page of every file gets a row, and
the report lists any page that ended up with no text at all.

Location: <data dir>/<subject>/ledger.sqlite
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

from examprep import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id          INTEGER PRIMARY KEY,
    sha256      TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    kind        TEXT NOT NULL,
    size        INTEGER,
    pages       INTEGER DEFAULT 0,
    status      TEXT NOT NULL,          -- done | failed
    error       TEXT,
    note        TEXT,
    seconds     REAL,
    ingested_at TEXT
);
CREATE TABLE IF NOT EXISTS pages (
    id                     INTEGER PRIMARY KEY,
    file_id                INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    page_no                INTEGER NOT NULL,   -- 1-based page, slide or text section
    title                  TEXT,
    text                   TEXT,               -- text layer / slide text / txt section
    tables_md              TEXT,               -- tables as Markdown
    notes                  TEXT,               -- PPTX speaker notes
    ocr_text               TEXT,               -- full-page OCR (scanned pages)
    layout_json            TEXT,               -- [[line, font_size, bold], ...] for topic detection
    n_images               INTEGER DEFAULT 0,
    needs_page_ocr         INTEGER DEFAULT 0,
    has_formula_candidates INTEGER DEFAULT 0,
    hash                   TEXT,
    status                 TEXT NOT NULL,      -- ok | needs_ocr | empty
    UNIQUE (file_id, page_no)
);
CREATE TABLE IF NOT EXISTS images (
    hash     TEXT PRIMARY KEY,
    width    INTEGER,
    height   INTEGER,
    ocr_text TEXT,
    status   TEXT NOT NULL                 -- pending | done | failed
);
CREATE TABLE IF NOT EXISTS page_images (
    page_id    INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
    image_hash TEXT NOT NULL REFERENCES images(hash),
    locator    TEXT,                       -- where the image sits inside its file
    PRIMARY KEY (page_id, image_hash)
);
CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY,
    started_at  TEXT,
    seconds     REAL,
    report_json TEXT
);
"""


def safe_subject(subject: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", subject.strip()).strip("_").lower()
    if not s:
        raise ValueError("subject name must contain letters or digits")
    return s


def subject_dir(subject: str) -> Path:
    d = config.data_dir() / safe_subject(subject)
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path(subject: str) -> Path:
    return subject_dir(subject) / "ledger.sqlite"


def connect(subject: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path(subject))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def file_by_hash(con: sqlite3.Connection, sha256: str) -> sqlite3.Row | None:
    return con.execute("SELECT * FROM files WHERE sha256 = ?", (sha256,)).fetchone()


def replace_file(con: sqlite3.Connection, src, status: str, pages: int, error: str | None, seconds: float) -> int:
    con.execute("DELETE FROM files WHERE sha256 = ?", (src.sha256,))
    cur = con.execute(
        "INSERT INTO files (sha256, name, kind, size, pages, status, error, note, seconds, ingested_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (src.sha256, src.name, src.kind, src.size, pages, status, error, src.note, round(seconds, 2), now()),
    )
    return int(cur.lastrowid)


def insert_page(con: sqlite3.Connection, file_id: int, page) -> int:
    cur = con.execute(
        "INSERT INTO pages (file_id, page_no, title, text, tables_md, notes, ocr_text, layout_json, n_images,"
        " needs_page_ocr, has_formula_candidates, hash, status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            file_id,
            page.page_no,
            page.title,
            page.text,
            page.tables_md,
            page.notes,
            page.ocr_text,
            json.dumps(page.layout) if page.layout else None,
            len(page.images),
            int(page.needs_page_ocr),
            int(page.has_formula_candidates),
            page.hash,
            page.status,
        ),
    )
    page_id = int(cur.lastrowid)
    for img in page.images:
        con.execute(
            "INSERT OR IGNORE INTO images (hash, width, height, status) VALUES (?,?,?, 'pending')",
            (img.hash, img.width, img.height),
        )
        con.execute(
            "INSERT OR IGNORE INTO page_images (page_id, image_hash, locator) VALUES (?,?,?)",
            (page_id, img.hash, img.locator),
        )
    return page_id


def pending_ocr_for_file(con: sqlite3.Connection, file_id: int) -> tuple[list[tuple[str, str]], list[int]]:
    """(image hash, locator) pairs still waiting for OCR, and page numbers that need full-page OCR."""
    imgs = con.execute(
        "SELECT DISTINCT i.hash, pi.locator FROM pages p"
        " JOIN page_images pi ON pi.page_id = p.id JOIN images i ON i.hash = pi.image_hash"
        " WHERE p.file_id = ? AND i.status = 'pending' AND pi.locator IS NOT NULL",
        (file_id,),
    ).fetchall()
    seen, unique = set(), []
    for h, loc in imgs:
        if h not in seen:
            seen.add(h)
            unique.append((h, loc))
    pages = [
        r[0]
        for r in con.execute(
            "SELECT page_no FROM pages WHERE file_id = ? AND needs_page_ocr = 1 AND ocr_text IS NULL ORDER BY page_no",
            (file_id,),
        )
    ]
    return unique, pages


def page_has_text(row: sqlite3.Row, image_texts: list[str]) -> bool:
    parts = [row["text"], row["tables_md"], row["notes"], row["ocr_text"], *image_texts]
    return any(p and p.strip() for p in parts)


def refresh_page_status(con: sqlite3.Connection, ocr_available: bool = True) -> None:
    """Recompute each page's status after OCR results came in. A page stays
    needs_ocr while its scan or any of its images has not been read yet."""
    rows = con.execute("SELECT * FROM pages").fetchall()
    for row in rows:
        imgs = con.execute(
            "SELECT i.ocr_text, i.status FROM page_images p JOIN images i ON i.hash = p.image_hash WHERE p.page_id = ?",
            (row["id"],),
        ).fetchall()
        pending_page = row["needs_page_ocr"] and row["ocr_text"] is None
        pending_imgs = any(i["status"] == "pending" for i in imgs)
        if pending_page or pending_imgs:
            status = "needs_ocr"
        elif page_has_text(row, [i["ocr_text"] for i in imgs]):
            status = "ok"
        else:
            status = "empty"
        if status != row["status"]:
            con.execute("UPDATE pages SET status = ? WHERE id = ?", (status, row["id"]))


def coverage(con: sqlite3.Connection) -> dict:
    """Summary used by the CLI report, the API and the UI."""
    files = [dict(r) for r in con.execute("SELECT id, name, kind, pages, status, error, note FROM files ORDER BY name")]
    counts = {
        r["status"]: r["n"] for r in con.execute("SELECT status, COUNT(*) AS n FROM pages GROUP BY status")
    }
    empty = [
        f"{r['name']} p.{r['page_no']}"
        for r in con.execute(
            "SELECT f.name, p.page_no FROM pages p JOIN files f ON f.id = p.file_id"
            " WHERE p.status = 'empty' ORDER BY f.name, p.page_no"
        )
    ]
    needs_ocr = con.execute("SELECT COUNT(*) FROM pages WHERE status = 'needs_ocr'").fetchone()[0]
    img = {r["status"]: r["n"] for r in con.execute("SELECT status, COUNT(*) AS n FROM images GROUP BY status")}
    formula = con.execute("SELECT COUNT(*) FROM pages WHERE has_formula_candidates = 1").fetchone()[0]
    return {
        "files": files,
        "pages_total": sum(counts.values()),
        "pages_by_status": counts,
        "pages_needing_ocr": needs_ocr,
        "empty_pages": empty,
        "unique_images_by_status": img,
        "formula_candidate_pages": formula,
    }
