"""Full-coverage notes, built so that no page is skipped.

For each topic of the topic map, in order:
1. Batch notes: the topic's pages go to the model a few at a time (config/answer.yaml
   notes.batch_pages, 3 to 5), each labelled [P1], [P2], ... Every point the model writes
   lists the pages it comes from.
2. Coverage check (code): every content page must be the source of at least one point,
   and the points for a page must contain enough of that page's key terms (a page whose
   points miss its terms was only skimmed). Pages that fail go back to the model on their
   own, up to notes.max_retries times; anything still missing is listed in the report.
3. Merge: the model groups the points into sections in topic-map order and marks exact
   repeats. It works with point numbers only, so it cannot drop or change a point: any
   point it leaves out is added back, and a repeat is removed only when the point kept
   covers the same pages.

Cover and contents pages, and near-duplicate pages (same slide shown again), are not sent
and are counted separately. Page citations are stored with every point (for the coverage
check and the marker test) but are not shown in the notes: each topic and section shows
the pages it covers instead.

Notes are stored in the ledger (table notes) and made the first time a topic is asked
for; they are made again only when the topic's pages, the topic map or the settings change.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Callable

from examprep import config, llm, tracing
from examprep.answer import terms as termlib
from examprep.ingest import ledger
from examprep.search import chunks as chunking
from examprep.search import search as searching

Log = Callable[[str], None]
PROMPT_VERSION = "notes-1"
MAX_PAGE_WORDS = 1200
_PLABEL = re.compile(r"P\d+")

SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    topic_id    INTEGER PRIMARY KEY,
    fingerprint TEXT NOT NULL,       -- topic pages, topic map and settings the notes were made from
    data        TEXT NOT NULL,       -- JSON: sections, points with page ids, coverage
    created     TEXT
);
"""


def migrate(con: sqlite3.Connection) -> None:
    chunking.migrate(con)
    con.executescript(SCHEMA)
    con.commit()


def _settings() -> dict:
    return config.answer()["notes"]


# ------------------------------------------------------------ inputs


class Topic:
    """One topic: its pages (with text cleaned of repeated lines) and sections."""

    def __init__(self, con: sqlite3.Connection, topic_id: int, pages: list[chunking.PageInfo]):
        row = con.execute("SELECT id, name, topic_no FROM topics WHERE id = ?", (topic_id,)).fetchone()
        if row is None:
            raise ValueError(f"no topic with id {topic_id}")
        self.id, self.name, self.no = row["id"], row["name"], row["topic_no"]
        self.pages = [p for p in pages if p.topic_id == topic_id]
        self.sections = [
            (r["heading"], r["start_page"], r["end_page"])
            for r in con.execute(
                "SELECT heading, start_page, end_page FROM sections WHERE topic_id = ? ORDER BY start_page",
                (topic_id,),
            )
        ]
        self.content = [p for p in self.pages if not p.skip and len(p.text.split()) >= 3]
        self.no_text = [p for p in self.pages if not p.skip and len(p.text.split()) < 3]
        self.skipped = [p for p in self.pages if p.skip]

    def heading_of(self, page: chunking.PageInfo) -> str | None:
        for heading, start, end in self.sections:
            if start <= page.page_no <= end and heading:
                return heading
        return page.heading


def fingerprint(con: sqlite3.Connection, topic: Topic) -> str:
    h = hashlib.sha256()
    ids = [p.id for p in topic.pages]
    hashes = {r[0]: r[1] for r in con.execute("SELECT id, IFNULL(hash, '') FROM pages")}
    h.update(json.dumps([PROMPT_VERSION, _settings(), topic.name, topic.sections], default=str).encode())
    h.update(json.dumps([[i, hashes.get(i, ""), p.skip] for i, p in zip(ids, topic.pages, strict=True)]).encode())
    h.update(json.dumps(sorted(chunking._formulas(con).get(i, []) for i in ids), default=str).encode())
    return h.hexdigest()[:24]


def _page_text(page: chunking.PageInfo, formulas: dict[int, list[tuple[int, str]]]) -> str:
    words = page.text.split()
    text = page.text if len(words) <= MAX_PAGE_WORDS else " ".join(words[:MAX_PAGE_WORDS]) + " ..."
    found = formulas.get(page.id, [])
    if found:
        text += "\nFormulas on this page: " + "; ".join(label for _, label in found)
    return text


# ------------------------------------------------------------ model steps


def _batch_messages(topic: Topic, batch: list[chunking.PageInfo], formulas, extra: str | None = None) -> list[dict]:
    rules = [
        "Write exam revision notes for the pages in <material>.",
        "Cover every page: each page label must appear in the pages list of at least one point.",
        "Keep every definition, step, condition, rule, formula, example, comparison and number a student needs "
        "for an exam. Leave out slide furniture such as repeated titles, instructor names and 'thank you'.",
        "Each point is one or two sentences of Markdown. Bold the key term. Write formulas in LaTeX between $ signs.",
        "pages lists the labels of the pages the point comes from. Use only the labels given.",
        "Use only what the pages say; never add facts of your own.",
        'Reply with JSON only: {"points": [{"text": "...", "pages": ["P1"]}]}',
    ]
    material = "\n\n".join(
        f"[P{n}] ({p.file} p.{p.page_no}" + (f", {topic.heading_of(p)}" if topic.heading_of(p) else "") + ")\n"
        f"{_page_text(p, formulas)}"
        for n, p in enumerate(batch, start=1)
    )
    user = f"Topic: {topic.name}\n\n<material>\n{material}\n</material>"
    if extra:
        user += "\n\n" + extra
    return [
        {"role": "system", "content": llm.SYSTEM_GUARD + "\n\n" + "\n".join(f"- {r}" for r in rules)},
        {"role": "user", "content": user},
    ]


def _points(reply, batch: list[chunking.PageInfo]) -> list[dict]:
    """[{"text", "page_ids"}] from a reply, keeping only labels that were given."""
    items = reply.get("points") if isinstance(reply, dict) else reply
    out = []
    for it in items or []:
        if isinstance(it, str):
            it = {"text": it, "pages": _PLABEL.findall(it)}
        if not isinstance(it, dict) or not str(it.get("text") or "").strip():
            continue
        labels = [str(x).strip().strip("[]") for x in (it.get("pages") or [])]
        labels += _PLABEL.findall(str(it["text"]))
        ids = []
        for lab in labels:
            m = re.fullmatch(r"P(\d+)", lab)
            if m and 1 <= int(m.group(1)) <= len(batch):
                pid = batch[int(m.group(1)) - 1].id
                if pid not in ids:
                    ids.append(pid)
        text = re.sub(r"\s*\[P\d+(?:\s*,\s*P\d+)*\]", "", str(it["text"])).strip()
        out.append({"text": text, "page_ids": ids})
    return out


def check_pages(
    points: list[dict], pages: list[chunking.PageInfo], stats: termlib.TermStats
) -> tuple[list[int], dict[int, list[str]]]:
    """(uncovered page ids, {thin page id: missing key terms}) for these points."""
    cfg = _settings()
    n_terms, need = int(cfg.get("page_terms", 5)), float(cfg.get("min_term_share", 0.4))
    uncovered, thin = [], {}
    for p in pages:
        text = "\n".join(pt["text"] for pt in points if p.id in pt["page_ids"])
        if not text:
            uncovered.append(p.id)
            continue
        key = stats.key_terms(p.text, n_terms)
        if len(key) >= 2 and termlib.share_present(key, text) < need:
            have = set(termlib.present(key, text))
            thin[p.id] = [t for t in key if t not in have]
    return uncovered, thin


def _batch_notes(topic: Topic, batch, formulas, stats, log: Log) -> tuple[list[dict], list[int], dict]:
    """Points for one batch of pages, with retries for pages left uncovered or thin."""
    try:
        points = _points(
            llm.chat_json(_batch_messages(topic, batch, formulas), max_tokens=4096, trace_name="write-notes"), batch
        )
    except llm.LLMError as e:
        log(f"    notes request failed ({e}); retrying page by page")
        points = []
    uncovered, thin = check_pages(points, batch, stats)
    for _ in range(int(_settings().get("max_retries", 2))):
        redo = [p for p in batch if p.id in uncovered or p.id in thin]
        if not redo:
            break
        notes = []
        for n, p in enumerate(redo, start=1):
            if p.id in thin:
                notes.append(f"[P{n}] was covered too briefly: include {', '.join(thin[p.id])}")
            else:
                notes.append(f"[P{n}] had no points")
        extra = "These pages were missed before. Write points for every one of them:\n" + "\n".join(notes)
        try:
            more = _points(
                llm.chat_json(
                    _batch_messages(topic, redo, formulas, extra), max_tokens=4096, trace_name="rewrite-missed-notes"
                ),
                redo,
            )
        except llm.LLMError:
            continue
        points += more
        uncovered, thin = check_pages(points, batch, stats)
    return points, uncovered, thin


def _merge(topic: Topic, points: list[dict], pages_by_id: dict[int, chunking.PageInfo]) -> list[dict]:
    """Sections of points in topic-map order. The model only orders and groups point numbers."""
    fallback = _group_by_heading(topic, points, pages_by_id)
    if len(points) <= 8:
        return fallback
    heads = [h for h, _, _ in topic.sections if h]

    def page_nos(pt: dict) -> str:
        return ", ".join(str(pages_by_id[i].page_no) for i in pt["page_ids"] if i in pages_by_id)

    listing = "\n".join(f"{n}. {pt['text']} (pages: {page_nos(pt)})" for n, pt in enumerate(points, start=1))
    rules = [
        "Organise these revision-note points for one topic into sections, in the order the topic is taught.",
        "Use the topic map's section headings where they fit; otherwise write a short heading.",
        "Every point number must appear in exactly one section, unless it says the same thing as another "
        "point: then list it in duplicates with the number of the point it repeats.",
        "Do not rewrite points; refer to them by number only.",
        'Reply with JSON only: {"sections": [{"heading": "...", "points": [1, 2]}], '
        '"duplicates": [{"drop": 5, "keep": 2}]}',
    ]
    user = f"Topic: {topic.name}\nSection headings from the topic map: {'; '.join(heads) or 'none'}\n\n"
    user += f"<material>\n{listing}\n</material>"
    try:
        reply = llm.chat_json(
            [
                {"role": "system", "content": llm.SYSTEM_GUARD + "\n\n" + "\n".join(f"- {r}" for r in rules)},
                {"role": "user", "content": user},
            ],
            max_tokens=4096,
            trace_name="merge-notes",
        )
    except llm.LLMError:
        return fallback
    if not isinstance(reply, dict) or not isinstance(reply.get("sections"), list):
        return fallback
    n = len(points)
    pts = [dict(p, page_ids=list(p["page_ids"])) for p in points]
    dropped: set[int] = set()
    for d in reply.get("duplicates") or []:
        try:
            drop, keep = int(d["drop"]), int(d["keep"])
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= drop <= n and 1 <= keep <= n and drop != keep and keep not in dropped:
            # Removed only if the kept point then covers every page of the repeat.
            pts[keep - 1]["page_ids"] += [i for i in pts[drop - 1]["page_ids"] if i not in pts[keep - 1]["page_ids"]]
            dropped.add(drop)
    sections, used = [], set()
    for sec in reply["sections"]:
        if not isinstance(sec, dict):
            continue
        nums = []
        for x in sec.get("points") or []:
            try:
                k = int(x)
            except (TypeError, ValueError):
                continue
            if 1 <= k <= n and k not in used and k not in dropped:
                used.add(k)
                nums.append(k)
        if nums:
            sections.append(
                {"heading": str(sec.get("heading") or topic.name)[:120], "points": [pts[k - 1] for k in nums]}
            )
    left = [pts[k - 1] for k in range(1, n + 1) if k not in used and k not in dropped]
    if left:
        sections.append({"heading": "Other points", "points": left})
    return sections or fallback


def _group_by_heading(topic: Topic, points: list[dict], pages_by_id: dict) -> list[dict]:
    sections: list[dict] = []
    for pt in points:
        first = next((pages_by_id[i] for i in pt["page_ids"] if i in pages_by_id), None)
        heading = (topic.heading_of(first) if first else None) or topic.name
        if sections and sections[-1]["heading"] == heading:
            sections[-1]["points"].append(pt)
        else:
            sections.append({"heading": heading, "points": [pt]})
    return sections


# ------------------------------------------------------------ entry points


def make(con: sqlite3.Connection, topic: Topic, log: Log = print) -> dict:
    """Notes for one topic (always calls the model; see topic_notes for the cached version)."""
    cfg = _settings()
    size = max(1, min(int(cfg.get("batch_pages", 4)), 8))
    formulas = chunking._formulas(con)
    stats = termlib.TermStats(con)
    by_id = {p.id: p for p in topic.pages}
    points: list[dict] = []
    uncovered: list[int] = []
    thin: dict[int, list[str]] = {}
    with (
        llm.track_usage() as usage,
        tracing.trace(
            "make-topic-notes",
            input=topic.name,
            tags=["notes"],
            metadata={"topic_id": topic.id, "pages": len(topic.content)},
        ) as root,
    ):
        for i in range(0, len(topic.content), size):
            batch = topic.content[i : i + size]
            pts, unc, th = _batch_notes(topic, batch, formulas, stats, log)
            points += pts
            uncovered += unc
            thin.update(th)
        sections = _merge(topic, points, by_id) if points else []
        n_content = len(topic.content)
        share = (n_content - len(set(uncovered))) / n_content if n_content else 1.0
        tracing.update(
            root,
            output={"sections": [x.get("heading") for x in sections], "points": len(points)},
            metadata={"uncovered_pages": len(set(uncovered)), "thin_pages": len(thin)},
        )
        if root is not None:
            tracing.score("notes-coverage", round(share, 3), data_type="NUMERIC")

    def ref(pid: int) -> dict:
        p = by_id[pid]
        return {"page_id": pid, "file": p.file, "page_no": p.page_no}

    def where(ids) -> str:
        return searching.where(
            [{"name": by_id[i].file.replace("\\", "/").rsplit("/", 1)[-1], "page_no": by_id[i].page_no} for i in ids]
        )

    content_ids = [p.id for p in topic.content]
    covered = [i for i in content_ids if i not in uncovered]
    for sec in sections:
        ids = sorted(
            {i for pt in sec["points"] for i in pt["page_ids"]}, key=lambda i: (by_id[i].file, by_id[i].page_no)
        )
        sec["pages"] = where(ids) if ids else ""
    return {
        "topic_id": topic.id,
        "topic": topic.name,
        "topic_no": topic.no,
        "sections": sections,
        "pages": where(content_ids) if content_ids else "",
        "coverage": {
            "content_pages": len(content_ids),
            "covered": len(covered),
            "share": round(len(covered) / len(content_ids), 3) if content_ids else 1.0,
            "uncovered": [ref(i) for i in uncovered],
            "thin": [{**ref(i), "missing_terms": t} for i, t in thin.items() if i not in uncovered],
            "no_text": [ref(p.id) for p in topic.no_text],
            "skipped": len(topic.skipped),
        },
        "llm": {
            "calls": usage["calls"],
            "cached_calls": usage["cached"],
            "tokens": usage["prompt_tokens"] + usage["completion_tokens"],
        },
        "created": ledger.now(),
    }


def topic_notes(subject: str, topic_id: int, redo: bool = False, log: Log = print, con=None) -> dict:
    """Stored notes for a topic, made first if missing or out of date."""
    own = con is None
    con = con or ledger.connect(ledger.safe_subject(subject))
    try:
        migrate(con)
        topic = Topic(con, topic_id, chunking.load_pages(con))
        fp = fingerprint(con, topic)
        row = con.execute("SELECT fingerprint, data FROM notes WHERE topic_id = ?", (topic_id,)).fetchone()
        if row and row[0] == fp and not redo:
            return json.loads(row[1])
        log(f"  notes: {topic.name} ({len(topic.content)} pages)")
        data = make(con, topic, log)
        data["fingerprint"] = fp
        con.execute(
            "INSERT OR REPLACE INTO notes (topic_id, fingerprint, data, created) VALUES (?,?,?,?)",
            (topic_id, fp, json.dumps(data, ensure_ascii=False), data["created"]),
        )
        con.commit()
        return data
    finally:
        if own:
            con.close()


def topic_ids(subject: str, wanted: list[str] | None = None) -> list[int]:
    """Topic ids in topic-map order. wanted: ids or parts of names (case-insensitive)."""
    con = ledger.connect(ledger.safe_subject(subject))
    try:
        migrate(con)
        rows = con.execute("SELECT id, name FROM topics ORDER BY topic_no").fetchall()
    finally:
        con.close()
    if not rows:
        raise ValueError("No topic map yet: run examprep-topics first.")
    if not wanted:
        return [r[0] for r in rows]
    out = []
    for w in wanted:
        w = str(w).strip()
        hits = [
            r[0] for r in rows if (w.isdigit() and r[0] == int(w)) or (not w.isdigit() and w.lower() in r[1].lower())
        ]
        if not hits:
            raise ValueError(f"no topic matches '{w}'")
        out += [h for h in hits if h not in out]
    return out


def subject_notes(subject: str, wanted: list[str] | None = None, redo: bool = False, log: Log = print) -> dict:
    """Notes for several topics (all by default) and the coverage report."""
    subject = ledger.safe_subject(subject)
    ids = topic_ids(subject, wanted)
    con = ledger.connect(subject)
    try:
        notes = [topic_notes(subject, i, redo, log, con) for i in ids]
    finally:
        con.close()
    content = sum(n["coverage"]["content_pages"] for n in notes)
    covered = sum(n["coverage"]["covered"] for n in notes)
    report = {
        "subject": subject,
        "topics": len(notes),
        "content_pages": content,
        "covered_pages": covered,
        "coverage": round(covered / content, 4) if content else 1.0,
        "uncovered": [dict(u, topic=n["topic"]) for n in notes for u in n["coverage"]["uncovered"]],
        "thin": [dict(t, topic=n["topic"]) for n in notes for t in n["coverage"]["thin"]],
        "no_text": [dict(u, topic=n["topic"]) for n in notes for u in n["coverage"]["no_text"]],
        "skipped_pages": sum(n["coverage"]["skipped"] for n in notes),
        "llm_calls": sum(n["llm"]["calls"] for n in notes),
        "run_at": ledger.now(),
    }
    if not wanted:
        out = ledger.subject_dir(subject) / "notes"
        out.mkdir(parents=True, exist_ok=True)
        (out / "notes_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"notes": notes, "report": report}
