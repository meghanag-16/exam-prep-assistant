"""Build the topic map for a subject from its page ledger.

Steps: headings -> near-duplicate pages -> deck boundaries (combined PDFs) ->
contiguous topic segments -> topic names (LLM, with heading/keyword fallback)
-> optional syllabus mapping. Topic names edited by the user are kept when
the map is rebuilt.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml

from examprep import config, llm
from examprep.ingest import ledger, parsers
from examprep.structure import analysis, embed, sections, store

Log = Callable[[str], None]


SMALL_WORDS = {"and", "of", "the", "in", "for", "to", "on", "with", "a", "an", "or", "vs"}


def _tidy(name: str) -> str:
    """'2.3 DATABASE DESIGN' -> 'Database Design' (short all-caps words like SQL stay)."""
    name = sections.strip_number(" ".join(name.split()))
    name = sections.LABEL.sub("", name).strip(" :-–") or name
    if name.isupper() and len(name) > 4:
        words = []
        for k, w in enumerate(name.split()):
            low, core = w.lower(), w.strip("[]()")
            if low in SMALL_WORDS:
                words.append(low if k else w[:1] + low[1:])
            elif len(core) <= 3 or not re.search(r"[AEIOU]", core) or re.search(r"[\d+]", core):
                words.append(w)  # SQL, DBMS, ER, B+
            else:
                words.append(w[:1] + w[1:].lower())
        name = " ".join(words)
    return name


def _fallback_name(section_heads: list[str | None], kw: list[str], title: str | None = None) -> tuple[str, str]:
    """Name used when no LLM is reachable: the chapter title or section heading(s), else key terms."""
    if title and not sections.is_generic(title):
        return _tidy(title)[:80], "heading"
    bases = [h for h in sections.base_headings(list(section_heads)) if h]
    hs = list(dict.fromkeys(_tidy(h) for h in bases if not sections.is_generic(h)))
    hs = hs or list(dict.fromkeys(_tidy(h) for h in bases))
    if hs:
        name = hs[0]
        if len(hs) > 1 and len(hs[0]) + len(hs[1]) + 5 <= 80:
            name = f"{hs[0]} and {hs[1]}"
        return name[:80], "heading"
    if kw:
        return ", ".join(kw[:3]).title(), "keywords"
    return "Untitled topic", "keywords"


def _llm_names(items: list[dict], log: Log) -> dict[int, str]:
    """items: [{"id", "sections", "keywords"}] -> {id: name}"""
    names: dict[int, str] = {}
    cfg = config.structure()["naming"]
    if not cfg.get("use_llm", True) or not llm.available_providers():
        return names
    batch = int(cfg.get("batch_size", 12))
    for start in range(0, len(items), batch):
        chunk = items[start : start + batch]
        material = json.dumps(chunk, ensure_ascii=False)
        prompt = (
            "Each item is one topic from a student's lecture slides or notes: its chapter title (if any), "
            "its section headings with the points under them, and key terms. Give each a short, specific "
            "topic name of at most 7 words, the way a syllabus would name it (for example "
            "'Deadlock Avoidance and Banker's Algorithm'). Never use course titles, instructor names "
            "or generic words like 'Introduction' alone. Use only the given material. Reply with JSON only: "
            '{"names": [{"id": <id>, "name": "<name>"}]}\n\n'
            f"<material>{material}</material>"
        )
        try:
            data = llm.chat_json([{"role": "system", "content": llm.SYSTEM_GUARD}, {"role": "user", "content": prompt}])
        except llm.LLMError as e:
            log(f"LLM naming unavailable ({e}); using slide headings instead.")
            break
        for entry in data.get("names", []) if isinstance(data, dict) else []:
            try:
                name = " ".join(str(entry["name"]).split())[:80]
                if name:
                    names[int(entry["id"])] = name
            except (KeyError, ValueError, TypeError):
                continue
    return names


def _update_ids(con, column: str, value, ids: list[int]) -> None:
    con.execute(f"UPDATE pages SET {column} = ? WHERE id IN ({','.join('?' * len(ids))})", (value, *ids))


def build(subject: str, log: Log = print) -> dict:
    started = time.time()
    subject = ledger.safe_subject(subject)
    if not ledger.db_path(subject).exists():
        raise FileNotFoundError(f"No ledger for '{subject}'. Run examprep-ingest first.")
    scfg = config.structure()["sections"]
    con = ledger.connect(subject)
    try:
        store.migrate(con)
        pages = store.load_pages(con)
        if not pages:
            raise ValueError("The ledger has no readable pages yet.")
        backend = embed.backend_name()
        log(f"{len(pages)} pages, similarity method: {backend}")
        medians = analysis.file_medians(pages)
        styles = sections.file_styles(pages)

        # User-edited names survive a rebuild (matched by file and page range).
        kept = {
            (r["file_id"], r["start_page"], r["end_page"]): r["name"]
            for r in con.execute("SELECT * FROM topics WHERE name_source = 'user'")
        }

        # 1. Decks: one per PPTX/TXT file and per notes-style file (dense pages:
        #    its chapters are found below); combined slide PDFs are split by evidence.
        files: dict[int, list[int]] = {}
        for i, p in enumerate(pages):
            files.setdefault(p.file_id, []).append(i)
        raw_vectors = None
        if any(p.kind == "pdf" for p in pages):
            raw_vectors = embed.embed([p.text for p in pages], con)
        decks = []  # (file_id, deck_no, [page indices], evidence)
        for file_id, idxs in files.items():
            fpages = [pages[i] for i in idxs]
            if fpages[0].kind == "pdf" and not styles[file_id].notes:
                assert raw_vectors is not None  # set whenever there is a PDF page
                starts, evidence = analysis.deck_starts(fpages, raw_vectors[idxs], medians.get(file_id))
                # A page whose heading is numbered 2.4, 2.4.2 ... continues a chapter: not a new deck.
                keep = [
                    k
                    for k, st in enumerate(starts)
                    if k == 0
                    or not any(h.depth >= 2 for h in sections.heading_lines(fpages[st], styles[file_id], set()))
                ]
                starts, evidence = [starts[k] for k in keep], [evidence[k] for k in keep]
            elif fpages[0].kind == "pdf":
                starts, evidence = [0], [{"page": fpages[0].page_no, "reasons": ["notes-style file"]}]
            else:
                starts, evidence = [0], [{"page": fpages[0].page_no, "reasons": ["one deck per file"]}]
            bounds = [*starts, len(idxs)]
            for d in range(len(starts)):
                decks.append((file_id, d + 1, idxs[bounds[d] : bounds[d + 1]], evidence[d]))

        # 2. Ignore repeated lines (course name, instructor, ...) and find headings.
        boiler = sections.boilerplate([[pages[i] for i in idx] for _, _, idx, _ in decks])
        lines = [sections.clean_lines(p, boiler) for p in pages]
        hs = [sections.headings(p, ln, styles.get(p.file_id), boiler) for p, ln in zip(pages, lines, strict=True)]
        heads = [h for h, _, _ in hs]
        subs = [s for _, s, _ in hs]
        hlines = [x for _, _, x in hs]
        max_words = int(scfg.get("filler_max_words", 6))
        edge_pages = {i for idxs in files.values() for i in (idxs[0], idxs[-1])}
        filler = [
            (not heads[i] and len(" ".join(lines[i]).split()) < max_words)
            or sections.is_front_matter(heads[i])
            or (i in edge_pages and len(" ".join(lines[i]).split()) < 3 * max_words)  # cover / closing page
            for i in range(len(pages))
        ]
        # In notes, a page without a heading continues the last section of the page before
        # (which may not be that page's main heading when one page holds several sections).
        sec_heads = list(heads)
        carry: str | None = None
        for i, p in enumerate(pages):
            style = styles.get(p.file_id)
            if i == 0 or pages[i - 1].file_id != p.file_id:
                carry = None
            if not (style and style.notes) or filler[i]:
                continue
            if heads[i]:
                carry = sections.last_heading(hlines[i], style, heads[i])
            elif carry:
                sec_heads[i] = carry
        clean_text = ["\n".join(ln) for ln in lines]
        vectors = embed.embed(clean_text, con)
        clean_pages = [
            store.Page(p.id, p.file_id, p.file_name, p.kind, p.page_no, p.title, t, p.layout, p.hash)
            for p, t in zip(pages, clean_text, strict=True)
        ]
        dups = analysis.near_duplicates(clean_pages)

        con.execute(
            "UPDATE pages SET heading = NULL, subheading = NULL, is_filler = NULL, duplicate_of = NULL,"
            " deck_id = NULL, topic_id = NULL"
        )
        con.execute("DELETE FROM topics")
        con.execute("DELETE FROM decks")
        for i, p in enumerate(pages):
            con.execute(
                "UPDATE pages SET heading = ?, subheading = ?, is_filler = ?, duplicate_of = ? WHERE id = ?",
                (heads[i], "; ".join(subs[i])[:500] or None, int(filler[i]), dups.get(p.id), p.id),
            )

        # 3. Topics per deck: chapters when the file has them, else sections
        #    (runs of the same heading) grouped by size and similarity, else
        #    content similarity alone when pages have no usable headings.
        use_share = float(scfg.get("use_headings_share", 0.5))
        tcfg = config.structure()["topics"]
        min_pages, max_pages = int(tcfg["min_pages"]), int(tcfg["max_pages"])
        topic_rows = []  # (deck_id, file_id, [page idx], [Section], deck idx, chapter title)
        deck_report = []
        for file_id, deck_no, idx, deck_evidence in decks:
            content = [i for i in idx if not filler[i]]
            with_head = [i for i in content if sec_heads[i]]
            title = next((heads[i] for i in idx if heads[i] and not filler[i]), None)
            cur = con.execute(
                "INSERT INTO decks (file_id, deck_no, start_page, end_page, title, evidence) VALUES (?,?,?,?,?,?)",
                (file_id, deck_no, pages[idx[0]].page_no, pages[idx[-1]].page_no, title, json.dumps(deck_evidence)),
            )
            deck_id = int(cur.lastrowid or 0)
            _update_ids(con, "deck_id", deck_id, [pages[i].id for i in idx])
            dv = vectors[idx]
            d_heads, d_subs, d_fill = [sec_heads[i] for i in idx], [subs[i] for i in idx], [filler[i] for i in idx]
            chaps, how = sections.chapters(
                [hlines[i] for i in idx], [heads[i] for i in idx], d_fill, styles[file_id].notes
            )
            groups: list[tuple[list[sections.Section], str | None]] = []
            if chaps:
                secs = sections.build_sections(d_heads, d_subs, d_fill)
                starts = [c.start for c in chaps]
                titles = [c.title for c in chaps]
                if starts[0] > 0:  # pages before the first chapter: cover, contents, preface
                    if starts[0] <= min_pages or all(d_fill[: starts[0]]):
                        starts[0] = 0
                    else:
                        starts, titles = [0, *starts], [None, *titles]
                bounds = [*starts, len(idx)]
                for k in range(len(starts)):
                    a, b = bounds[k], bounds[k + 1] - 1
                    part = [
                        sections.Section(max(s.start, a), min(s.end, b), s.heading, s.subheadings)
                        for s in secs
                        if s.end >= a and s.start <= b
                    ]
                    groups.append((part, titles[k]))
            elif content and len(with_head) >= use_share * len(content):
                how = "slide headings"
                secs = sections.build_sections(d_heads, d_subs, d_fill)
                groups = [(g, None) for g in sections.group_sections(secs, dv)]
            else:
                how = "content similarity (few headings found)"
                groups = [([sections.Section(a, b, None)], None) for a, b in analysis.segment(dv, d_heads)]
            deck_report.append(
                {
                    "file": pages[idx[0]].file_name,
                    "deck": deck_no,
                    "pages": f"{pages[idx[0]].page_no}-{pages[idx[-1]].page_no}",
                    "title": title,
                    "why": deck_evidence["reasons"],
                    "topics_from": how,
                }
            )
            for g, chapter_title in groups:
                a, b = g[0].start, g[-1].end
                for start in range(a, b + 1, max_pages):  # split very long topics by size
                    end = min(b, start + max_pages - 1)
                    part = [
                        sections.Section(max(s.start, start), min(s.end, end), s.heading, s.subheadings)
                        for s in g
                        if s.end >= start and s.start <= end
                    ]
                    t = chapter_title if start == a else None
                    topic_rows.append((deck_id, file_id, idx[start : end + 1], part, idx, t))

        # 4. Names: LLM (batched) with heading fallback.
        kws = embed.keywords(clean_text, [rows[2] for rows in topic_rows])
        items, fallback = [], {}
        for n, (_, _, _, secs, _, chapter_title) in enumerate(topic_rows):
            fallback[n] = _fallback_name([s.heading for s in secs], kws[n], chapter_title)
            item = {
                "id": n,
                "sections": [{"heading": s.heading, "points": s.subheadings[:8]} for s in secs if s.heading],
                "keywords": kws[n],
            }
            if chapter_title:
                item["chapter"] = chapter_title
            items.append(item)
        llm_names = _llm_names(items, log)

        topics_out = []
        for n, (deck_id, file_id, tidx, secs, deck_idx, _) in enumerate(topic_rows):
            sp, ep = pages[tidx[0]].page_no, pages[tidx[-1]].page_no
            if (file_id, sp, ep) in kept:
                name, source = kept[(file_id, sp, ep)], "user"
            elif n in llm_names:
                name, source = llm_names[n], "llm"
            else:
                name, source = fallback[n]
            cur = con.execute(
                "INSERT INTO topics (deck_id, file_id, topic_no, name, name_source, start_page, end_page, keywords)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (deck_id, file_id, n + 1, name, source, sp, ep, json.dumps(kws[n])),
            )
            tid = int(cur.lastrowid or 0)
            _update_ids(con, "topic_id", tid, [pages[i].id for i in tidx])
            sec_out = []
            for s in secs:
                s_sp, s_ep = pages[deck_idx[s.start]].page_no, pages[deck_idx[s.end]].page_no
                con.execute(
                    "INSERT INTO sections (topic_id, heading, start_page, end_page, subheadings) VALUES (?,?,?,?,?)",
                    (tid, s.heading, s_sp, s_ep, json.dumps(s.subheadings)),
                )
                if s.heading:
                    sec_out.append({"heading": s.heading, "pages": f"{s_sp}-{s_ep}", "points": s.subheadings})
            topics_out.append(
                {
                    "topic": n + 1,
                    "name": name,
                    "file": pages[tidx[0]].file_name,
                    "pages": f"{sp}-{ep}",
                    "named_by": source,
                    "sections": sec_out,
                }
            )
        con.commit()
        report = {
            "subject": subject,
            "seconds": round(time.time() - started, 1),
            "similarity_method": backend,
            "pages": len(pages),
            "duplicate_pages": len(dups),
            "filler_pages": sum(filler),
            "ignored_repeated_lines": len(boiler),
            "decks": deck_report,
            "topics": topics_out,
        }
        (ledger.subject_dir(subject) / "topics_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        export_yaml(con, ledger.subject_dir(subject) / "topics.yaml")
        return report
    finally:
        con.close()


# ------------------------------------------------------------ editing


def export_yaml(con, path: Path) -> None:
    rows = con.execute(
        "SELECT t.id, t.name, t.start_page, t.end_page, f.name AS file FROM topics t JOIN files f ON f.id = t.file_id"
        " ORDER BY t.topic_no"
    ).fetchall()
    data = []
    for r in rows:
        secs = []
        for s in con.execute(
            "SELECT heading, start_page, end_page, subheadings FROM sections WHERE topic_id = ? ORDER BY start_page",
            (r["id"],),
        ):
            if not s["heading"]:
                continue
            points = json.loads(s["subheadings"] or "[]")
            label = f"{s['heading']} (p.{s['start_page']}-{s['end_page']})"
            secs.append(label + (": " + "; ".join(points) if points else ""))
        entry = {"id": r["id"], "name": r["name"], "file": r["file"], "pages": f"{r['start_page']}-{r['end_page']}"}
        if secs:
            entry["sections"] = secs
        data.append(entry)
    header = (
        "# Edit only the `name` fields, then run: examprep-topics --apply-edits\n"
        "# `sections` lists the slide headings and points inside each topic (for reference).\n"
    )
    path.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")


def apply_edits(subject: str) -> int:
    """Read data/<subject>/topics.yaml and save renamed topics. Returns how many changed."""
    subject = ledger.safe_subject(subject)
    path = ledger.subject_dir(subject) / "topics.yaml"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Build the topic map first.")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    con = ledger.connect(subject)
    try:
        store.migrate(con)
        changed = 0
        for entry in data:
            try:
                tid, name = int(entry["id"]), " ".join(str(entry["name"]).split())[:80]
            except (KeyError, TypeError, ValueError):
                continue
            row = con.execute("SELECT name FROM topics WHERE id = ?", (tid,)).fetchone()
            if row and name and row["name"] != name:
                con.execute("UPDATE topics SET name = ?, name_source = 'user' WHERE id = ?", (name, tid))
                changed += 1
        con.commit()
        return changed
    finally:
        con.close()


def topic_map(subject: str) -> list[dict]:
    subject = ledger.safe_subject(subject)
    con = ledger.connect(subject)
    try:
        store.migrate(con)
        return [
            dict(r)
            for r in con.execute(
                "SELECT t.id, t.topic_no, t.name, t.name_source, f.name AS file, t.start_page, t.end_page,"
                " d.deck_no, d.title AS deck_title,"
                " (SELECT group_concat(s.heading, '; ') FROM sections s WHERE s.topic_id = t.id) AS sections"
                " FROM topics t JOIN files f ON f.id = t.file_id"
                " JOIN decks d ON d.id = t.deck_id ORDER BY t.topic_no"
            )
        ]
    finally:
        con.close()


def rename_topic(subject: str, topic_id: int, name: str) -> None:
    con = ledger.connect(ledger.safe_subject(subject))
    try:
        store.migrate(con)
        con.execute(
            "UPDATE topics SET name = ?, name_source = 'user' WHERE id = ?", (" ".join(name.split())[:80], topic_id)
        )
        con.commit()
    finally:
        con.close()


# ------------------------------------------------------------ syllabus

UNIT_LINE = re.compile(r"^\s*(unit|module|chapter)\s*[-:]?\s*([ivxlc\d]+)\b[\s:.\-–]*(.*)$", re.I)


def parse_syllabus(path: Path) -> list[tuple[str | None, str]]:
    """Split a syllabus (txt, pdf or pptx) into (unit, point) pairs."""
    from examprep.ingest import sources

    kind = sources.detect_kind(path)
    if kind == "pdf":
        text = "\n".join(p.text for p in parsers.parse_pdf(path))
    elif kind == "pptx":
        text = "\n".join(p.text for p in parsers.parse_pptx(path))
    elif kind == "txt":
        text = parsers.read_text(path)
    else:
        raise ValueError("syllabus must be a .txt, .pdf or .pptx file")
    points: list[tuple[str | None, str]] = []
    unit = None
    for raw in text.split("\n"):
        line = raw.strip(" \t•*-–·").strip()
        if not line:
            continue
        m = UNIT_LINE.match(line)
        if m:
            unit = f"{m.group(1).title()} {m.group(2).upper()}"
            line = m.group(3).strip()
            if not line:
                continue
        for part in re.split(r"[;,]|\s[–-]\s", line):
            part = part.strip(" .:")
            if (
                len(part) >= 4
                and re.search(r"[A-Za-z]{3}", part)
                and not re.fullmatch(r"\d+\s*(hours?|hrs?|periods?)", part, re.I)
            ):
                points.append((unit, part))
    return points


def map_syllabus(subject: str, syllabus: Path, log: Log = print) -> dict:
    subject = ledger.safe_subject(subject)
    points = parse_syllabus(Path(syllabus))
    if not points:
        raise ValueError("No syllabus points found in the file.")
    con = ledger.connect(subject)
    try:
        store.migrate(con)
        pages = store.load_pages(con)
        topic_of = {r["id"]: r["topic_id"] for r in con.execute("SELECT id, topic_id FROM pages")}
        if not any(topic_of.values()):
            raise ValueError("Build the topic map first (examprep-topics --subject ...).")
        backend = embed.backend_name()
        cfg = config.structure()["syllabus"]
        threshold = float(
            cfg["match_threshold_embeddings" if backend == "sentence-transformers" else "match_threshold_tfidf"]
        )
        pv, qv = embed.embed_pair([p.text for p in pages], [t for _, t in points], con)
        sims = qv @ pv.T if len(pages) else np.zeros((len(points), 0))
        con.execute("DELETE FROM syllabus_points")
        out = []
        for k, (unit, text) in enumerate(points):
            j = int(np.argmax(sims[k])) if sims.shape[1] else -1
            score = float(sims[k, j]) if j >= 0 else 0.0
            tid = topic_of.get(pages[j].id) if j >= 0 else None
            covered = score >= threshold
            con.execute(
                "INSERT INTO syllabus_points (unit, text, topic_id, score, covered) VALUES (?,?,?,?,?)",
                (unit, text, tid if covered else None, round(score, 3), int(covered)),
            )
            name = None
            if covered and tid:
                row = con.execute("SELECT name FROM topics WHERE id = ?", (tid,)).fetchone()
                name = row["name"] if row else None
            out.append(
                {
                    "unit": unit,
                    "point": text,
                    "covered": covered,
                    "topic": name,
                    "best_page": f"{pages[j].file_name} p.{pages[j].page_no}" if j >= 0 else None,
                    "score": round(score, 3),
                }
            )
        con.commit()
        report = {
            "subject": subject,
            "similarity_method": backend,
            "threshold": threshold,
            "points": len(out),
            "covered": sum(1 for o in out if o["covered"]),
            "items": out,
        }
        (ledger.subject_dir(subject) / "syllabus_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report
    finally:
        con.close()


def deck_starts_by_file(subject: str) -> dict[str, list[int]]:
    con = ledger.connect(ledger.safe_subject(subject))
    try:
        store.migrate(con)
        out: dict[str, list[int]] = {}
        for r in con.execute(
            "SELECT f.name, d.start_page FROM decks d JOIN files f ON f.id = d.file_id ORDER BY f.name, d.start_page"
        ):
            out.setdefault(r["name"], []).append(r["start_page"])
        return out
    finally:
        con.close()
