"""Coverage marker test: proof that the notes really use every page.

A copy of the subject's ledger is made (the real one is never changed). A made-up fact is
added to each of N random content pages (config/answer.yaml notes.marker_facts, 20 by
default), such as "Remember for the exam: the Varenko index of this method is 4817.",
and notes are made for the topics holding those pages. A fact counts as found when both
its made-up name and its number appear in the notes. The copy is removed afterwards.

Writes data/<subject>/notes/marker_report.json.
"""

from __future__ import annotations

import json
import random
import shutil
from collections.abc import Callable
from typing import Any

from examprep import config
from examprep.ingest import ledger
from examprep.notes import build
from examprep.search import chunks as chunking

Log = Callable[[str], None]

NAMES = (
    "Varenko Quillmar Tessaly Brondex Mirovan Calderot Pexwright Zandrel Orvane Lumisk Hadrix Fennimore "
    "Gorvath Jaxter Kelvane Nyquor Paxley Rendalo Sorvik Taliqua Ulvermon Wexford Yarrowin Zeltane Abernix "
    "Bexley Corvane Dravik Elmquist Faroden"
).split()


def run(subject: str, facts: int | None = None, seed: int = 7, log: Log = print) -> dict:
    subject = ledger.safe_subject(subject)
    facts = int(facts or config.answer()["notes"].get("marker_facts", 20))
    copy = ledger.safe_subject(f"{subject}_marker_test")
    shutil.rmtree(ledger.subject_dir(copy), ignore_errors=True)
    ledger.subject_dir(copy).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ledger.db_path(subject), ledger.db_path(copy))
    try:
        con = ledger.connect(copy)
        try:
            build.migrate(con)
            con.execute("DELETE FROM notes")
            pages = [p for p in chunking.load_pages(con) if not p.skip and p.topic_id and len(p.text.split()) >= 3]
            if not pages:
                raise ValueError("No content pages with a topic: run examprep-topics first.")
            rng = random.Random(seed)  # noqa: S311 (picks test pages, not a secret)
            chosen = rng.sample(pages, min(facts, len(pages)))
            planted: list[dict[str, Any]] = []
            for n, p in enumerate(chosen):
                name, number = NAMES[n % len(NAMES)], rng.randint(1000, 9999)
                sentence = f"Remember for the exam: the {name} index of this method is {number}."
                con.execute("UPDATE pages SET text = IFNULL(text, '') || ? WHERE id = ?", ("\n" + sentence, p.id))
                planted.append(
                    {"name": name, "number": str(number), "file": p.file, "page_no": p.page_no, "topic_id": p.topic_id}
                )
            con.commit()
            topics = list(dict.fromkeys(x["topic_id"] for x in planted))
            log(f"Planted {len(planted)} facts on pages of {len(topics)} topic(s); making their notes.")
            texts: dict[int, str] = {}
            for tid in topics:
                data = build.topic_notes(copy, tid, redo=True, log=log, con=con)
                texts[tid] = "\n".join(pt["text"] for sec in data["sections"] for pt in sec["points"]).lower()
        finally:
            con.close()
    finally:
        shutil.rmtree(ledger.subject_dir(copy), ignore_errors=True)
    for x in planted:
        text = texts.get(x["topic_id"], "")
        x["found"] = x["name"].lower() in text and x["number"] in text
    found = sum(x["found"] for x in planted)
    report = {
        "subject": subject,
        "planted": len(planted),
        "found": found,
        "missing": [{k: x[k] for k in ("name", "number", "file", "page_no")} for x in planted if not x["found"]],
        "run_at": ledger.now(),
    }
    out = ledger.subject_dir(subject) / "notes"
    out.mkdir(parents=True, exist_ok=True)
    (out / "marker_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
