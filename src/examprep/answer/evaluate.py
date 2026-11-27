"""Measure grounded answers on the evaluation set (data/<subject>/eval/questions.yaml).

For the questions with a known source page:
- answered from the slides: share not flagged "not in your slides" (the rest are false flags).
- valid citations: share of answers whose citations all name a passage that was given
  and where every sentence or point that states something carries a citation.
- cites the source page: share of answers citing a passage that holds the known page.
- passed all checks: valid citations, anti-generic check and length.
- RAGAS (where installed, on up to evaluation.ragas_questions answers): faithfulness,
  response groundedness and context relevance (answer/ragas_eval.py).
For the questions outside the material:
- correctly flagged: share answered with "This is not in your slides" (they then get an
  outside answer: trusted sites or general knowledge, counted by source).
Plus p50 and p95 latency and tokens per answer (from the providers' usage figures).

With extra explanations (a few questions, they use Tavily credits): the share of
"beyond your slides" points with a valid trusted link, and how often the relevance gate
rejects passages.

Answers are written fresh (the answer cache is not used); the LLM reply cache still
applies, and calls it answered are counted, since they make latency look better.
Writes data/<subject>/eval/answer_report.json and answer_report.md.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from typing import Any

from examprep import config, tracing
from examprep.answer import grounded, ragas_eval
from examprep.ingest import ledger
from examprep.search import evalset
from examprep.search.evaluate import _percentile

Log = Callable[[str], None]


def _share(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


def run(
    subject: str,
    limit: int | None = None,
    deeper: int | None = None,
    answerer: grounded.Answerer | None = None,
    log: Log = print,
) -> dict:
    subject = ledger.safe_subject(subject)
    cfg = config.answer()["evaluation"]
    limit = int(limit if limit is not None else cfg.get("max_questions", 50))
    deeper = int(deeper if deeper is not None else cfg.get("deeper_questions", 5))
    data = evalset.load(subject)
    a = answerer or grounded.Answerer(subject)
    # Every answer of this run is its own trace, grouped in Langfuse as one session.
    session = f"answer-eval-{subject}-{time.strftime('%Y%m%d-%H%M%S')}"
    traced: dict[str, Any] = {"session_id": session, "tags": ["answer-eval"]}
    try:
        real = []
        for q in data["questions"]:
            pages = evalset.resolve(a.con, q.get("pages"))
            if q.get("question") and pages:
                real.append((q, pages))
        real = real[:limit]
        outside = [q for q in data["outside"] if q.get("question")]
        if not real and not outside:
            raise ValueError("No usable questions in questions.yaml.")
        rows, times, tokens, cached_calls, ragas_items = [], [], [], 0, []
        for n, (q, pages) in enumerate(real, start=1):
            r = a.answer(q["question"], q.get("marks"), use_cache=False, **traced)
            inside = r["in_slides"]
            held = {p["page_id"] for s in r["sources"] for p in s.get("pages", [])} if inside else set()
            row = {
                "id": q.get("id"),
                "question": q["question"],
                "refused": not inside,
                "reason": r.get("reason"),
                "citations_valid": bool(r["checks"].get("citations_valid")) if inside else None,
                "cites_source_page": bool(pages & held) if inside else None,
                "passed_checks": (not r["checks"].get("problems")) if inside else None,
                "term_share": r["checks"].get("term_share"),
                "attempts": r["attempts"],
                "ms": r["ms"],
                "tokens": r["tokens"]["prompt"] + r["tokens"]["completion"],
            }
            rows.append(row)
            if inside and r["sources"]:
                ragas_items.append(
                    {
                        "question": q["question"],
                        "answer": grounded.strip_labels(r.get("answer_with_labels") or r["answer"]),
                        "contexts": [s["text"] for s in r["sources"]],
                        "trace_id": r.get("trace_id"),
                    }
                )
            times.append(r["ms"])
            tokens.append(row["tokens"])
            cached_calls += r["tokens"]["cached_calls"]
            log(
                f"  {n}/{len(real)} {'NOT IN SLIDES' if not inside else 'answered'}"
                f"{'' if not inside else ', citations ' + ('ok' if row['citations_valid'] else 'NOT ok')}"
                f"  {r['ms']:.0f} ms"
            )
        out_rows = []
        for q in outside:
            r = a.answer(q["question"], q.get("marks"), use_cache=False, **traced)
            out_rows.append(
                {
                    "id": q.get("id"),
                    "question": q["question"],
                    "refused": not r["in_slides"],
                    "outside_source": r.get("outside_source"),
                    "in_syllabus": r.get("in_syllabus"),
                }
            )
            times.append(r["ms"])
            cached_calls += r["tokens"]["cached_calls"]
            flag = "flagged not in slides" if not r["in_slides"] else "ANSWERED FROM SLIDES"
            log(f"  outside: {flag}  {q['question'][:70]}")
        deep_rows = []
        for q, _pages in real[:deeper]:
            r = a.answer(q["question"], q.get("marks"), deeper=True, use_cache=False, **traced)
            c, info = r["checks"], r.get("web_info") or {}
            deep_rows.append(
                {
                    "id": q.get("id"),
                    "beyond_points": c.get("beyond_points", 0),
                    "beyond_with_valid_link": c.get("beyond_with_valid_link", 0),
                    "dropped_points": c.get("dropped_points", 0),
                    "candidates": info.get("candidates", 0),
                    "rejected": info.get("rejected", 0),
                    "error": info.get("error"),
                }
            )
        answered = [r for r in rows if not r["refused"]]
        ragas: dict = {"status": "not installed (pip install -e .[eval])"}
        n_ragas = int(config.answer()["evaluation"].get("ragas_questions", 20))
        if ragas_items and n_ragas > 0 and ragas_eval.available():
            try:
                res = ragas_eval.score(ragas_items[:n_ragas], log)
                ragas = {
                    "status": "ok",
                    "judge": res["judge"],
                    "answers": len(res["rows"]),
                    **ragas_eval.means(res["rows"]),
                }
                if res["errors"]:
                    ragas["errors"] = res["errors"]
                for it, scores in zip(ragas_items, res["rows"], strict=False):  # shown on each answer's trace
                    for name, value in scores.items():
                        if value is not None and it.get("trace_id"):
                            tracing.score(f"ragas-{name.replace('_', '-')}", value, trace_id=it["trace_id"])
            except Exception as e:  # noqa: BLE001 (RAGAS problems must not lose the other numbers)
                ragas = {"status": f"failed: {e.__class__.__name__}: {str(e)[:160]}"}
        elif not ragas_items:
            ragas = {"status": "no answers to score"}
        beyond = sum(d["beyond_points"] for d in deep_rows)
        cands = sum(d["candidates"] for d in deep_rows)
        report = {
            "subject": subject,
            "questions": len(rows),
            "outside_questions": len(out_rows),
            "answered": _share(len(answered), len(rows)),
            "false_refusals": _share(len(rows) - len(answered), len(rows)),
            "valid_citations": _share(sum(bool(r["citations_valid"]) for r in answered), len(answered)),
            "cites_source_page": _share(sum(bool(r["cites_source_page"]) for r in answered), len(answered)),
            "passed_all_checks": _share(sum(bool(r["passed_checks"]) for r in answered), len(answered)),
            "correct_refusals": _share(sum(r["refused"] for r in out_rows), len(out_rows)),
            "outside_answer_sources": {
                src or "none": sum(1 for r in out_rows if r["refused"] and r["outside_source"] == src)
                for src in ("trusted", "general", None)
            },
            "outside_not_in_syllabus": _share(
                sum(r["in_syllabus"] is False for r in out_rows), sum(r["in_syllabus"] is not None for r in out_rows)
            ),
            "ragas": ragas,
            "p50_ms": _percentile(times, 0.5),
            "p95_ms": _percentile(times, 0.95),
            "tokens_per_answer_mean": round(statistics.mean(tokens), 1) if tokens else None,
            "tokens_per_answer_p95": _percentile([float(t) for t in tokens], 0.95),
            "mean_attempts": round(statistics.mean(r["attempts"] for r in answered), 2) if answered else None,
            "llm_calls_from_cache": cached_calls,
            "extra": {
                "questions": len(deep_rows),
                "beyond_points": beyond,
                "beyond_with_valid_link": _share(sum(d["beyond_with_valid_link"] for d in deep_rows), beyond),
                "gate_rejection_rate": _share(sum(d["rejected"] for d in deep_rows), cands),
                "errors": sorted({d["error"] for d in deep_rows if d["error"]}),
            },
            "model_note": "tokens are the providers' own usage figures; 0 when a reply came from the cache",
            "rows": rows,
            "outside": out_rows,
            "extra_rows": deep_rows,
            "run_at": ledger.now(),
        }
        out = ledger.subject_dir(subject) / "eval"
        out.mkdir(parents=True, exist_ok=True)
        (out / "answer_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "answer_report.md").write_text(markdown(report), encoding="utf-8")
        return report
    finally:
        if answerer is None:
            a.close()


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def markdown(r: dict) -> str:
    e = r["extra"]
    lines = [
        f"# Answer evaluation: {r['subject']}",
        "",
        f"{r['questions']} questions with a known source page, {r['outside_questions']} outside the material.",
        "",
        "| Measure | Value |",
        "|---|---|",
        f"| Answered from the slides | {_pct(r['answered'])} |",
        f"| Answers with valid citations | {_pct(r['valid_citations'])} |",
        f"| Answers citing the known source page | {_pct(r['cites_source_page'])} |",
        f"| Answers passing every check | {_pct(r['passed_all_checks'])} |",
        f'| Outside questions flagged "not in your slides" | {_pct(r["correct_refusals"])} |',
        f'| Real questions wrongly flagged "not in your slides" | {_pct(r["false_refusals"])} |',
        f'| Outside questions flagged "not in your syllabus" | {_pct(r["outside_not_in_syllabus"])} |',
        f"| Latency p50 / p95 | {r['p50_ms']} ms / {r['p95_ms']} ms |",
        f"| Tokens per answer, mean / p95 | {r['tokens_per_answer_mean']} / {r['tokens_per_answer_p95']} |",
        f"| Mean attempts per answer | {r['mean_attempts']} |",
        f"| LLM calls answered from the reply cache | {r['llm_calls_from_cache']} |",
        "",
        "Valid citations: every citation names a passage that was retrieved, and every statement has one.",
        "",
        "## RAGAS",
        "",
        *(
            [
                f"Judge model {r['ragas']['judge']}, {r['ragas']['answers']} answers.",
                "",
                f"- Faithfulness (claims supported by the passages): {r['ragas'].get('faithfulness')}",
                f"- Response groundedness: {r['ragas'].get('response_groundedness')}",
                f"- Context relevance (passages relevant to the question): {r['ragas'].get('context_relevance')}",
            ]
            if r["ragas"].get("status") == "ok"
            else [f"Not measured: {r['ragas'].get('status')}."]
        ),
        "",
        "Outside questions answered from: "
        + ", ".join(f"{k} {v}" for k, v in r["outside_answer_sources"].items())
        + " (trusted = trusted sites, general = model knowledge, none = no answer).",
        "",
        "## Extra explanations",
        "",
        f"{e['questions']} questions, {e['beyond_points']} points beyond the slides.",
        "",
        f"- Beyond-the-slides points with a valid trusted link: {_pct(e['beyond_with_valid_link'])}",
        f"- Trusted passages rejected by the relevance gate: {_pct(e['gate_rejection_rate'])}",
    ]
    if e["errors"]:
        lines.append("- Errors: " + "; ".join(e["errors"]))
    bad = [x for x in r["rows"] if x["refused"] or x["citations_valid"] is False]
    if bad:
        lines += ["", "## Answers to look at", ""]
        for x in bad:
            why = f"flagged not in slides: {x['reason']}" if x["refused"] else "citations not valid"
            lines.append(f"- {x['id']}: {x['question']} ({why})")
    wrong = [x for x in r["outside"] if not x["refused"]]
    if wrong:
        lines += ["", "## Outside questions answered as if they were in the slides", ""]
        lines += [f"- {x['id']}: {x['question']}" for x in wrong]
    return "\n".join(lines) + "\n"
