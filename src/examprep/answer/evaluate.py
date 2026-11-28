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

Answers are written fresh: the answer cache is not used and, with evaluation.fresh_llm_calls
(the default), neither is the LLM reply cache, so latency and tokens are real figures (this
uses API quota on each run; the launcher runs it only when something it depends on changed).
Calls answered from the cache are counted in the report.
For every answer, the report lists the pages or sites it used, so an outside question that
was answered from the slides can be checked.
Writes data/<subject>/eval/answer_report.json and answer_report.md. The report keeps a key of
everything it depends on (index, questions, settings, prompts); is_current tells whether a new
run would measure the same thing.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import statistics
import time
from collections.abc import Callable
from typing import Any

from examprep import config, llm, tracing
from examprep.answer import grounded, ragas_eval
from examprep.ingest import ledger
from examprep.search import evalset, index
from examprep.search import search as searching
from examprep.search.evaluate import _percentile

Log = Callable[[str], None]


def _share(n: int, d: int) -> float | None:
    return round(n / d, 3) if d else None


def _counts(limit: int | None, deeper: int | None) -> tuple[int, int, int]:
    cfg = config.answer()["evaluation"]
    return (
        int(limit if limit is not None else cfg.get("max_questions", 20)),
        int(deeper if deeper is not None else cfg.get("deeper_questions", 2)),
        int(cfg.get("max_outside_questions", 5)),
    )


def inputs_key(subject: str, limit: int | None = None, deeper: int | None = None) -> str:
    """Changes when the index, the questions, the chosen search setting, the answer, model or
    trusted-site settings, the answer prompt or the question counts change."""
    h = hashlib.sha256()
    h.update(str((index.read_meta(subject) or {}).get("fingerprint")).encode())
    for p in (evalset.path(subject), searching.settings_path(subject)):
        h.update(p.read_bytes() if p.exists() else b"")
    models = {k: v for k, v in config.models().items() if k != "tracing"}
    for part in (config.answer(), models, config.trusted_sites(), config.exam_pattern()):
        h.update(json.dumps(part, sort_keys=True, default=str).encode())
    h.update(repr((grounded.PROMPT_VERSION, _counts(limit, deeper))).encode())
    return h.hexdigest()[:24]


def is_current(subject: str, limit: int | None = None, deeper: int | None = None) -> bool:
    """True when the last report measured exactly what a new run would measure."""
    subject = ledger.safe_subject(subject)
    p = ledger.subject_dir(subject) / "eval" / "answer_report.json"
    try:
        report = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(report.get("inputs")) and report["inputs"] == inputs_key(subject, limit, deeper)


def run(
    subject: str,
    limit: int | None = None,
    deeper: int | None = None,
    answerer: grounded.Answerer | None = None,
    log: Log = print,
) -> dict:
    subject = ledger.safe_subject(subject)
    key = inputs_key(subject, limit, deeper)
    limit, deeper, max_outside = _counts(limit, deeper)
    data = evalset.load(subject)
    a = answerer or grounded.Answerer(subject)
    # Every answer of this run is its own trace, grouped in Langfuse as one session.
    session = f"answer-eval-{subject}-{time.strftime('%Y%m%d-%H%M%S')}"
    traced: dict[str, Any] = {"session_id": session, "tags": ["answer-eval"]}
    stack = contextlib.ExitStack()
    try:
        real = []
        for q in data["questions"]:
            pages = evalset.resolve(a.con, q.get("pages"))
            if q.get("question") and pages:
                real.append((q, pages))
        real = real[:limit]
        outside = [q for q in data["outside"] if q.get("question")][:max_outside]
        if not real and not outside:
            raise ValueError("No usable questions in questions.yaml.")
        rows, times, tokens, cached_calls, ragas_items = [], [], [], 0, []
        fresh = bool(config.answer()["evaluation"].get("fresh_llm_calls", True))
        if fresh:
            stack.enter_context(llm.fresh())
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
        if outside:
            log(f"Outside questions ({len(outside)}): each searches trusted sites, reads pages, writes an answer.")
        for n, q in enumerate(outside, start=1):
            log(f"  outside {n}/{len(outside)}: {q['question'][:70]} ...")
            r = a.answer(q["question"], q.get("marks"), use_cache=False, **traced)
            out_rows.append(
                {
                    "id": q.get("id"),
                    "question": q["question"],
                    "refused": not r["in_slides"],
                    "outside_source": r.get("outside_source"),
                    "in_syllabus": r.get("in_syllabus"),
                    # Where the answer came from: slide pages, or site links for a trusted answer.
                    "used": [x.get("where") or x.get("url") for x in r.get("sources") or []][:5],
                    "top_retrieved": [x.get("where") for x in (r.get("retrieved") or [])[:3]],
                }
            )
            times.append(r["ms"])
            cached_calls += r["tokens"]["cached_calls"]
            flag = "flagged not in slides" if not r["in_slides"] else "ANSWERED FROM SLIDES"
            log(f"    {flag}, answer from {r.get('outside_source') or 'slides'}  {r['ms']:.0f} ms")
            if r["in_slides"]:
                log(f"    pages used: {'; '.join(str(u) for u in out_rows[-1]['used']) or 'none'}")
        deep_rows = []
        if real[:deeper]:
            log(f"Extra explanations ({len(real[:deeper])} questions, with trusted-site search):")
        for n, (q, _pages) in enumerate(real[:deeper], start=1):
            log(f"  deeper {n}/{len(real[:deeper])}: {q['question'][:70]} ...")
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
        stack.close()  # RAGAS below makes its own calls
        answered = [r for r in rows if not r["refused"]]
        ragas: dict = {"status": "not installed (pip install -e .[eval])"}
        n_ragas = int(config.answer()["evaluation"].get("ragas_questions", 10))
        if ragas_items and n_ragas > 0 and ragas_eval.available():
            log(f"RAGAS: scoring {min(n_ragas, len(ragas_items))} answers (about 6 judge calls each, spaced out).")
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
            "fresh_llm_calls": fresh,
            "extra": {
                "questions": len(deep_rows),
                "beyond_points": beyond,
                "beyond_with_valid_link": _share(sum(d["beyond_with_valid_link"] for d in deep_rows), beyond),
                "gate_rejection_rate": _share(sum(d["rejected"] for d in deep_rows), cands),
                "errors": sorted({d["error"] for d in deep_rows if d["error"]}),
            },
            "model_note": "tokens are the providers' own usage figures; 0 when a reply came from the cache "
            "(fresh_llm_calls false)",
            "rows": rows,
            "outside": out_rows,
            "extra_rows": deep_rows,
            "inputs": key,
            "run_at": ledger.now(),
        }
        out = ledger.subject_dir(subject) / "eval"
        out.mkdir(parents=True, exist_ok=True)
        (out / "answer_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        (out / "answer_report.md").write_text(markdown(report), encoding="utf-8")
        return report
    finally:
        stack.close()
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
        lines += [
            "If the pages used really answer the question, the question is in the material: move it to the "
            "questions with a known page in questions.yaml.",
            "",
        ]
        lines += [
            f"- {x['id']}: {x['question']} (pages used: {'; '.join(str(u) for u in x.get('used') or []) or 'none'})"
            for x in wrong
        ]
    return "\n".join(lines) + "\n"
