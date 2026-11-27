"""Measure retrieval on the evaluation set and pick the best settings.

For every chunking method and search mode:
- recall@k: share of questions whose source page is in the top k results.
- MRR: mean of 1 / rank of the first result holding the source page (0 when it is
  not in the first mrr_depth results).
- latency p50 and p95 per question.

For the questions outside the material, the best score each mode gives is recorded
next to the scores of real questions; a score threshold that separates them is
suggested (used later to answer "not found in your course material").

Writes data/<subject>/eval/retrieval_report.json and retrieval_report.md.
With save_best, the best chunking and mode go to data/<subject>/search_settings.json
and become the defaults for search.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable

from examprep import config
from examprep.ingest import ledger
from examprep.search import evalset, index
from examprep.search import search as searching

Log = Callable[[str], None]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (pos - lo), 1)


def threshold(inside: list[float], outside: list[float]) -> dict | None:
    """The score cut that best separates real questions (above) from outside ones (below)."""
    if not inside or not outside:
        return None
    cands = sorted(set(inside + outside))
    best = None
    for i, t in enumerate(cands):
        cut = (t + cands[i + 1]) / 2 if i + 1 < len(cands) else t + 1e-6
        right = sum(s >= cut for s in inside) + sum(s < cut for s in outside)
        acc = right / (len(inside) + len(outside))
        if best is None or acc > best["accuracy"]:
            best = {
                "threshold": round(cut, 4),
                "accuracy": round(acc, 3),
                "real_kept": round(sum(s >= cut for s in inside) / len(inside), 3),
                "outside_refused": round(sum(s < cut for s in outside) / len(outside), 3),
            }
    return best


def _top_score(hits, mode: str) -> float | None:
    if not hits:
        return None
    h = hits[0]
    return h.rerank_score if mode == "hybrid_rerank" and h.rerank_score is not None else h.score


def run(
    subject: str,
    methods: list[str] | None = None,
    modes: list[str] | None = None,
    save_best: bool = False,
    searcher: searching.Searcher | None = None,
    log: Log = print,
) -> dict:
    subject = ledger.safe_subject(subject)
    cfg = config.search()["evaluation"]
    k, depth = int(cfg.get("k", 5)), max(int(cfg.get("mrr_depth", 10)), int(cfg.get("k", 5)))
    data = evalset.load(subject)
    s = searcher or searching.Searcher(subject)
    try:
        methods = methods or list(s.meta.get("methods", {}))
        modes = modes or list(index.MODES)
        if s.encoder is None:
            log("No embedding model here: only keyword search can be measured (run on Kaggle for all modes).")
            modes = [m for m in modes if m == "keyword"]
        questions = []
        skipped = []
        for q in data["questions"]:
            pages = evalset.resolve(s.con, q.get("pages"))
            if q.get("question") and pages:
                questions.append((q, pages))
            else:
                skipped.append(q.get("id"))
        if skipped:
            log(f"Skipped {len(skipped)} question(s) whose page was not found: {', '.join(map(str, skipped))}")
        if not questions:
            raise ValueError("No usable questions in questions.yaml (each needs a question and a page that exists).")
        rows = []
        misses: dict[str, list[dict]] = {}
        for method in methods:
            for mode in modes:
                t0 = time.time()
                found, rr, times, real_scores, out_scores = 0, [], [], [], []
                ran = mode
                miss = []
                for q, pages in questions:
                    hits, info = s.search(q["question"], method, mode, depth, cut=False)
                    ran = info["mode"]
                    times.append(info["ms"])
                    rank = next((i + 1 for i, h in enumerate(hits) if pages & {p["page_id"] for p in h.pages}), None)
                    found += bool(rank and rank <= k)
                    rr.append(1 / rank if rank else 0.0)
                    if not rank or rank > k:
                        miss.append({"id": q.get("id"), "question": q["question"], "rank": rank})
                    score = _top_score(hits, ran)
                    if score is not None:
                        real_scores.append(score)
                for q in data["outside"]:
                    if not q.get("question"):
                        continue
                    hits, info = s.search(q["question"], method, mode, 1, cut=False)
                    score = _top_score(hits, info["mode"])
                    if score is not None:
                        out_scores.append(score)
                row = {
                    "chunking": method,
                    "mode": mode,
                    "ran_as": ran,
                    f"recall@{k}": round(found / len(questions), 3),
                    "mrr": round(sum(rr) / len(rr), 3),
                    "p50_ms": _percentile(times, 0.5),
                    "p95_ms": _percentile(times, 0.95),
                    "real_top_score_median": round(statistics.median(real_scores), 4) if real_scores else None,
                    "outside_top_score_median": round(statistics.median(out_scores), 4) if out_scores else None,
                    "refusal_threshold": threshold(real_scores, out_scores),
                }
                rows.append(row)
                misses[f"{method}/{mode}"] = miss
                log(
                    f"  {method:<8} {mode:<14} recall@{k} {row[f'recall@{k}']:.2f}  MRR {row['mrr']:.2f}  "
                    f"p95 {row['p95_ms']} ms  ({time.time() - t0:.0f}s)"
                )
        measured = [r for r in rows if r["ran_as"] == r["mode"]] or rows
        best = max(measured, key=lambda r: (r[f"recall@{k}"], r["mrr"], -r["p95_ms"]))
        report = {
            "subject": subject,
            "questions": len(questions),
            "outside_questions": len([q for q in data["outside"] if q.get("question")]),
            "skipped": skipped,
            "k": k,
            "index": {m: s.meta["methods"][m] for m in methods if m in s.meta.get("methods", {})},
            "model": s.meta.get("model"),
            "results": rows,
            "best": {"chunking": best["chunking"], "mode": best["mode"]},
            "misses": misses,
            "run_at": ledger.now(),
        }
        out = ledger.subject_dir(subject) / "eval"
        out.mkdir(parents=True, exist_ok=True)
        (out / "retrieval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out / "retrieval_report.md").write_text(markdown(report), encoding="utf-8")
        if save_best:
            searching.settings_path(subject).write_text(
                json.dumps(
                    {
                        **report["best"],
                        "why": f"best recall@{k}, then MRR, in the evaluation of {report['run_at']}",
                        f"recall@{k}": best[f"recall@{k}"],
                        "mrr": best["mrr"],
                        "refusal_threshold": (best.get("refusal_threshold") or {}).get("threshold"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        return report
    finally:
        if searcher is None:
            s.close()


def markdown(report: dict) -> str:
    k = report["k"]
    lines = [
        f"# Retrieval evaluation: {report['subject']}",
        "",
        f"{report['questions']} questions with a known source page, {report['outside_questions']} questions outside "
        f"the material. Embedding model: {report['model'] or 'none (keyword only)'}.",
        "",
        f"Recall@{k}: share of questions whose page is in the top {k}. MRR: how high the right page ranks "
        "(1 = always first). Latency is per question on this machine.",
        "",
        f"| Chunking | Mode | Recall@{k} | MRR | p50 ms | p95 ms | Refusal threshold (accuracy) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in report["results"]:
        t = r.get("refusal_threshold")
        thr = f"{t['threshold']} ({t['accuracy']:.0%})" if t else "-"
        mode = r["mode"] if r["ran_as"] == r["mode"] else f"{r['mode']} (ran as {r['ran_as']})"
        lines.append(
            f"| {r['chunking']} | {mode} | {r[f'recall@{k}']:.2f} | {r['mrr']:.2f} | {r['p50_ms']} | {r['p95_ms']} | "
            f"{thr} |"
        )
    lines += ["", f"Best: {report['best']['chunking']} chunks with {report['best']['mode']} search.", ""]
    lines += ["| Chunking | Chunks | Words per chunk | Pages per chunk |", "|---|---|---|---|"]
    for m, v in report["index"].items():
        lines.append(f"| {m} | {v['chunks']} | {v['avg_words']} | {v['avg_pages']} |")
    lines += [
        "",
        "Bigger chunks hold more pages, so they find the right page more easily but give the answer step less "
        "focused text; compare words per chunk along with recall.",
    ]
    best_key = f"{report['best']['chunking']}/{report['best']['mode']}"
    miss = report["misses"].get(best_key, [])
    if miss:
        lines += ["", f"## Questions the best setting missed (not in the top {k})", ""]
        for m in miss:
            where = f"rank {m['rank']}" if m["rank"] else "not found"
            lines.append(f"- {m['id']}: {m['question']} ({where})")
    return "\n".join(lines) + "\n"
