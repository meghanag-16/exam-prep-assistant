"""Command-line entry points. Everything the Kaggle launcher or the app does goes
through these, so the same code runs locally, on Kaggle and on the Space.

  examprep-ingest <files/folders/zips> --subject <name> [--push] [--pull] [--no-ocr] [--force] [--exclude PATTERN]
  examprep-report --subject <name>
  examprep-topics --subject <name> [--syllabus FILE] [--if-changed] [--push] [--pull] [--apply-edits] [--export]
                  [--check-boundaries FILE]
  examprep-formulas [files/folders/zips] --subject <name> [--limit N] [--captions] [--retry-failed] [--push] [--pull]
  examprep-index --subject <name> [--methods page window heading] [--if-changed] [--push] [--pull]
  examprep-search "question" --subject <name> [--chunking C] [--mode M] [-k 5] [--json] [--pull]
  examprep-evalset --subject <name> [--count 50] [--outside 10] [--redo] [--download] [--upload] [--push] [--pull]
  examprep-eval --subject <name> [--methods ...] [--modes ...] [--save-best] [--if-changed] [--push] [--pull]
  examprep-ask "question" --subject <name> [--marks 5] [--deeper] [--no-cache] [--json] [--pull]
  examprep-answer-eval --subject <name> [--limit 20] [--deeper 2] [--if-changed] [--push] [--pull]
  examprep-notes --subject <name> [--topic ID|NAME ...] [--redo] [--export md pdf] [--marker-test] [--push] [--pull]
  examprep-revise --subject <name> [--topic ID|NAME ...] [--redo] [--export md pdf] [--push] [--pull]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from examprep.ingest import hfsync, ledger, pipeline


def _resolve_subject(given: str | None) -> str:
    """Use the given subject, or the only subject that exists locally."""
    from examprep import config

    if given:
        return ledger.safe_subject(given)
    found = sorted(p.name for p in config.data_dir().iterdir() if (p / "ledger.sqlite").exists())
    if len(found) == 1:
        print(f"Subject: {found[0]}")
        return found[0]
    if not found:
        sys.exit("No subjects yet. Run examprep-ingest first, or pass --subject.")
    sys.exit("Several subjects exist, so pass --subject. Subjects: " + ", ".join(found))


def ingest_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="examprep-ingest",
        description="Read any mix of .zip, .pdf, .pptx, .ppt and .txt files into the subject's page ledger.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="files, folders (searched recursively) or zips")
    parser.add_argument("--subject", required=True, help="subject name, e.g. operating_systems")
    parser.add_argument(
        "--push",
        action="store_true",
        help="upload the ledger to the Hugging Face dataset repo afterwards "
        "(if there is no local ledger yet, the existing remote one is downloaded first)",
    )
    parser.add_argument("--pull", action="store_true", help="download the remote ledger first (replaces the local one)")
    parser.add_argument("--no-ocr", action="store_true", help="skip OCR even if Tesseract is installed")
    parser.add_argument("--force", action="store_true", help="re-read files that were already ingested")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help="skip files whose name matches, e.g. --exclude '*syllabus*' (can be repeated)",
    )
    args = parser.parse_args(argv)

    subject = ledger.safe_subject(args.subject)
    if args.pull or (args.push and not ledger.db_path(subject).exists()):
        try:
            got = hfsync.pull(subject)
            print(
                "Downloaded existing ledger from Hugging Face."
                if got
                else "No ledger on Hugging Face yet; starting fresh."
            )
        except hfsync.SyncError as e:
            sys.exit(str(e))

    report = pipeline.ingest(
        args.inputs, subject, use_ocr=False if args.no_ocr else None, force=args.force, exclude=args.exclude
    )
    print()
    print(pipeline.format_report(report))

    if args.push:
        try:
            where = hfsync.push(subject)
            print(f"\nUploaded ledger to {where}")
        except hfsync.SyncError as e:
            sys.exit(str(e))


def report_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="examprep-report", description="Show the page ledger summary for a subject.")
    parser.add_argument("--subject", help="optional when only one subject exists")
    parser.add_argument("--pull", action="store_true", help="download the latest ledger from Hugging Face first")
    parser.add_argument("--json", action="store_true", help="print the full summary as JSON")
    args = parser.parse_args(argv)
    subject = _resolve_subject(args.subject)
    if args.pull:
        try:
            hfsync.pull(subject)
        except hfsync.SyncError as e:
            sys.exit(str(e))
    if not ledger.db_path(subject).exists():
        sys.exit(f"No ledger for '{subject}' yet. Run examprep-ingest first.")
    con = ledger.connect(subject)
    try:
        cov = ledger.coverage(con)
    finally:
        con.close()
    if args.json:
        print(json.dumps(cov, indent=2))
        return
    by = cov["pages_by_status"]
    print(f"Subject: {subject}")
    for f in cov["files"]:
        extra = f" ({f['error']})" if f["error"] else ""
        print(f"  [{f['status']}] {f['name']}  {f['pages']} page(s){extra}")
    ok, need, empty = by.get("ok", 0), by.get("needs_ocr", 0), by.get("empty", 0)
    print(f"{cov['pages_total']} page(s): {ok} ok, {need} need OCR, {empty} empty")
    if cov["empty_pages"]:
        print("Pages with no text: " + ", ".join(cov["empty_pages"][:50]))


def topics_main(argv: list[str] | None = None) -> None:
    import yaml

    from examprep.structure import build

    parser = argparse.ArgumentParser(
        prog="examprep-topics",
        description="Build the topic map (decks, topics, page ranges) from the subject's page ledger.",
    )
    parser.add_argument("--subject", help="optional when only one subject exists")
    parser.add_argument("--syllabus", type=Path, help="syllabus file (.txt/.pdf/.pptx) to check coverage against")
    parser.add_argument("--pull", action="store_true", help="download the latest ledger from Hugging Face first")
    parser.add_argument(
        "--push", action="store_true", help="upload the ledger and topic map to Hugging Face afterwards"
    )
    parser.add_argument("--apply-edits", action="store_true", help="save names edited in data/<subject>/topics.yaml")
    parser.add_argument(
        "--export", action="store_true", help="write data/<subject>/topics.yaml for editing, without rebuilding"
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="keep the existing topic map (only run --syllabus / --check-boundaries)",
    )
    parser.add_argument(
        "--if-changed", action="store_true", help="only rebuild when the pages or the topic settings changed"
    )
    parser.add_argument(
        "--check-boundaries",
        type=Path,
        metavar="FILE",
        help="YAML of true deck start pages per file, to measure boundary detection accuracy",
    )
    args = parser.parse_args(argv)
    subject = _resolve_subject(args.subject)

    if args.pull or (args.push and not ledger.db_path(subject).exists()):
        try:
            if not hfsync.pull(subject):
                sys.exit(f"No ledger for '{subject}' on Hugging Face yet.")
            print("Downloaded the latest ledger from Hugging Face.")
        except hfsync.SyncError as e:
            sys.exit(str(e))
    if not ledger.db_path(subject).exists():
        sys.exit(f"No ledger for '{subject}' yet. Run examprep-ingest first.")

    if args.apply_edits:
        print(f"{build.apply_edits(subject)} topic name(s) updated.")
    elif args.export:
        con = ledger.connect(subject)
        try:
            from examprep.structure import store

            store.migrate(con)
            build.export_yaml(con, ledger.subject_dir(subject) / "topics.yaml")
        finally:
            con.close()
        print(f"Wrote {ledger.subject_dir(subject) / 'topics.yaml'}. Edit the names, then run with --apply-edits.")
    else:
        if args.if_changed and build.is_current(subject):
            print("Topic map is up to date (pages and topic settings unchanged).")
        elif not args.no_rebuild or not build.topic_map(subject):
            report = build.build(subject)
            print(f"\nSimilarity method: {report['similarity_method']}   time: {report['seconds']}s")
            print(
                f"{report['pages']} pages, {report['duplicate_pages']} near-duplicate pages (kept, marked), "
                f"{report['filler_pages']} cover/contents/closing pages, "
                f"{report['ignored_repeated_lines']} repeated lines ignored"
            )
            print("\nDecks:")
            for d in report["decks"]:
                why = ", ".join(d["why"])
                print(f"  {d['file']}  deck {d['deck']}  pages {d['pages']}  {d['title'] or ''}  ({why})")
                print(f"      topics from: {d.get('topics_from', '')}")
            print("\nTopics:")
            for t in report["topics"]:
                print(f"  {t['topic']:>3}. {t['name']}  [{t['file']} p.{t['pages']}]  ({t['named_by']})")
                for sec in t.get("sections", []):
                    points = ("; ".join(sec["points"]))[:150]
                    print(f"         - {sec['heading']} (p.{sec['pages']})" + (f": {points}" if points else ""))
            print(
                f"\nTo rename topics: edit {ledger.subject_dir(subject) / 'topics.yaml'}, then run with --apply-edits."
            )
        if args.syllabus:
            rep = build.map_syllabus(subject, args.syllabus)
            print(f"\nSyllabus: {rep['covered']} of {rep['points']} points found in the slides")
            for i in rep["items"]:
                if not i["covered"]:
                    print(f"  NOT FOUND: {(i['unit'] + ': ') if i['unit'] else ''}{i['point']}")

    if args.check_boundaries:
        truth = yaml.safe_load(args.check_boundaries.read_text(encoding="utf-8")) or {}
        from examprep.structure.analysis import boundary_accuracy

        acc = boundary_accuracy(build.deck_starts_by_file(subject), {k: list(v) for k, v in truth.items()})
        print("\nDeck boundary accuracy: " + ", ".join(f"{k}={v}" for k, v in acc.items()))

    if args.push:
        try:
            print(f"\nUploaded to {hfsync.push(subject)}")
        except hfsync.SyncError as e:
            sys.exit(str(e))


def formulas_main(argv: list[str] | None = None) -> None:
    from examprep.formulas import extract

    parser = argparse.ArgumentParser(
        prog="examprep-formulas",
        description="Extract equations, formulas, matrices and expressions as checked LaTeX.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="the ingested course files/folders/zips (needed to read formula images and scanned pages)",
    )
    parser.add_argument("--subject", help="optional when only one subject exists")
    parser.add_argument("--limit", type=int, help="at most this many items of each kind in this run (saves quota)")
    parser.add_argument("--captions", action="store_true", help="also describe diagram pages (uses image quota)")
    parser.add_argument("--retry-failed", action="store_true", help="send items that failed before again")
    parser.add_argument("--pull", action="store_true", help="download the latest ledger from Hugging Face first")
    parser.add_argument("--push", action="store_true", help="upload the ledger to Hugging Face afterwards")
    args = parser.parse_args(argv)
    subject = _resolve_subject(args.subject)
    if args.pull or (args.push and not ledger.db_path(subject).exists()):
        try:
            if not hfsync.pull(subject):
                sys.exit(f"No ledger for '{subject}' on Hugging Face yet.")
        except hfsync.SyncError as e:
            sys.exit(str(e))
    if not ledger.db_path(subject).exists():
        sys.exit(f"No ledger for '{subject}' yet. Run examprep-ingest first.")
    if args.retry_failed:
        con = ledger.connect(subject)
        try:
            from examprep.formulas import store

            store.migrate(con)
            con.execute("DELETE FROM formula_jobs WHERE status = 'failed'")
            con.commit()
        finally:
            con.close()
    rep = extract.extract(subject, args.inputs, limit=args.limit, captions=args.captions)
    run = rep["this_run"]
    print(
        f"\nThis run: {run['sent']} item(s) read, {run['formulas']} formula(s) found"
        + (f", {rep['captions_added']} diagram caption(s)" if args.captions else "")
    )
    if run["waiting_for_files"]:
        print(f"{run['waiting_for_files']} image(s)/scanned page(s) skipped: pass the course files to read them.")
    if run["no_vision_model"]:
        print(f"{run['no_vision_model']} image(s) skipped: no model that reads images (set GEMINI_API_KEY).")
    if rep["stopped"]:
        print(rep["stopped"])
    kinds = ", ".join(f"{k} {v}" for k, v in sorted(rep["by_kind"].items()))
    print(
        f"Subject total: {rep['formulas']} formula(s) ({rep['unique_formulas']} unique) on "
        f"{rep['pages_with_formulas']} page(s)" + (f": {kinds}" if kinds else "")
    )
    print(f"Needs review: {rep['needs_review']} (invalid LaTeX: {rep['invalid_latex']})")
    left = rep["still_to_do"]
    if any(left.values()):
        print(f"Still to do: {left['text']} page text(s), {left['image']} image(s), {left['page']} scanned page(s)")
    print(f"Review list: {ledger.subject_dir(subject) / 'formulas.md'}")
    if args.push:
        try:
            print(f"Uploaded to {hfsync.push(subject)}")
        except hfsync.SyncError as e:
            sys.exit(str(e))


def _pull_or_require(subject: str, pull: bool) -> None:
    if pull:
        try:
            if not hfsync.pull(subject):
                sys.exit(f"No ledger for '{subject}' on Hugging Face yet.")
        except hfsync.SyncError as e:
            sys.exit(str(e))
    if not ledger.db_path(subject).exists():
        sys.exit(f"No ledger for '{subject}' yet. Run examprep-ingest first.")


def _push(subject: str) -> None:
    try:
        print(f"Uploaded to {hfsync.push(subject)}")
    except hfsync.SyncError as e:
        sys.exit(str(e))


def index_main(argv: list[str] | None = None) -> None:
    from examprep.search import index

    parser = argparse.ArgumentParser(
        prog="examprep-index",
        description="Cut the pages into chunks (per page, fixed size with overlap, per heading) and build the "
        "search index (keyword + embeddings in an embedded Qdrant).",
    )
    parser.add_argument("--subject", help="optional when only one subject exists")
    parser.add_argument("--methods", nargs="+", choices=["page", "window", "heading"], help="default: all three")
    parser.add_argument("--if-changed", action="store_true", help="only rebuild when pages, topics or settings changed")
    parser.add_argument("--pull", action="store_true", help="download the latest ledger from Hugging Face first")
    parser.add_argument("--push", action="store_true", help="upload the ledger and index to Hugging Face afterwards")
    args = parser.parse_args(argv)
    subject = _resolve_subject(args.subject)
    _pull_or_require(subject, args.pull or (args.push and not ledger.db_path(subject).exists()))
    if args.if_changed and not index.is_stale(subject):
        print("Index is up to date.")
    else:
        rep = index.build(subject, args.methods)
        print(f"\nEmbedding model: {rep['model'] or 'none (keyword-only index: install the ml extra for vectors)'}")
        for m, v in rep["methods"].items():
            print(f"  {m:<8} {v['chunks']} chunks, {v['avg_words']} words and {v['avg_pages']} pages per chunk")
        print(f"Time: {rep['seconds']}s")
    index.close()
    if args.push:
        _push(subject)


def search_main(argv: list[str] | None = None) -> None:
    from examprep.search import index, search

    parser = argparse.ArgumentParser(prog="examprep-search", description="Search the course material.")
    parser.add_argument("query", help="the question, in quotes")
    parser.add_argument("--subject", help="optional when only one subject exists")
    parser.add_argument("--chunking", choices=["page", "window", "heading"], help="default: best from examprep-eval")
    parser.add_argument("--mode", choices=list(index.MODES), help="default: best from examprep-eval")
    parser.add_argument("-k", type=int, help="number of results (default 5)")
    parser.add_argument("--pull", action="store_true", help="download the latest ledger and index first")
    parser.add_argument("--json", action="store_true", help="print the results as JSON")
    args = parser.parse_args(argv)
    subject = _resolve_subject(args.subject)
    _pull_or_require(subject, args.pull)
    if not index.read_meta(subject):
        print("No search index yet: building it now.")
        index.build(subject)
    elif index.is_stale(subject):
        print("Note: the material changed since the index was built; run examprep-index to update it.")
    try:
        out = search.search(subject, args.query, args.chunking, args.mode, args.k)
    finally:
        index.close()
    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return
    for n in out["notes"]:
        print(n)
    print(f"{out['chunking']} chunks, {out['mode']} search, {out['ms']} ms\n")
    for i, r in enumerate(out["results"], 1):
        score = r["rerank_score"] if r["rerank_score"] is not None else r["score"]
        print(f"{i}. {r['where']}  (score {score})")
        if r["topic"]:
            print(f"   Topic: {r['topic']}")
        print("   " + " ".join(r["text"].split())[:300])
        print()
    if not out["results"]:
        print("Nothing found.")


def evalset_main(argv: list[str] | None = None) -> None:
    from examprep.search import evalset

    parser = argparse.ArgumentParser(
        prog="examprep-evalset",
        description="Manage the hand-picked evaluation questions in data/<subject>/eval/questions.yaml (known "
        "source page, plus questions outside the material): upload, download, or write a starting draft "
        "when the file does not exist yet.",
    )
    parser.add_argument("--subject", help="optional when only one subject exists")
    parser.add_argument("--count", type=int, help="questions with a source page (default 50)")
    parser.add_argument("--outside", type=int, help="questions outside the material (default 10)")
    parser.add_argument("--redo", action="store_true", help="write a new draft even if questions.yaml exists")
    parser.add_argument("--download", action="store_true", help="only download questions.yaml from Hugging Face")
    parser.add_argument("--upload", action="store_true", help="only upload your edited questions.yaml")
    parser.add_argument("--pull", action="store_true", help="download the latest ledger from Hugging Face first")
    parser.add_argument("--push", action="store_true", help="upload to Hugging Face afterwards")
    args = parser.parse_args(argv)
    subject = _resolve_subject(args.subject)
    try:
        if args.download:
            n = hfsync.pull_search(subject, overwrite_questions=True)
            p = evalset.path(subject)
            sys.exit(f"Downloaded {p}" if p.exists() else f"No questions.yaml on Hugging Face yet ({n} files).")
        if args.upload:
            print(f"Uploaded {hfsync.push_questions(subject)}")
            return
    except hfsync.SyncError as e:
        sys.exit(str(e))
    _pull_or_require(subject, args.pull or (args.push and not ledger.db_path(subject).exists()))
    if evalset.path(subject).exists() and not args.redo:
        data = evalset.load(subject)
        print(
            f"{evalset.path(subject)} already has {len(data['questions'])} question(s) and "
            f"{len(data['outside'])} outside question(s); kept (use --redo for a new draft)."
        )
    else:
        try:
            rep = evalset.generate(subject, args.count, args.outside)
        except RuntimeError as e:
            sys.exit(str(e))
        print(f"\nWrote {rep['questions']} question(s) and {rep['outside']} outside question(s) to {rep['path']}")
        print("Read them before trusting the numbers: fix or delete wrong ones, add your own.")
    if args.push:
        _push(subject)


def eval_main(argv: list[str] | None = None) -> None:
    from examprep.search import evaluate, index

    parser = argparse.ArgumentParser(
        prog="examprep-eval",
        description="Measure recall@5 and MRR of keyword, vector, hybrid and hybrid+rerank search for each "
        "chunking method, on data/<subject>/eval/questions.yaml.",
    )
    parser.add_argument("--subject", help="optional when only one subject exists")
    parser.add_argument("--methods", nargs="+", choices=["page", "window", "heading"])
    parser.add_argument("--modes", nargs="+", choices=list(index.MODES))
    parser.add_argument("--save-best", action="store_true", help="make the best setting the default for search")
    parser.add_argument(
        "--if-changed", action="store_true", help="skip when the index, questions and settings are unchanged"
    )
    parser.add_argument("--pull", action="store_true", help="download the latest ledger, index and questions first")
    parser.add_argument("--push", action="store_true", help="upload the report to Hugging Face afterwards")
    args = parser.parse_args(argv)
    subject = _resolve_subject(args.subject)
    _pull_or_require(subject, args.pull)
    if not index.read_meta(subject) or index.is_stale(subject):
        print("Index missing or out of date: building it first.")
        index.build(subject)
    from examprep.search import search as searching

    have_best = searching.settings_path(subject).exists() or not args.save_best
    if args.if_changed and have_best and evaluate.is_current(subject, args.methods, args.modes):
        index.close()
        print("Search evaluation skipped: index, questions and settings unchanged since the last report.")
        print(f"Report: {ledger.subject_dir(subject) / 'eval' / 'retrieval_report.md'}")
        return
    try:
        rep = evaluate.run(subject, args.methods, args.modes, save_best=args.save_best)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    finally:
        index.close()
    k = rep["k"]
    want = (rep["best"]["chunking"], rep["best"]["mode"])
    best = next(r for r in rep["results"] if (r["chunking"], r["mode"]) == want)
    print(
        f"\nBest: {best['chunking']} chunks, {best['mode']} search: recall@{k} {best[f'recall@{k}']:.2f}, "
        f"MRR {best['mrr']:.2f}" + (" (saved as the default)" if args.save_best else "")
    )
    print(f"Report: {ledger.subject_dir(subject) / 'eval' / 'retrieval_report.md'}")
    if args.push:
        _push(subject)


def _ensure_index(subject: str) -> None:
    from examprep.search import index

    if not index.read_meta(subject):
        print("No search index yet: building it now.")
        index.build(subject)
    elif index.is_stale(subject):
        print("Note: the material changed since the index was built; run examprep-index to update it.")


def ask_main(argv: list[str] | None = None) -> None:
    from examprep import tracing
    from examprep.answer import grounded
    from examprep.search import index

    parser = argparse.ArgumentParser(
        prog="examprep-ask",
        description="Answer a question from the course material, with checked citations. A question the "
        "slides do not cover is flagged and answered from trusted sites (or general knowledge, labelled).",
    )
    parser.add_argument("question", help="the question, in quotes")
    parser.add_argument("--subject", help="optional when only one subject exists")
    parser.add_argument(
        "--marks", type=int, help="mark level from config/exam_pattern.yaml: 2, 4, 5, 6, 7, 8, 10 or 15 (default 5)"
    )
    parser.add_argument(
        "--deeper", action="store_true", help="add extra explanations, with trusted-site passages (uses Tavily)"
    )
    parser.add_argument("--no-cache", action="store_true", help="write a new answer even if this was asked before")
    parser.add_argument("--pull", action="store_true", help="download the latest ledger and index first")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    parser.add_argument("--session", help="group questions in Langfuse under this session id (when tracing is on)")
    args = parser.parse_args(argv)
    subject = _resolve_subject(args.subject)
    _pull_or_require(subject, args.pull)
    _ensure_index(subject)
    try:
        r = grounded.answer(
            subject, args.question, args.marks, args.deeper, use_cache=not args.no_cache, session_id=args.session
        )
    except ValueError as e:
        sys.exit(str(e))
    finally:
        index.close()
    if args.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return
    for n in r.get("search", {}).get("notes", []):
        print(n)
    print(grounded.to_markdown(r))
    if not r["in_slides"]:
        print(f"(Not in the slides: {r.get('reason')})")
    else:
        c = r["checks"]
        status = "all checks passed" if not c.get("problems") else "checks still failing: " + " ".join(c["problems"])
        print(f"Citations {'valid' if c.get('citations_valid') else 'NOT valid'}; {status}")
    t = r["tokens"]
    print(
        f"Answer id {r['id']}{' (from the cache)' if r.get('cached') else ''}; {r['ms']} ms; "
        f"{t['prompt'] + t['completion']} tokens in {t['calls']} call(s)"
    )
    url = tracing.trace_url(r.get("trace_id"))
    if url:
        print(f"Trace: {url}")
    tracing.flush()


def answer_eval_main(argv: list[str] | None = None) -> None:
    from examprep.answer import evaluate as answer_eval
    from examprep.search import index

    parser = argparse.ArgumentParser(
        prog="examprep-answer-eval",
        description="Measure grounded answers on data/<subject>/eval/questions.yaml: valid citations, correct "
        "refusals, latency and tokens per answer.",
    )
    parser.add_argument("--subject", help="optional when only one subject exists")
    parser.add_argument("--limit", type=int, help="at most this many in-material questions (default 20)")
    parser.add_argument(
        "--deeper", type=int, help="questions also answered with extra explanations (default 2, uses Tavily)"
    )
    parser.add_argument(
        "--if-changed",
        action="store_true",
        help="skip when the index, questions, settings and prompts are unchanged since the last report",
    )
    parser.add_argument("--pull", action="store_true", help="download the latest ledger, index and questions first")
    parser.add_argument("--push", action="store_true", help="upload the report to Hugging Face afterwards")
    args = parser.parse_args(argv)
    subject = _resolve_subject(args.subject)
    _pull_or_require(subject, args.pull)
    _ensure_index(subject)
    if args.if_changed and answer_eval.is_current(subject, args.limit, args.deeper):
        index.close()
        print("Answer evaluation skipped: nothing it depends on changed since the last report.")
        print(f"Report: {ledger.subject_dir(subject) / 'eval' / 'answer_report.md'}")
        return
    try:
        rep = answer_eval.run(subject, args.limit, args.deeper)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))
    finally:
        index.close()

    def pct(x):
        return "-" if x is None else f"{x:.0%}"

    print(
        f"\nValid citations {pct(rep['valid_citations'])}, outside questions flagged {pct(rep['correct_refusals'])}, "
        f"real questions wrongly flagged {pct(rep['false_refusals'])}, p95 {rep['p95_ms']} ms, "
        f"{rep['tokens_per_answer_mean']} tokens per answer"
    )
    rg = rep["ragas"]
    if rg.get("status") == "ok":
        print(
            f"RAGAS ({rg['judge']}): faithfulness {rg['faithfulness']}, groundedness {rg['response_groundedness']}, "
            f"context relevance {rg['context_relevance']}"
        )
    else:
        print(f"RAGAS: {rg.get('status')}")
    print(f"Report: {ledger.subject_dir(subject) / 'eval' / 'answer_report.md'}")
    if args.push:
        _push(subject)


def notes_main(argv: list[str] | None = None) -> None:
    from examprep.notes import build, export, marker

    parser = argparse.ArgumentParser(
        prog="examprep-notes",
        description="Full-coverage notes per topic (every page checked), with Markdown and PDF export.",
    )
    parser.add_argument("--subject", help="optional when only one subject exists")
    parser.add_argument("--topic", action="append", default=[], help="topic id or part of its name (repeatable)")
    parser.add_argument("--all", action="store_true", help="every topic (the default when no --topic is given)")
    parser.add_argument("--redo", action="store_true", help="make the notes again even if stored ones are current")
    parser.add_argument("--export", nargs="+", choices=["md", "pdf"], default=["md"], help="files to write")
    parser.add_argument("--marker-test", action="store_true", help="also run the coverage marker test (20 facts)")
    parser.add_argument("--pull", action="store_true", help="download the latest ledger from Hugging Face first")
    parser.add_argument("--push", action="store_true", help="upload the ledger and notes to Hugging Face afterwards")
    parser.add_argument(
        "--push-every",
        type=int,
        default=5,
        metavar="N",
        help="with --push, also upload after every N topics, so a stopped run keeps its notes (0: only at the end)",
    )
    args = parser.parse_args(argv)
    subject = _resolve_subject(args.subject)
    _pull_or_require(subject, args.pull or (args.push and not ledger.db_path(subject).exists()))

    def progress(done: int, total: int) -> None:
        if args.push and args.push_every > 0 and done % args.push_every == 0 and done < total:
            try:
                hfsync.push(subject, message=f"notes progress {done}/{total} for {subject}")
                print(f"  uploaded notes so far ({done} of {total} topics)")
            except hfsync.SyncError as e:
                print(f"  could not upload progress ({e}); continuing")

    try:
        out = build.subject_notes(subject, args.topic or None, args.redo, after_topic=progress)
    except ValueError as e:
        sys.exit(str(e))
    rep = out["report"]
    md = export.notes_markdown(subject, out["notes"], rep)
    name = "notes" if not args.topic else "notes_" + "_".join(str(n["topic_id"]) for n in out["notes"])
    files = export.write(md, ledger.subject_dir(subject) / "notes" / name, args.export)
    print(
        f"\nCoverage: {rep['covered_pages']} of {rep['content_pages']} content pages ({rep['coverage']:.0%}) "
        f"in {rep['topics']} topic(s); {rep['llm_calls']} model call(s)"
    )
    for u in rep["uncovered"]:
        print(f"  not covered: {u['file']} p.{u['page_no']} ({u['topic']})")
    if rep["thin"]:
        print(f"  {len(rep['thin'])} page(s) covered briefly (see notes_report.json)")
    for f in files:
        print(f"Wrote {f}")
    if args.marker_test:
        m = marker.run(subject)
        print(f"Marker test: {m['found']} of {m['planted']} planted facts found in the notes")
        for x in m["missing"]:
            print(f"  missing: {x['name']} {x['number']} ({x['file']} p.{x['page_no']})")
    if args.push:
        _push(subject)


def revise_main(argv: list[str] | None = None) -> None:
    from examprep.notes import export, revision

    parser = argparse.ArgumentParser(
        prog="examprep-revise",
        description="Quick revision sheets: keywords, short points, comparison tables and formula tables.",
    )
    parser.add_argument("--subject", help="optional when only one subject exists")
    parser.add_argument("--topic", action="append", default=[], help="topic id or part of its name (repeatable)")
    parser.add_argument("--redo", action="store_true", help="make the sheets again even if stored ones are current")
    parser.add_argument("--export", nargs="+", choices=["md", "pdf"], default=["md"], help="files to write")
    parser.add_argument("--pull", action="store_true", help="download the latest ledger from Hugging Face first")
    parser.add_argument("--push", action="store_true", help="upload the ledger and sheets to Hugging Face afterwards")
    args = parser.parse_args(argv)
    subject = _resolve_subject(args.subject)
    _pull_or_require(subject, args.pull or (args.push and not ledger.db_path(subject).exists()))
    try:
        sheets = revision.subject_revision(subject, args.topic or None, args.redo)
    except ValueError as e:
        sys.exit(str(e))
    title = subject if not args.topic else ", ".join(s["topic"] for s in sheets)
    name = "revision" if not args.topic else "revision_" + "_".join(str(s["topic_id"]) for s in sheets)
    files = export.write(
        export.revision_markdown(subject, sheets, title), ledger.subject_dir(subject) / "notes" / name, args.export
    )
    n_f = sum(len(s["formulas"]) for s in sheets)
    n_ex = sum(1 for s in sheets for f in s["formulas"] if f.get("example"))
    print(f"\n{len(sheets)} topic(s), {n_f} formula(s), {n_ex} worked example(s) computed with SymPy")
    for f in files:
        print(f"Wrote {f}")
    if args.push:
        _push(subject)
