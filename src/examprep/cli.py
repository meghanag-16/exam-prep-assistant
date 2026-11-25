"""Command-line entry points. Everything the Kaggle launcher or the app does goes
through these, so the same code runs locally, on Kaggle and on the Space.

  examprep-ingest <files/folders/zips> --subject <name> [--push] [--pull] [--no-ocr] [--force]
  examprep-report --subject <name>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from examprep.ingest import hfsync, ledger, pipeline


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
    args = parser.parse_args(argv)

    subject = ledger.safe_subject(args.subject)
    if args.pull or (args.push and not ledger.db_path(subject).exists()):
        try:
            got = hfsync.pull(subject)
            print("Downloaded existing ledger from Hugging Face." if got else "No ledger on Hugging Face yet; starting fresh.")
        except hfsync.SyncError as e:
            sys.exit(str(e))

    report = pipeline.ingest(args.inputs, subject, use_ocr=False if args.no_ocr else None, force=args.force)
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
    parser.add_argument("--subject", required=True)
    parser.add_argument("--pull", action="store_true", help="download the latest ledger from Hugging Face first")
    parser.add_argument("--json", action="store_true", help="print the full summary as JSON")
    args = parser.parse_args(argv)
    subject = ledger.safe_subject(args.subject)
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
    print(
        f"{cov['pages_total']} page(s): {by.get('ok', 0)} ok, {by.get('needs_ocr', 0)} need OCR, {by.get('empty', 0)} empty"
    )
    if cov["empty_pages"]:
        print("Pages with no text: " + ", ".join(cov["empty_pages"][:50]))
