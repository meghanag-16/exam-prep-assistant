"""Ingestion pipeline: inputs -> unique course files -> parsed pages -> OCR ->
page ledger -> coverage report.

The same function runs from the CLI (laptop or Kaggle) and from the app upload.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from examprep.ingest import ledger, ocr, parsers, sources
from examprep.ingest.convert import PptConversionError, ppt_to_pptx

Log = Callable[[str], None]


def _parse(src: sources.SourceFile, work_dir: Path) -> tuple[list[parsers.PageData], Path, str]:
    """Parse one file. Returns pages, the path images should be read from, and its kind."""
    if src.kind == "pdf":
        return parsers.parse_pdf(src.path), src.path, "pdf"
    if src.kind == "pptx":
        return parsers.parse_pptx(src.path), src.path, "pptx"
    if src.kind == "ppt":
        converted = ppt_to_pptx(src.path, work_dir / "converted" / src.sha256[:12])
        return parsers.parse_pptx(converted), converted, "pptx"
    if src.kind == "txt":
        return parsers.parse_txt(src.path), src.path, "txt"
    raise ValueError(f"unsupported kind {src.kind}")


def ingest(
    inputs: list[Path],
    subject: str,
    use_ocr: bool | None = None,
    force: bool = False,
    exclude: list[str] | tuple[str, ...] = (),
    log: Log = print,
) -> dict:
    """Ingest files/folders/zips into the subject's ledger and return a report."""
    started = time.time()
    subject = ledger.safe_subject(subject)
    if use_ocr is None:
        use_ocr = ocr.tesseract_available()
    elif use_ocr and not ocr.tesseract_available():
        log("Tesseract not found: OCR is skipped and those pages are marked needs_ocr.")
        use_ocr = False

    con = ledger.connect(subject)
    work_dir = Path(tempfile.mkdtemp(prefix="examprep_"))
    results: list[dict] = []
    ocr_sources: dict[int, tuple[Path, str]] = {}  # file_id -> (path to read images from, kind)

    try:
        collected = sources.collect(inputs, work_dir, exclude)
        for s in collected.skipped:
            results.append({"file": s.name, "status": "skipped", "detail": s.reason})
        log(f"{len(collected.files)} unique course file(s), {len(collected.skipped)} skipped.")

        for src in collected.files:
            existing = ledger.file_by_hash(con, src.sha256)
            if existing is not None and existing["status"] == "done" and not force:
                same = "" if existing["name"] == src.name else f" (same content as {existing['name']})"
                results.append(
                    {"file": src.name, "status": "already ingested", "detail": "unchanged, not read again" + same}
                )
                if use_ocr and src.kind in {"pdf", "pptx", "ppt"}:
                    imgs, pages = ledger.pending_ocr_for_file(con, existing["id"])
                    if imgs or pages:
                        path, kind = src.path, src.kind
                        if src.kind == "ppt":
                            try:
                                path, kind = ppt_to_pptx(src.path, work_dir / "converted" / src.sha256[:12]), "pptx"
                            except Exception:
                                continue
                        ocr_sources[existing["id"]] = (path, kind)
                continue

            t0 = time.time()
            log(f"reading {src.name} ...")
            try:
                pages, img_path, img_kind = _parse(src, work_dir)
            except PptConversionError as e:
                ledger.replace_file(con, src, "failed", 0, str(e), time.time() - t0)
                con.commit()
                results.append({"file": src.name, "status": "failed", "detail": str(e)})
                continue
            except Exception as e:
                msg = f"could not read file ({e.__class__.__name__}: {e})"[:400]
                ledger.replace_file(con, src, "failed", 0, msg, time.time() - t0)
                con.commit()
                results.append({"file": src.name, "status": "failed", "detail": msg})
                continue

            file_id = ledger.replace_file(con, src, "done", len(pages), None, time.time() - t0)
            for page in pages:
                ledger.insert_page(con, file_id, page)
            con.commit()
            if img_kind in {"pdf", "pptx"}:
                ocr_sources[file_id] = (img_path, img_kind)
            detail = f"{len(pages)} page(s)" + (f"; {src.note}" if src.note else "")
            results.append({"file": src.name, "status": "ingested", "detail": detail})

        ocr_stats = {"jobs": 0, "failed": 0, "seconds": 0.0}
        if use_ocr and ocr_sources:
            ocr_stats = _run_ocr(con, ocr_sources, log)
        ledger.refresh_page_status(con, ocr_available=use_ocr)
        con.commit()

        cov = ledger.coverage(con)
        report = {
            "subject": subject,
            "seconds": round(time.time() - started, 1),
            "ocr_enabled": use_ocr,
            "ocr": ocr_stats,
            "this_run": results,
            "coverage": cov,
        }
        con.execute(
            "INSERT INTO runs (started_at, seconds, report_json) VALUES (?,?,?)",
            (ledger.now(), report["seconds"], json.dumps(report)),
        )
        con.commit()
        (ledger.subject_dir(subject) / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        con.close()
        sources.cleanup(work_dir)


def _run_ocr(con, ocr_sources: dict[int, tuple[Path, str]], log: Log) -> dict:
    jobs: list[ocr.OcrJob] = []
    seen_imgs: set[str] = set()
    page_keys: dict[str, tuple[int, int]] = {}
    for file_id, (path, kind) in ocr_sources.items():
        imgs, pages = ledger.pending_ocr_for_file(con, file_id)
        for h, loc in imgs:
            if h in seen_imgs:
                continue
            seen_imgs.add(h)
            jobs.append(ocr.OcrJob(h, "image", str(path), kind, loc))
        if kind == "pdf":
            for page_no in pages:
                key = f"page:{file_id}:{page_no}"
                page_keys[key] = (file_id, page_no)
                jobs.append(ocr.OcrJob(key, "page", str(path), kind, str(page_no)))
    if not jobs:
        return {"jobs": 0, "failed": 0, "seconds": 0.0}

    t0 = time.time()
    n_workers = min(ocr.worker_count(), len(jobs))
    log(f"OCR: {len(jobs)} job(s) ({len(seen_imgs)} unique images, {len(page_keys)} scanned pages)")
    log(f"OCR running on {n_workers} core(s) ...")
    step = max(1, len(jobs) // 10)

    def progress(done: int, total: int) -> None:
        if done % step == 0 or done == total:
            log(f"  OCR {done}/{total}")

    results = ocr.run_jobs(jobs, progress)
    failed = 0
    for key, res in results.items():
        text, err = res[0], res[1]
        layout = res[2] if len(res) > 2 else None
        if key in page_keys:
            file_id, page_no = page_keys[key]
            con.execute(
                "UPDATE pages SET ocr_text = ?, has_formula_candidates = MAX(has_formula_candidates, ?)"
                " WHERE file_id = ? AND page_no = ?",
                (text if err is None else "", int(parsers.formula_candidate(text or "")), file_id, page_no),
            )
            if layout:  # scanned page: line sizes from OCR stand in for the missing text layer
                con.execute(
                    "UPDATE pages SET layout_json = ? WHERE file_id = ? AND page_no = ? AND layout_json IS NULL",
                    (json.dumps(layout), file_id, page_no),
                )
        else:
            con.execute(
                "UPDATE images SET ocr_text = ?, status = ? WHERE hash = ?",
                (text, "done" if err is None else "failed", key),
            )
        failed += err is not None
    con.commit()
    return {"jobs": len(jobs), "failed": failed, "seconds": round(time.time() - t0, 1)}


def format_report(report: dict) -> str:
    cov = report["coverage"]
    lines = [
        f"Subject: {report['subject']}   time: {report['seconds']}s   OCR: {'on' if report['ocr_enabled'] else 'off'}"
    ]
    lines.append("")
    lines.append("This run:")
    for r in report["this_run"]:
        lines.append(f"  [{r['status']}] {r['file']}  {r['detail']}")
    lines.append("")
    by = cov["pages_by_status"]
    lines.append(
        f"Ledger: {len(cov['files'])} file(s), {cov['pages_total']} page(s): "
        f"{by.get('ok', 0)} ok, {by.get('needs_ocr', 0)} need OCR, {by.get('empty', 0)} empty"
    )
    if report["ocr"]["jobs"]:
        o = report["ocr"]
        lines.append(f"OCR: {o['jobs']} job(s) in {o['seconds']}s, {o['failed']} failed")
    imgs = cov["unique_images_by_status"]
    if imgs:
        lines.append("Unique images: " + ", ".join(f"{v} {k}" for k, v in sorted(imgs.items())))
    lines.append(f"Formula-candidate pages: {cov['formula_candidate_pages']}")
    if cov["pages_needing_ocr"]:
        lines.append(
            f"{cov['pages_needing_ocr']} page(s) still need OCR. Run again where Tesseract is installed "
            "(Kaggle or the Space) with the same files; finished files are not read twice."
        )
    if cov["empty_pages"]:
        shown = cov["empty_pages"][:30]
        more = len(cov["empty_pages"]) - len(shown)
        lines.append("Pages with no text at all: " + ", ".join(shown) + (f" and {more} more" if more > 0 else ""))
    else:
        lines.append("Pages with no text at all: none")
    return "\n".join(lines)
