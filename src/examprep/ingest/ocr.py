"""OCR with Tesseract, run in parallel across CPU cores.

Tesseract is a system program. It is installed automatically on Kaggle (by
deploy/kaggle_run.py) and in the Hugging Face Space (Dockerfile). It is NOT
needed on the laptop: without it, pages that need OCR are marked `needs_ocr`
in the ledger and get filled in on the next run where Tesseract is available.

Only two kinds of work are sent to OCR:
- whole pages that have (almost) no text layer, i.e. scanned pages, and
- unique images above a minimum size. Images are identified by hash, so a
  logo repeated on 300 slides is read once, and images read in earlier runs
  are never read again.
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from examprep import config


@dataclass(frozen=True)
class OcrJob:
    key: str  # image hash, or "page:<file sha>:<page no>"
    what: str  # "image" | "page"
    path: str  # file to read from (the converted .pptx for .ppt files)
    file_kind: str  # pdf | pptx
    locator: str  # image locator, or page number for "page" jobs


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def worker_count() -> int:
    n = int(config.ingest()["ocr"].get("workers", 0) or 0)
    return n if n > 0 else max(1, os.cpu_count() or 1)


def _run_job(job: OcrJob, lang: str, dpi: int, timeout: int) -> tuple[str, str | None, str | None]:
    # Runs inside a worker process.
    os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # one core per Tesseract, the pool gives parallelism
    import pytesseract

    from examprep.ingest.parsers import image_bytes, render_page_png, to_pil

    try:
        if job.what == "page":
            data = render_page_png(Path(job.path), int(job.locator), dpi)
        else:
            data = image_bytes(Path(job.path), job.file_kind, job.locator)
        text = pytesseract.image_to_string(to_pil(data), lang=lang, timeout=timeout)
        return job.key, " ".join(text.split()) if text else "", None
    except Exception as e:  # one bad image must not stop the run
        return job.key, None, f"{e.__class__.__name__}: {e}"[:300]


def run_jobs(jobs: list[OcrJob], progress=None) -> dict[str, tuple[str | None, str | None]]:
    """Run OCR jobs in parallel. Returns key -> (text, error)."""
    if not jobs:
        return {}
    cfg = config.ingest()["ocr"]
    lang, dpi, timeout = cfg.get("lang", "eng"), int(cfg.get("dpi", 300)), int(cfg.get("timeout_seconds", 120))
    results: dict[str, tuple[str | None, str | None]] = {}
    workers = min(worker_count(), len(jobs))
    if workers == 1:
        for i, job in enumerate(jobs, 1):
            key, text, err = _run_job(job, lang, dpi, timeout)
            results[key] = (text, err)
            if progress:
                progress(i, len(jobs))
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_job, job, lang, dpi, timeout) for job in jobs]
        for i, fut in enumerate(as_completed(futures), 1):
            key, text, err = fut.result()
            results[key] = (text, err)
            if progress:
                progress(i, len(jobs))
    return results
