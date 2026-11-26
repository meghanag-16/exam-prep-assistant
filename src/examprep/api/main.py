"""FastAPI backend: health, upload/ingest and coverage. More endpoints are added as features land."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from examprep import __version__
from examprep.ingest import ledger, pipeline

app = FastAPI(title="examprep", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/ingest")
def ingest(subject: str = Form(...), files: list[UploadFile] = File(...)) -> dict:
    """Upload any mix of .zip, .pdf, .pptx, .ppt and .txt files for one subject."""
    try:
        subject = ledger.safe_subject(subject)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    tmp = Path(tempfile.mkdtemp(prefix="examprep_upload_"))
    try:
        paths = []
        for i, f in enumerate(files):
            name = Path(f.filename or f"upload_{i}").name
            target = tmp / f"{i:03d}" / name
            target.parent.mkdir(parents=True)
            with open(target, "wb") as out:
                shutil.copyfileobj(f.file, out)
            paths.append(target)
        return pipeline.ingest(paths, subject, log=lambda _msg: None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.get("/coverage/{subject}")
def coverage(subject: str) -> dict:
    subject = ledger.safe_subject(subject)
    if not ledger.db_path(subject).exists():
        raise HTTPException(404, f"no ledger for subject '{subject}'")
    con = ledger.connect(subject)
    try:
        return ledger.coverage(con)
    finally:
        con.close()


@app.get("/topics/{subject}")
def topics(subject: str) -> list[dict]:
    from examprep.structure import build

    subject = ledger.safe_subject(subject)
    if not ledger.db_path(subject).exists():
        raise HTTPException(404, f"no ledger for subject '{subject}'")
    return build.topic_map(subject)


@app.get("/formulas/{subject}")
def formulas(subject: str, status: str | None = None) -> list[dict]:
    """Formulas found in the subject's material; status=needs_review lists the ones to check."""
    from examprep.formulas import extract

    subject = ledger.safe_subject(subject)
    if not ledger.db_path(subject).exists():
        raise HTTPException(404, f"no ledger for subject '{subject}'")
    rows = extract.formulas(subject)
    return [r for r in rows if not status or r["status"] == status]
