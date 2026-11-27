"""FastAPI backend: health, upload/ingest, coverage, topics, formulas, search, grounded answers,
feedback, notes and quick revision (JSON, Markdown or PDF)."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, Field

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


@app.get("/search/{subject}")
def search(subject: str, q: str, chunking: str | None = None, mode: str | None = None, k: int = 5) -> dict:
    """Search the course material: results with file, page, topic and text."""
    from examprep.search import index
    from examprep.search import search as searching

    subject = ledger.safe_subject(subject)
    if not ledger.db_path(subject).exists():
        raise HTTPException(404, f"no ledger for subject '{subject}'")
    if not index.read_meta(subject):
        raise HTTPException(404, f"no search index for subject '{subject}' (run examprep-index)")
    if not q.strip() or len(q) > 500:
        raise HTTPException(400, "q must be 1 to 500 characters")
    try:
        return searching.search(subject, q, chunking, mode, max(1, min(k, 20)))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _need_index(subject: str) -> str:
    from examprep.search import index

    subject = ledger.safe_subject(subject)
    if not ledger.db_path(subject).exists():
        raise HTTPException(404, f"no ledger for subject '{subject}'")
    if not index.read_meta(subject):
        raise HTTPException(404, f"no search index for subject '{subject}' (run examprep-index)")
    return subject


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    marks: int | None = Field(default=None, ge=1, le=20)
    deeper: bool = False
    # Groups the questions of one chat in Langfuse (Sessions view); any id the client chooses.
    session_id: str | None = Field(default=None, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")


@app.post("/ask/{subject}")
def ask(subject: str, body: AskBody) -> dict:
    """An answer built only from the course material, with checked citations, or a refusal."""
    from examprep.answer import grounded

    subject = _need_index(subject)
    try:
        return grounded.answer(subject, body.question, body.marks, body.deeper, session_id=body.session_id)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


class FeedbackBody(BaseModel):
    answer_id: int | None = None
    rating: int = Field(description="1 for thumbs up, -1 for thumbs down")
    comment: str | None = Field(default=None, max_length=2000)


@app.post("/feedback/{subject}")
def feedback(subject: str, body: FeedbackBody) -> dict:
    from examprep.answer import grounded

    subject = ledger.safe_subject(subject)
    if not ledger.db_path(subject).exists():
        raise HTTPException(404, f"no ledger for subject '{subject}'")
    try:
        return {"id": grounded.feedback(subject, body.answer_id, body.rating, body.comment)}
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _as(fmt: str, data, md_text, name: str):
    from examprep.notes import export

    if fmt == "json":
        return data
    md = md_text()
    if fmt == "md":
        return PlainTextResponse(md, media_type="text/markdown; charset=utf-8")
    if fmt == "pdf":
        with tempfile.TemporaryDirectory() as tmp:
            pdf = export.to_pdf(md, Path(tmp) / f"{name}.pdf").read_bytes()
        return Response(
            pdf, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{name}.pdf"'}
        )
    raise HTTPException(400, "format must be json, md or pdf")


@app.get("/notes/{subject}/{topic_id}")
def notes(subject: str, topic_id: int, format: str = "json"):  # noqa: A002 (query parameter name)
    """Full-coverage notes for one topic, made the first time they are asked for."""
    from examprep.notes import build, export

    subject = ledger.safe_subject(subject)
    if not ledger.db_path(subject).exists():
        raise HTTPException(404, f"no ledger for subject '{subject}'")
    try:
        data = build.topic_notes(subject, topic_id, log=lambda _m: None)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    return _as(format, data, lambda: export.notes_markdown(subject, [data]), f"notes_{subject}_{topic_id}")


@app.get("/revision/{subject}")
def revision(subject: str, topic_id: list[int] | None = Query(default=None), format: str = "json"):  # noqa: A002
    """Quick revision for some topics (topic_id, repeatable) or the whole subject."""
    from examprep.notes import export
    from examprep.notes import revision as rev

    subject = ledger.safe_subject(subject)
    if not ledger.db_path(subject).exists():
        raise HTTPException(404, f"no ledger for subject '{subject}'")
    try:
        sheets = rev.subject_revision(subject, [str(t) for t in topic_id] if topic_id else None, log=lambda _m: None)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    return _as(format, sheets, lambda: export.revision_markdown(subject, sheets), f"revision_{subject}")
