"""Streamlit frontend: upload course files and see page coverage.
Search, answers and study tools are added as they are built."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import streamlit as st

from examprep import config
from examprep.check import secret_status
from examprep.ingest import hfsync, ledger, ocr, pipeline

st.set_page_config(page_title="Exam Prep", layout="wide")
st.title("Exam Prep Assistant")

upload_tab, coverage_tab, setup_tab = st.tabs(["Upload", "Coverage", "Setup"])

with upload_tab:
    st.caption(
        "Upload any mix of .zip, .pdf, .pptx, .ppt and .txt files. Every page is recorded. "
        "Very large first uploads are faster on Kaggle (see README)."
    )
    subject = st.text_input("Subject", placeholder="e.g. operating_systems")
    files = st.file_uploader(
        "Course files", type=["zip", "pdf", "pptx", "ppt", "txt"], accept_multiple_files=True
    )
    push = st.checkbox("Save to the Hugging Face dataset repo afterwards", value=True)
    if not ocr.tesseract_available():
        st.info("Tesseract is not installed here, so scanned pages and images will be marked 'needs OCR'.")
    if st.button("Process", disabled=not (subject and files)):
        tmp = Path(tempfile.mkdtemp(prefix="examprep_ui_"))
        try:
            paths = []
            for i, f in enumerate(files):
                target = tmp / f"{i:03d}" / Path(f.name).name
                target.parent.mkdir(parents=True)
                target.write_bytes(f.getbuffer())
                paths.append(target)
            box = st.empty()
            lines: list[str] = []

            def log(msg: str) -> None:
                lines.append(msg)
                box.code("\n".join(lines[-15:]))

            with st.spinner("Processing ..."):
                report = pipeline.ingest(paths, subject, log=log)
            st.code(pipeline.format_report(report))
            if push:
                try:
                    st.success(f"Saved to {hfsync.push(subject)}")
                except hfsync.SyncError as e:
                    st.warning(str(e))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

with coverage_tab:
    subjects = sorted(p.name for p in config.data_dir().iterdir() if (p / "ledger.sqlite").exists())
    if not subjects:
        st.write("No subjects ingested yet.")
    else:
        chosen = st.selectbox("Subject", subjects)
        con = ledger.connect(chosen)
        try:
            cov = ledger.coverage(con)
        finally:
            con.close()
        by = cov["pages_by_status"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pages", cov["pages_total"])
        c2.metric("With text", by.get("ok", 0))
        c3.metric("Need OCR", by.get("needs_ocr", 0))
        c4.metric("No text", by.get("empty", 0))
        st.dataframe(cov["files"], width="stretch")
        if cov["empty_pages"]:
            st.write("Pages with no text: " + ", ".join(cov["empty_pages"]))

with setup_tab:
    st.subheader("API keys")
    for name, present in secret_status().items():
        st.write(("✅ " if present else "❌ ") + name)
    st.subheader("Trusted sites")
    st.write(", ".join(config.allowed_domains()))
