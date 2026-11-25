# Exam preparation assistant

An exam-preparation tool that works from your own course material. It reads every page of your slides and notes (including text inside images), records each page so nothing is silently skipped, and organises the material into decks and topics even when files have meaningless names like `unit_3.pdf` or `1.pptx`.

## What it does

- **Reads every page.** Each page, slide or text section gets its own record in a page ledger, with a coverage report that names any page where no text could be found.
- **Reads text in images.** Scanned pages and images inside PDFs and slides are read with OCR. Repeated images such as logos are read only once.
- **Handles messy uploads.** Any mix of files and zips in one go. Files are identified by their content, not their extension, and duplicates are skipped.
- **Finds the structure.** Splits a combined 200 to 300 page PDF back into its original decks, groups pages into topics with page ranges, and names each topic. Works on slides and notes from any college without per-template settings: headings are found from font size, bold text, numbering (1.2, 2.3.1) and slide titles; chapters or units become topics; repeated lines such as the course name, instructor, college and page headers are ignored; cover, contents and closing pages never become topics.
- **Checks the syllabus.** Given a syllabus file, it lists which syllabus points are covered by the slides and which are not.
- **Incremental.** Files already processed are never processed again; adding one new file only processes that file.

## What it supports

| | |
|---|---|
| Course files | `.pdf`, `.pptx`, `.ppt`, `.txt`, `.zip` (including zips inside zips), and whole folders |
| From slides | slide titles, body text, tables, speaker notes, images |
| From PDFs | text with font sizes and bold, tables, images, scanned pages (with line sizes from OCR) |
| Layouts | lecture slides, combined slide PDFs, typed notes and book-style chapters, scanned slides |
| Syllabus | `.txt`, `.pdf` or `.pptx` |
| LLM providers | Gemini, Groq, OpenRouter (free tiers, tried in that order) |
| Storage | SQLite page ledger per subject, synced to a Hugging Face dataset repo |
| Interfaces | command line, Streamlit app, FastAPI |

## Folder structure

    config/
      models.yaml          LLM providers, embedding model
      ingest.yaml          file limits, OCR settings
      structure.yaml       duplicate, deck and topic detection settings
      exam_pattern.yaml    marks-based answer templates
      trusted_sites.yaml   sites allowed for extra explanations
    src/examprep/
      cli.py               command-line entry points
      check.py             setup check
      config.py            settings loader
      llm.py               LLM client with provider fallback
      ingest/              reading files: sources, parsers, OCR, page ledger, sync
      structure/           decks, topics, duplicates, syllabus mapping
      api/                 FastAPI backend
      ui/                  Streamlit app
    deploy/kaggle_run.py   launcher script for Kaggle
    .github/workflows/     lint and tests on every push
    .pre-commit-config.yaml  the same checks before each commit
    tests/                 pytest tests
    .streamlit/            app security and privacy settings
    Dockerfile, start.sh   Hugging Face Space image

Data is written to `data/<subject>/` (not committed).

## How to run

Requirements: Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

**1. Set up**

    uv sync
    copy .env.example .env        (Windows; cp on macOS/Linux. Then fill in your keys.)
    uv run examprep-check

**2. Add course files for a subject**

    uv run examprep-ingest "path\to\files_or_folder_or_zip" --subject operating_systems
    uv run examprep-report --subject operating_systems

Options: `--push` uploads the ledger to the Hugging Face dataset repo, `--exclude "*syllabus*"` skips matching files, `--no-ocr`, `--force`.

**3. Build the topic map**

    uv run examprep-topics --subject operating_systems --syllabus path\to\syllabus.pdf

To rename topics, edit `data/<subject>/topics.yaml` and run `uv run examprep-topics --subject operating_systems --apply-edits`. `--pull` and `--push` sync with the dataset repo.

**4. Run the app**

    uv run streamlit run src/examprep/ui/app.py
    uv run uvicorn examprep.api.main:app --port 8000

**Large uploads on Kaggle (free).** Create a Kaggle notebook, paste `deploy/kaggle_run.py`, set `SUBJECT`, attach a private dataset with the course files, add the keys from `.env.example` as Kaggle Secrets and run. It installs the OCR tools, ingests the files, builds the topic map and uploads everything to the dataset repo. Set `EXAMPREP_FORCE=1` (or `FORCE = True`) to re-read files already ingested, for example to redo OCR after an update.

**Checks**

    uv run pytest
    uv run ruff check .
    uv run ruff format --check .

To run lint and formatting automatically before each commit: `uvx pre-commit install`.

## Limitations

- Search, marks-based answers, notes, MCQs and past-paper analysis are not available yet.
- OCR reads printed text well, but handwriting, diagrams and equations poorly. Handwritten notes have not been tested.
- Scanned pages ingested before OCR line sizes were recorded need re-reading (`--force`) to get good topic headings.
- Without Tesseract installed (for example on a laptop), scanned pages and images are marked "needs OCR" until processed on Kaggle or the hosted app.
- Old `.ppt` files need LibreOffice or Microsoft PowerPoint to convert; otherwise re-save them as `.pptx`.
- Deck and topic detection are heuristic and should be checked by hand; topic names can be edited.
- Without the embedding model (the `ml` extra), a simpler similarity method is used, which is less accurate.
- Free LLM tiers have rate limits that providers change without notice, and free providers may use submitted content. Upload only material you are allowed to share.
