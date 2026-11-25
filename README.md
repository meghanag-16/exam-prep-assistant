# examprep

Exam-prep assistant over your own course material. It processes every page of your PDFs, PPTs and TXT files (including OCR of images), gives marks-based answers with page citations, adds extra explanations only from trusted sites, and covers notes, MCQs, formula tables and past-paper analysis.

Status: ingestion and page ledger are working. Topics, formulas, search, answering and the study tools are being added.

## Layout

    config/            YAML settings (models, exam pattern, trusted sites, ingestion)
    src/examprep/      Python package
      config.py        settings loader (secrets from env, the rest from config/)
      check.py         setup check: `uv run examprep-check`
      cli.py           `examprep-ingest` and `examprep-report` commands
      ingest/
        sources.py     finds files in folders/zips, checks real type, removes duplicates
        parsers.py     PDF (PyMuPDF), PPTX (python-pptx), TXT readers
        convert.py     old .ppt -> .pptx
        ocr.py         Tesseract OCR in parallel (scanned pages and unique images)
        ledger.py      SQLite page ledger, one row per page/slide/section
        pipeline.py    runs everything and builds the coverage report
        hfsync.py      download/upload the ledger to the Hugging Face dataset repo
      api/             FastAPI backend
      ui/              Streamlit frontend
    deploy/kaggle_run.py   Kaggle launcher script (sets up the machine, calls the CLI)
    tests/             pytest tests
    Dockerfile, start.sh   Hugging Face Docker Space image

## Where things run

- Laptop: only editing code and running the light setup check. No OCR, no ML models.
- Kaggle: heavy ingestion (OCR, embeddings, formula extraction) on the free GPU, by running deploy/kaggle_run.py, which only calls the same CLI you run locally. No notebooks in the repo.
- Hugging Face Space: the hosted app everyone uses (FastAPI + Streamlit).
- Hugging Face Dataset repo: permanent storage the Space and Kaggle share.

## Keeping everything inside the project folder

`uv sync` installs packages only into `.venv` in this folder. uv also keeps a download cache and its own Python copy in your user folder by default. To keep those inside the project too, run this once in Command Prompt before `uv sync` (it applies to new terminals):

    setx UV_CACHE_DIR "%CD%\.uv-cache"
    setx UV_PYTHON_INSTALL_DIR "%CD%\.uv-python"

Then open a new terminal, `uv cache clean`, and `uv sync` again. Deleting the project folder removes everything.

## Local setup (Windows, uv)

    uv sync
    copy .env.example .env      (then fill in your keys)
    uv run examprep-check
    uv run pytest

Run the app locally:

    uv run uvicorn examprep.api.main:app --port 8000
    uv run streamlit run src/examprep/ui/app.py

## Ingesting course files

    uv run examprep-ingest "path\to\folder_or_files_or_zips" --subject operating_systems
    uv run examprep-report --subject operating_systems

- Accepts any mix of .zip, .pdf, .pptx, .ppt and .txt, and folders (searched recursively).
- Every page, slide or text section gets a row in data/<subject>/ledger.sqlite.
- Files already ingested are skipped (matched by content hash, not name).
- Without Tesseract (the laptop), scanned pages and images are marked `needs_ocr`.
  Running again on Kaggle with the same files fills them in.
- `--push` uploads the ledger to the dataset repo (subjects/<subject>/). On a fresh
  machine with no local ledger, it downloads the existing one first.
- `--no-ocr` skips OCR, `--force` re-reads files already done, `--pull` replaces the
  local ledger with the remote one.

The app's Upload tab and the API (`POST /ingest`, `GET /coverage/{subject}`) run the same pipeline.

## Old .ppt files

Conversion to .pptx is tried in this order:
1. LibreOffice, if installed. It is installed automatically in the Hugging Face Space and in the Kaggle notebook.
2. Your Microsoft PowerPoint on Windows, after `uv sync --extra windows-ppt`.
3. Otherwise the file is rejected with a message to re-save it as .pptx in PowerPoint.

## Secrets

Keys live only in `.env` locally, in Kaggle Secrets on Kaggle, and in Space secrets on Hugging Face. `.env` is in `.gitignore`. Never put keys in code or config files.

## Model names

Free model names and limits change. `examprep-check` lists each provider's models and warns if a configured name in `config/models.yaml` is not available.
