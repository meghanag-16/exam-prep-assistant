# Exam preparation assistant

An exam-preparation tool that works from your own course material. It reads every page of your slides and notes (including text inside images), records each page so nothing is silently skipped, and organises the material into decks and topics 

## What it does

- **Reads every page.** Each page, slide or text section gets its own record in a page ledger, with a coverage report that names any page where no text could be found.
- **Reads text in images.** Scanned pages and images inside PDFs and slides are read with OCR. Repeated images such as logos are read only once.
- **Handles messy uploads.** Any mix of files and zips in one go. Files are identified by their content, not their extension, and duplicates are skipped.
- **Finds the structure.** Splits a combined 200 to 300 page PDF back into its original decks, groups pages into topics with page ranges, and names each topic. Works on slides and notes from any college without per-template settings: headings are found from font size, bold text, numbering and slide titles; chapters or units become topics; repeated lines such as the course name, instructor, college and page headers are ignored; cover, contents and closing pages never become topics.
- **Extracts formulas.** Finds pages, images and scanned pages with maths (equations, formulas, matrices, algebraic expressions, inequalities, relational algebra) and converts them to LaTeX with the symbols, their meaning and the source page. Each formula's LaTeX is checked before it is stored; broken or doubtful ones are listed for review with their page.
- **Searches your material.** Cuts the pages into chunks three ways (per slide, fixed size with overlap, per heading) and searches them with keywords (BM25) and meaning (bge-small embeddings) together, merged with reciprocal rank fusion and re-scored by a cross-encoder (bge-reranker-base). Every result gives its file and page.
- **Measures its own search.** Drafts an evaluation set (questions with a known source page, plus questions outside the material), measures recall@5, MRR and latency for every chunking and search mode, and keeps the best setting.
- **Checks the syllabus.** Given a syllabus file, it lists which syllabus points are covered by the slides and which are not.
- **Incremental.** Files already processed are never processed again; adding one new file only processes that file.

## What it supports

| | |
|---|---|
| Course files | `.pdf`, `.pptx`, `.ppt`, `.txt`, `.zip` (including zips inside zips), and whole folders |
| From slides | slide titles, body text, tables, speaker notes, images |
| From PDFs | text with font sizes and bold, tables, images, scanned pages (with line sizes from OCR) |
| Layouts | lecture slides, combined slide PDFs, typed notes and book-style chapters (including formula sheets with bold capital headings), scanned slides |
| Formulas | typed maths in PDFs (maths fonts and symbols), PowerPoint equation objects, formula images and scanned pages (read by an image model), stored as LaTeX |
| Syllabus | `.txt`, `.pdf` or `.pptx` |
| LLM providers | Gemini, Groq, OpenRouter (free tiers, tried in that order), each with several models: when one model's free quota runs out or it is busy, the next is used. Requests are spaced to each model's per-minute limit, and every reply is cached in `data/llm_cache.sqlite` (synced to the dataset repo), so a repeated request never calls the API again. Set `EXAMPREP_LLM_CACHE=0` to switch the cache off. |
| Search | keyword (BM25), vector (bge-small-en-v1.5), hybrid (reciprocal rank fusion) and hybrid plus reranking (bge-reranker-base), in an embedded Qdrant |
| Storage | SQLite page ledger and Qdrant search index per subject, synced to a Hugging Face dataset repo |
| Interfaces | command line, Streamlit app, FastAPI |

## Folder structure

    config/
      models.yaml          LLM providers, embedding model
      ingest.yaml          file limits, OCR settings
      structure.yaml       duplicate, deck and topic detection settings
      formulas.yaml        formula extraction batch sizes and image settings
      search.yaml          chunking, search and evaluation settings
      exam_pattern.yaml    marks-based answer templates
      trusted_sites.yaml   sites allowed for extra explanations
    src/examprep/
      cli.py               command-line entry points
      check.py             setup check
      config.py            settings loader
      llm.py               LLM client with provider fallback
      ingest/              reading files: sources, parsers, OCR, page ledger, sync
      structure/           decks, topics, duplicates, syllabus mapping
      formulas/            formula detection, LaTeX extraction, LaTeX checks, formula tables
      search/              chunking, Qdrant index, hybrid search and reranking, evaluation set and metrics
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

**4. Extract formulas**

    uv run examprep-formulas "path\to\files_or_folder_or_zip" --subject operating_systems

Page text is read from the ledger. The course files are needed only for formula images, scanned pages and PDF pages with garbled maths text; without them images and scanned pages wait for the next run, and garbled pages are read from their text until a later run with the files replaces them. The results go to `data/<subject>/formulas.md` (formulas needing review listed first) and `formulas_report.json`. Options: `--limit 20` sends at most 20 requests, `--captions` also writes short descriptions of diagram pages, `--retry-failed`, `--pull`, `--push`. Runs are resumable: if a quota runs out, the next run continues where it stopped and nothing done is sent again.

**5. Build the search index and search**

    uv run examprep-index --subject operating_systems
    uv run examprep-search "what is the banker's algorithm" --subject operating_systems

Each chunk keeps its pages, topic and the formulas on its pages. Embeddings are cached in the ledger, so a rebuild only embeds new text. The embedding model and reranker need the `ml` extra (installed on Kaggle and the Space); without it the index is keyword-only and search says so. Options: `--chunking page|window|heading`, `--mode keyword|vector|hybrid|hybrid_rerank`, `-k 5`, `--json`.

**6. Measure search**

    uv run examprep-evalset --subject operating_systems
    uv run examprep-eval --subject operating_systems --save-best

`examprep-evalset` drafts `data/<subject>/eval/questions.yaml` with the LLM: 50 questions with their source page and 10 questions outside the material. Check and edit it by hand (add your own with `from: hand`). `examprep-eval` measures recall@5, MRR and latency for every chunking method and search mode, writes `data/<subject>/eval/retrieval_report.md`, and with `--save-best` makes the best setting the default for search. On a laptop without the `ml` extra only keyword search can be measured, so run it on Kaggle. `--download` and `--upload` move `questions.yaml` between your laptop and the dataset repo.

**7. Run the app**

    uv run streamlit run src/examprep/ui/app.py
    uv run uvicorn examprep.api.main:app --port 8000

**Large uploads on Kaggle (free).** Create a Kaggle notebook, paste `deploy/kaggle_run.py`, set `SUBJECT`, attach a private dataset with the course files, add the keys from `.env.example` as Kaggle Secrets and run. It installs the OCR tools, ingests the files, builds the topic map, extracts formulas and uploads everything to the dataset repo. Set `EXAMPREP_FORCE=1` (or `FORCE = True`) to re-read files already ingested, for example to redo OCR after an update.

**Checks**

    uv run pytest
    uv run ruff check .
    uv run ruff format --check .

To run lint and formatting automatically before each commit: `uvx pre-commit install`.

## Limitations

- Marks-based answers, notes, MCQs and past-paper analysis are not available yet.
- The evaluation questions are drafted by an LLM; numbers are only as good as the questions, so read and fix them. Bigger chunks (per heading, windows) cover more pages, which makes finding the right page easier; the report shows words and pages per chunk next to recall.
- The embedded Qdrant index can be opened by one process at a time; stop the app before running `examprep-index` on the same machine.
- Search has been tested with stand-in models only; the real bge models run on Kaggle and the Space.
- OCR reads printed text well, but handwriting, diagrams and equations poorly. Handwritten notes have not been tested. Equations on scanned pages are therefore sent as page images to the image model instead of being read from OCR text.
- Old slides often hold equations as WMF/EMF pictures; these are drawn with LibreOffice before OCR and formula reading, so they are read on Kaggle and in the Space, not on a laptop without LibreOffice.
- A reply cut off by the model's output limit is asked again with fewer pages at a time; a single page whose reply is still too long is marked failed and can be retried with `--retry-failed`.
- Formula images and scanned formula pages need a provider that accepts images (Gemini by default, `vision: true` in `config/models.yaml`). Without one they are left for a later run.
- Formula text taken from a PDF's text layer can be garbled (symbols in private font codes, fractions split over lines). Such pages (at least `garbled_symbol_chars` of these codes, `config/formulas.yaml`) are sent as page pictures to the image model instead. Without an image model or the course file, their text is used and the LaTeX may be wrong: check formulas marked for review.
- The LaTeX check is structural (braces, environments, known commands); it does not render the formula. Rendering with KaTeX comes with the website.
- Formula extraction has been tested with a fake LLM only, not yet against live providers.
- Files ingested before formula detection changed need re-reading (`--force`) so their formula pages are found.
- Scanned pages ingested before OCR line sizes were recorded need re-reading (`--force`) to get good topic headings.
- Without Tesseract installed (for example on a laptop), scanned pages and images are marked "needs OCR" until processed on Kaggle or the hosted app.
- Old `.ppt` files need LibreOffice or Microsoft PowerPoint to convert; otherwise re-save them as `.pptx`.
- Deck and topic detection are heuristic and should be checked by hand; topic names can be edited.
- Without the embedding model (the `ml` extra), a simpler similarity method is used, which is less accurate.
- Free LLM tiers have rate limits that providers change without notice, and free providers may use submitted content. Upload only material you are allowed to share.
