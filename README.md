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
- **Answers from your material first.** Marks-based answers for 2, 4, 5, 6, 7, 8, 10 and 15 marks, each with its own length and structure (`config/exam_pattern.yaml`), built only from the passages search kept for the question, citing file and page. Code checks that every citation names a passage that was retrieved, that every statement has a citation, that the answer uses the passages' own terms (not a generic answer) and that its length suits the mark level; a failed check goes back to the model for another try. Repeated questions are answered from a cache.
- **Questions not in your slides.** The answer starts with "This is not in your slides." and is still given in the same marks template: from allowlisted sites with their links when they have it, otherwise from the model's general knowledge, marked as such (`outside_answer` in `config/answer.yaml`). When a syllabus was given (`examprep-topics --syllabus`) and the question is not on it, the answer also says "This is not in your syllabus."
- **Extra explanations from trusted sites.** Optional intuition, examples, common mistakes and exam tips, with "From your slides" and "Beyond your slides" always kept apart. Outside passages come only from allowlisted sites (`config/trusted_sites.yaml`, searched with Tavily, then Exa, then DuckDuckGo when the one before fails or finds nothing, text extracted with trafilatura) and must pass a relevance gate against the question and the slides; a point beyond the slides is kept only with a valid site link. A self-check pass removes statements the passages do not support. Thumbs up or down feedback is logged.
- **Full-coverage notes.** Notes per topic, made a few pages at a time, with a code check that every page is used and that its key terms appear; missed or thin pages are sent again and anything still missing is reported. A marker test plants made-up facts on 20 random pages and checks that all of them reach the notes.
- **Quick revision.** Keywords, short points, comparison tables and formula tables (meaning, symbols, units, when to use, a worked example computed with SymPy, source page) for a topic or a whole subject, exported to Markdown and PDF.
- **Checks the syllabus.** Given a syllabus file, it lists which syllabus points are covered by the slides and which are not.
- **Measures its own answers.** Citation and refusal figures from code, plus RAGAS faithfulness, groundedness and context relevance scored by an LLM judge.
- **Traces every step.** With Langfuse keys set, every answer and notes run is a Langfuse trace with its steps nested inside: searches (retriever), syllabus and answer checks (evaluator, guardrail), web searches, and every LLM call (generation, with model, parameters, tokens, latency and cache hits). Traces carry the subject, mark level and feature tags, chat questions are grouped by session, and scores (checks passed, in slides, user thumbs up or down, RAGAS) are attached to each answer's trace. E-mail addresses, phone numbers and API keys are removed before anything is sent; `tracing: log_content: false` in `config/models.yaml` also leaves out all course text.
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
| LLM providers | Groq, OpenRouter, NVIDIA (optional), Gemma and Gemini (free tiers, tried in that order for text; images always go to Gemini, whose daily quota is small), each with several models: when one model's free quota runs out or it is busy, the next is used. Requests are spaced to each model's per-minute limit, and every reply is cached in `data/llm_cache.sqlite` (synced to the dataset repo), so a repeated request never calls the API again. Set `EXAMPREP_LLM_CACHE=0` to switch the cache off. |
| Search | keyword (BM25), vector (bge-small-en-v1.5), hybrid (reciprocal rank fusion) and hybrid plus reranking (bge-reranker-base), in an embedded Qdrant |
| Answers | marks-based (2, 4, 5, 6, 7, 8, 10, 15), checked citations, "not in your slides" and "not in your syllabus" notices with an outside answer, optional extra explanations from allowlisted sites, answer cache, feedback log; the flow runs as a LangGraph graph |
| Web search | Tavily, then Exa, then DuckDuckGo (no key), limited to allowlisted sites (checked again in code). Tavily and Exa have free monthly credits with no card. Brave Search can be switched on but its free credit needs a card on file. Google's Custom Search JSON API is not used: it is closed to new customers and Google has announced it ends on 1 January 2027 |
| Evaluation and tracing | RAGAS (answer quality), Langfuse (traces), mypy (type checks in CI) |
| Exports | Markdown and PDF for notes and revision sheets (PDF made with WeasyPrint, or PyMuPDF where WeasyPrint is not installed) |
| Storage | SQLite page ledger and Qdrant search index per subject, synced to a Hugging Face dataset repo |
| Interfaces | command line, Streamlit app, FastAPI |

## Folder structure

    config/
      models.yaml          LLM providers, embedding model, tracing
      ingest.yaml          file limits, OCR settings
      structure.yaml       duplicate, deck and topic detection settings
      formulas.yaml        formula extraction batch sizes and image settings
      search.yaml          chunking, search and evaluation settings
      exam_pattern.yaml    marks-based answer templates
      answer.yaml          answer checks, trusted-site, notes and revision settings
      answer_examples.yaml example answer for each mark level (shape and citation style)
      trusted_sites.yaml   sites allowed for outside answers and extra explanations, search providers
    src/examprep/
      cli.py               command-line entry points
      check.py             setup check
      config.py            settings loader
      llm.py               LLM client with provider fallback
      tracing.py           Langfuse tracing: traces, sessions, tags, scores, masking (off without keys)
      ingest/              reading files: sources, parsers, OCR, page ledger, sync
      structure/           decks, topics, duplicates, syllabus mapping
      formulas/            formula detection, LaTeX extraction, LaTeX checks, formula tables
      search/              chunking, Qdrant index, hybrid search and reranking, evaluation set and metrics
      answer/              grounded answers, citation and anti-generic checks, trusted sites, answer evaluation (with RAGAS)
      notes/               full-coverage notes, marker test, quick revision, Markdown and PDF export
      api/                 FastAPI backend
      ui/                  Streamlit app
    deploy/kaggle_run.py   launcher script for Kaggle
    .github/workflows/     lint, type check and tests on every push
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

**7. Ask questions**

    uv run examprep-ask "explain the banker's algorithm" --subject operating_systems --marks 10
    uv run examprep-ask "explain the banker's algorithm" --subject operating_systems --deeper

The answer lists its sources (file and page) and says whether every check passed. A question not in the slides gets "This is not in your slides." and an answer from trusted sites (Tavily with `TAVILY_API_KEY`, then Exa with `EXA_API_KEY`, then DuckDuckGo, which needs no key) or, failing that, from general knowledge; with a syllabus, a question not on it also gets "This is not in your syllabus." Set `outside_answer: none` in `config/answer.yaml` to only say it is not in the slides, or `trusted` to allow trusted sites but not general knowledge. `--deeper` adds extra explanations, using trusted sites for anything beyond the slides. Options: `--marks 2|4|5|6|7|8|10|15`, `--no-cache`, `--json`, `--session <id>` (groups questions in Langfuse). Edit the templates in `config/exam_pattern.yaml` and the example answers in `config/answer_examples.yaml` to change the shape of answers.

**8. Measure answers**

    uv run examprep-answer-eval --subject operating_systems

Answers the questions in `questions.yaml` and writes `data/<subject>/eval/answer_report.md`: share of answers with valid citations, share citing the known source page, outside questions correctly marked "not in your slides" (and where their answer came from), in-slide questions wrongly marked, p50 and p95 latency and tokens per answer, and RAGAS faithfulness, response groundedness and context relevance for 20 answers (`ragas_questions` in `config/answer.yaml`; needs the `eval` extra, installed on Kaggle). Five questions are also answered with extra explanations, to measure how many points beyond the slides carry a valid trusted link and how often the relevance gate rejects passages. Options: `--limit 50`, `--deeper 5`.

**9. Notes and quick revision**

    uv run examprep-notes --subject operating_systems --export md pdf --marker-test
    uv run examprep-revise --subject operating_systems --export md pdf

Files go to `data/<subject>/notes/` (`notes.md`, `notes.pdf`, `revision.md`, `revision.pdf`, `notes_report.json`, `marker_report.json`). Notes are stored in the ledger and made again only when a topic's pages, the topic map or the settings change. `--topic` (an id or part of a name, repeatable) limits the run to some topics; `--redo` makes them again.

**10. Run the app**

    uv run streamlit run src/examprep/ui/app.py
    uv run uvicorn examprep.api.main:app --port 8000

**Large uploads on Kaggle (free).** Create a Kaggle notebook, paste `deploy/kaggle_run.py`, set `SUBJECT`, attach a private dataset with the course files, add the keys from `.env.example` as Kaggle Secrets (Exa and Langfuse keys are optional) and run. It installs the OCR tools and the `ml`, `eval` and `pdf` extras, ingests the files, builds the topic map, extracts formulas, builds the search index, measures search and answers, makes the notes and revision sheets (with the marker test) and uploads everything to the dataset repo. `EXAMPREP_EVAL`, `EXAMPREP_ANSWER_EVAL`, `EXAMPREP_NOTES` and `EXAMPREP_MARKER` set to `0` skip those steps. Set `EXAMPREP_FORCE=1` (or `FORCE = True`) to re-read files already ingested, for example to redo OCR after an update.

**Checks**

    uv run pytest
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy

To run lint, formatting and type checks automatically before each commit: `uvx pre-commit install`.

**Tracing (optional).** Create a free project at cloud.langfuse.com, put `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST` in `.env` (and in Kaggle Secrets or the Space secrets), and set `LANGFUSE_TRACING_ENVIRONMENT` to `laptop`, `kaggle` or `space` to tell runs apart. `examprep-ask` prints the link to its trace. In Langfuse: Traces shows each answer (`answer-question`) and notes run (`make-topic-notes`); Sessions groups a chat's questions (`--session`, or `session_id` in `POST /ask`) and each answer evaluation run; Scores holds checks passed, in slides, user feedback and RAGAS. `tracing.sample_rate` in `config/models.yaml` sends only a share of traces.

## Limitations

- MCQs, practice papers and past-paper analysis are not available yet.
- Answers, notes and revision sheets have been tested with a fake LLM only, not yet against live providers. Exa, DuckDuckGo, Langfuse and RAGAS have been tested with stand-ins only (Langfuse's SDK was run locally to check what each trace contains, without a server). The checks catch invalid or missing citations and generic wording; they cannot prove that a cited passage really says what the sentence claims (the self-check pass and the "cites the source page" figure help, and reading answers is still needed).
- On a laptop without the `ml` extra there is no reranker, so the search cuts that drop weak passages do not apply; the model's own "not found" reply is then the only way a question is marked not in the slides. Run the answer evaluation on Kaggle or the Space for real numbers.
- Answers from general knowledge (when neither the slides nor trusted sites have the topic) carry no source and can be wrong; they are always marked as general knowledge.
- The syllabus check compares the question with the syllabus points (similarity, then an LLM judgement), so a question worded very differently from the syllabus can be misjudged.
- DuckDuckGo search goes through the unofficial ddgs library: it needs no key but is rate-limited and can stop working when the site changes, so it is the last provider.
- RAGAS scores use an LLM judge on free tiers, so they vary between runs and use quota; they are measured on 20 answers by default.
- The relevance gate thresholds for trusted-site passages (`config/trusted_sites.yaml`) are starting values; the answer report shows how often passages are rejected.
- Units, "when to use" and worked examples in formula tables are written by the model (marked as generated); only the example's result is computed, by SymPy.
- PDF exports show formulas as LaTeX source; rendered formulas come with the website.
- The evaluation questions are drafted by an LLM; numbers are only as good as the questions, so read and fix them. Bigger chunks (per heading, windows) cover more pages, which makes finding the right page easier; the report shows words and pages per chunk next to recall.
- The embedded Qdrant index can be opened by one process at a time; stop the app before running `examprep-index` on the same machine.
- Search has been tested with stand-in models in the tests; the real bge models run on Kaggle and the Space.
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
