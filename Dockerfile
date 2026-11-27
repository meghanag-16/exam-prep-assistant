# Hugging Face Docker Space image (Linux). Installs Tesseract and LibreOffice,
# so OCR and .ppt conversion work on the hosted app, Pango for WeasyPrint PDFs, and
# the bge embedding and reranker models (CPU-only torch) for hybrid search with reranking.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr libreoffice-impress libpango-1.0-0 libpangoft2-1.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Hugging Face Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    UV_PROJECT_ENVIRONMENT=/home/user/.venv EXAMPREP_DATA_DIR=/home/user/data
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra pdf --no-install-project
COPY --chown=user . .
RUN uv sync --frozen --no-dev --extra pdf
# CPU-only torch first (the PyPI build brings several GB of CUDA libraries a CPU Space
# cannot use), then sentence-transformers on top of it. start.sh runs with --no-sync,
# so these stay installed.
RUN uv pip install --python /home/user/.venv/bin/python \
        --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" \
    && uv pip install --python /home/user/.venv/bin/python "sentence-transformers>=3.0"

EXPOSE 7860
CMD ["bash", "start.sh"]
