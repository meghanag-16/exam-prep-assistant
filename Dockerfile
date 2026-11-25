# Hugging Face Docker Space image (Linux). Installs Tesseract and LibreOffice,
# so OCR and .ppt conversion work on the hosted app.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr libreoffice-impress \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Hugging Face Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    UV_PROJECT_ENVIRONMENT=/home/user/.venv EXAMPREP_DATA_DIR=/home/user/data
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY --chown=user . .
RUN uv sync --frozen --no-dev

EXPOSE 7860
CMD ["bash", "start.sh"]
