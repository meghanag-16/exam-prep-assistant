#!/usr/bin/env bash
# Starts the FastAPI backend (internal port 8000) and the Streamlit UI (port 7860,
# the port Hugging Face Spaces expose).
set -e
uv run --no-sync uvicorn examprep.api.main:app --host 127.0.0.1 --port 8000 &
exec uv run --no-sync streamlit run src/examprep/ui/app.py \
    --server.port 7860 --server.address 0.0.0.0 --server.headless true
