import pytest


@pytest.fixture(autouse=True)
def no_llm_cache(monkeypatch):
    """Tests fake the LLM; a reply saved by one test must not answer another."""
    monkeypatch.setenv("EXAMPREP_LLM_CACHE", "0")
