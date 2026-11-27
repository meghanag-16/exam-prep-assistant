import pytest


@pytest.fixture(autouse=True)
def no_llm_cache(monkeypatch):
    """Tests fake the LLM; a reply saved by one test must not answer another."""
    monkeypatch.setenv("EXAMPREP_LLM_CACHE", "0")


@pytest.fixture(autouse=True)
def no_live_web_search(monkeypatch):
    """The keyless DuckDuckGo provider would reach the internet: tests never do."""
    from examprep.answer import trusted

    def offline(query, domains, cfg):
        raise trusted.TrustedError("live search is not set up in tests")

    monkeypatch.setitem(trusted.PROVIDERS, "duckduckgo", offline)
