from pathlib import Path

from examprep import config
from examprep.ingest.convert import PptConversionError, available_converter, ppt_to_pptx


def test_yaml_configs_load():
    assert config.models()["llm"]["order"]
    for name in config.models()["llm"]["order"]:
        assert name in config.models()["llm"]["providers"]
    assert 2 in config.exam_pattern()["marks"]
    assert ".pdf" in config.ingest()["accepted_extensions"]


def test_allowed_domains_has_no_wikipedia():
    domains = config.allowed_domains()
    assert "geeksforgeeks.org" in domains
    assert not any("wikipedia" in d for d in domains)


def test_secret_blank_is_none(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "   ")
    assert config.secret("GROQ_API_KEY") is None


def test_ppt_without_converter_gives_clear_message(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("examprep.ingest.convert.available_converter", lambda: None)
    fake = tmp_path / "1.ppt"
    fake.write_bytes(b"not really a ppt")
    try:
        ppt_to_pptx(fake, tmp_path / "out")
    except PptConversionError as e:
        assert "Save As" in str(e)
    else:
        raise AssertionError("expected PptConversionError")


def test_health_endpoint():
    from fastapi.testclient import TestClient

    from examprep.api.main import app

    assert TestClient(app).get("/health").json()["status"] == "ok"
