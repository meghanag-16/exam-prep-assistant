from pathlib import Path

from examprep import config
from examprep.ingest.convert import PptConversionError, ppt_to_pptx


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


def test_llm_replies_are_cached(tmp_path, monkeypatch):
    from examprep import llm

    monkeypatch.setenv("EXAMPREP_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("EXAMPREP_LLM_CACHE", "1")
    monkeypatch.setattr(llm, "available_providers", lambda: ["groq"])
    monkeypatch.setattr(llm.config, "secret", lambda name: "test-key")
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    calls = []

    class Resp:
        status_code, headers = 200, {}

        def json(self):
            return {"choices": [{"message": {"content": '{"ok": 1}'}, "finish_reason": "stop"}]}

    monkeypatch.setattr(llm.httpx, "post", lambda *a, **k: calls.append(1) or Resp())
    msg = [{"role": "user", "content": "same question"}]
    assert llm.chat_json(msg) == {"ok": 1}
    assert llm.chat_json(msg) == {"ok": 1}
    assert len(calls) == 1  # the second answer came from the cache
    llm.chat_json([{"role": "user", "content": "another question"}])
    assert len(calls) == 2
    with llm.fresh():  # the answer evaluation measures real calls
        assert llm.chat_json(msg) == {"ok": 1}
    assert len(calls) == 3
    llm.chat_json(msg)
    assert len(calls) == 3  # cached again outside the block


def test_requests_are_spaced_per_model(monkeypatch):
    from examprep import llm

    clock = [100.0]
    waits = []
    monkeypatch.setattr(llm.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(llm.time, "sleep", lambda s: waits.append(round(s, 1)) or clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(llm, "_last_call", {})
    llm._pace("m", 5)
    llm._pace("m", 5)  # 5 a minute: 12 seconds apart
    llm._pace("other", 5)  # another model has its own limit
    assert waits == [12.0]
