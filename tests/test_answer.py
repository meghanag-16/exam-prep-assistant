"""Grounded answers: citation checks, regeneration, refusals, the answer cache, extra
explanations from trusted sites and the answer evaluation. The LLM and the web are faked."""

from __future__ import annotations

import json

import pytest
import yaml
from test_search import FakeEncoder, isolated, subject  # noqa: F401 (fixtures)

from examprep import llm
from examprep.answer import evaluate as answer_eval
from examprep.answer import grounded, terms, trusted
from examprep.search import index

GOOD = (
    "The **banker's algorithm** checks whether the system stays in a safe state before it grants a request [S1]. "
    "It uses the available, max and need matrices to decide whether every process can still finish [S1]."
)


class FakeLLM:
    """Replies in turn; records the messages it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.sent = []

    def __call__(self, messages, temperature=0.1, max_tokens=4096, **kw):
        self.sent.append(messages)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return reply(messages) if callable(reply) else reply


@pytest.fixture
def built(subject):  # noqa: F811
    index.build(subject, log=lambda m: None)
    return subject


def test_citation_label_parsing():
    text = "A [S1]. B [S2, S3] and C [W1][S1]."
    assert grounded.labels_in(text) == ["S1", "S2", "S3", "W1"]
    assert grounded.strip_labels(text) == "A. B and C."


def test_key_terms_skip_common_words():
    stats = terms.TermStats(texts=["deadlock course slide", "paging course slide", "thrashing course slide"] * 2)
    # "course" and "slide" are on every page: never key terms.
    assert stats.key_terms("Deadlocks in this course slide", 3) == ["deadlock"]
    assert terms.present(["deadlock", "paging"], "A deadlock example") == ["deadlock"]


def test_grounded_answer_with_valid_citations(built, monkeypatch):
    fake = FakeLLM({"found": True, "answer": GOOD})
    monkeypatch.setattr(llm, "chat_json", fake)
    r = grounded.answer(built, "What does the bankers algorithm check?", marks=2)
    assert r["in_slides"] and r["checks"]["citations_valid"] and not r["checks"]["problems"]
    assert r["attempts"] == 1 and "[1]" in r["answer"] and "[S1]" not in r["answer"]
    assert r["sources"][0]["n"] == 1 and "os_slides.pdf" in r["sources"][0]["where"]
    # Only the retrieved passages were sent, inside <material>, with their labels.
    user = fake.sent[0][1]["content"]
    assert "<material>" in user and "[S1]" in user
    # The second time the answer comes from the cache: no model call.
    again = grounded.answer(built, "what does the  Bankers algorithm check?", marks=2)
    assert again["cached"] and len(fake.sent) == 1 and again["answer"] == r["answer"]


def test_bad_citation_and_uncited_line_are_sent_back(built, monkeypatch):
    bad = {
        "found": True,
        "answer": GOOD.replace("[S1].", "[S9].", 1) + "\n- Banker's algorithm is used by every OS today.",
    }
    fake = FakeLLM(bad, {"found": True, "answer": GOOD})
    monkeypatch.setattr(llm, "chat_json", fake)
    r = grounded.answer(built, "Explain the bankers algorithm", marks=2, use_cache=False)
    assert r["attempts"] == 2 and r["checks"]["citations_valid"]
    feedback = fake.sent[1][-1]["content"]
    assert "S9" in feedback and "without a citation" in feedback


def test_generic_answer_fails_the_term_check(built, monkeypatch):
    generic = {
        "found": True,
        "answer": "It is an important method that is used widely in many real systems and it helps a lot with "
        "many kinds of problems that people face every day [S1].",
    }
    monkeypatch.setattr(llm, "chat_json", FakeLLM(generic))
    r = grounded.answer(built, "Explain the bankers algorithm", marks=2, use_cache=False)
    assert any("too generic" in p for p in r["checks"]["problems"])
    assert r["attempts"] == 3  # first try plus two regenerations, then shown with the problems recorded


def test_not_in_slides_answers_from_trusted_sites(built, monkeypatch):
    page = "The 1998 football world cup was won by France, who beat Brazil 3-0 in the final in Paris. " * 4
    monkeypatch.setattr(
        trusted,
        "search",
        lambda q, d: [{"url": "https://www.geeksforgeeks.org/wc", "title": "World cup", "content": page}],
    )
    monkeypatch.setattr(trusted, "fetch_text", lambda url, d: page)

    def reply(messages):
        if "do not cover this question" in messages[0]["content"]:
            return {"found": True, "answer": "France won the 1998 world cup, beating Brazil 3-0 in the final [W1]."}
        return {"found": False}

    monkeypatch.setattr(llm, "chat_json", FakeLLM(reply))
    r = grounded.answer(built, "Who won the 1998 football world cup?", use_cache=False)
    assert not r["in_slides"] and r["notice"].startswith(grounded.NOT_IN_SLIDES)
    assert r["outside_source"] == "trusted" and "[1]" in r["answer"] and r["sources"][0]["site"] == "geeksforgeeks.org"
    md = grounded.to_markdown(r)
    assert "not in your slides" in md and "Trusted sources" in md


def test_not_in_slides_falls_back_to_general_knowledge(built, monkeypatch):
    monkeypatch.setattr(trusted, "search", lambda q, d: [])

    def reply(messages):
        if "general knowledge" in messages[0]["content"]:
            return {"answer": "France won the 1998 world cup."}
        return {"found": False}

    monkeypatch.setattr(llm, "chat_json", FakeLLM(reply))
    r = grounded.answer(built, "Who won the 1998 football world cup?", use_cache=False)
    assert not r["in_slides"] and r["outside_source"] == "general" and "general knowledge" in r["notice"]
    # With outside answers turned off, only the notice is given.
    from examprep import config

    cfg = config.answer()
    monkeypatch.setitem(cfg["answer"], "outside_answer", "none")
    r = grounded.answer(built, "Who won the 1998 football world cup?", use_cache=False)
    assert not r["in_slides"] and r["outside_source"] is None and r["answer"] == grounded.REFUSAL
    # Nothing passes the search cuts: the slides are not asked about at all.
    calls = FakeLLM({"found": True, "answer": GOOD})
    monkeypatch.setattr(llm, "chat_json", calls)
    monkeypatch.setattr(grounded.searching.Searcher, "search", lambda self, q, **kw: ([], {"mode": "hybrid"}))
    r = grounded.answer(built, "Explain the bankers algorithm", use_cache=False)
    assert not r["in_slides"] and not calls.sent


def test_not_in_syllabus_notice(built, monkeypatch):
    from examprep.ingest import ledger

    con = ledger.connect(built)
    try:
        con.execute("INSERT INTO syllabus_points (unit, text) VALUES ('Unit 1', 'Process scheduling algorithms')")
        con.execute("INSERT INTO syllabus_points (unit, text) VALUES ('Unit 2', 'Paging and virtual memory')")
        con.commit()
    finally:
        con.close()

    def reply(messages):
        if "syllabus covers" in messages[0]["content"]:
            return {"covered": False, "point": None}
        return {"found": True, "answer": GOOD}

    monkeypatch.setattr(llm, "chat_json", FakeLLM(reply))
    r = grounded.answer(built, "What does the bankers algorithm check?", marks=2, use_cache=False)
    assert r["in_slides"] and r["in_syllabus"] is False and r["notice"] == grounded.NOT_IN_SYLLABUS
    assert grounded.NOT_IN_SYLLABUS in grounded.to_markdown(r)


def test_mark_levels():
    assert [grounded.marks_spec(m)["marks"] for m in (2, 4, 5, 6, 7, 8, 10, 15)] == [2, 4, 5, 6, 7, 8, 10, 15]
    assert grounded.marks_spec(16)["marks"] == 15 and grounded.marks_spec(3)["marks"] == 2
    assert "Pipelining" in grounded._example(4)  # no 4-mark example: the 5-mark one is used


def test_exa_is_used_when_tavily_fails(monkeypatch):
    calls = []

    def tavily(q, d, cfg):
        calls.append("tavily")
        raise trusted.TrustedError("TAVILY_API_KEY is not set")

    def exa(q, d, cfg):
        calls.append("exa")
        return [
            {"url": "https://www.programiz.com/x", "title": "x", "content": "c"},
            {"url": "https://spam.example/x", "title": "y", "content": "c"},
        ]

    monkeypatch.setitem(trusted.PROVIDERS, "tavily", tavily)
    monkeypatch.setitem(trusted.PROVIDERS, "exa", exa)
    found = trusted.search("what is recursion", ["programiz.com"])
    assert calls == ["tavily", "exa"] and [x["url"] for x in found] == ["https://www.programiz.com/x"]
    assert found[0]["provider"] == "exa"


def test_exa_request_and_reply(monkeypatch):
    sent = {}

    class Reply:
        status_code = 200

        def json(self):
            return {"results": [{"url": "https://www.programiz.com/r", "title": "R", "highlights": ["a", "b"]}]}

    def post(url, json, headers, timeout):
        sent.update(url=url, body=json, headers=headers)
        return Reply()

    monkeypatch.setattr(trusted.config, "secret", lambda name: "k" if name == "EXA_API_KEY" else None)
    monkeypatch.setattr(trusted.httpx, "post", post)
    out = trusted._exa("recursion", ["programiz.com"], {"max_results": 3})
    assert sent["url"] == trusted.EXA_URL and sent["headers"] == {"x-api-key": "k"}
    assert sent["body"]["includeDomains"] == ["programiz.com"] and sent["body"]["numResults"] == 3
    assert out == [{"url": "https://www.programiz.com/r", "title": "R", "content": "a b"}]


def test_duckduckgo_needs_no_key(monkeypatch):
    import sys
    import types

    asked = {}

    class FakeDDGS:
        def __init__(self, timeout):
            pass

        def text(self, query, max_results, backend):
            asked.update(query=query, backend=backend)
            return [{"href": "https://ocw.mit.edu/a", "title": "A", "body": "about recursion"}]

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=FakeDDGS))
    out = trusted._duckduckgo("recursion", ["ocw.mit.edu", "programiz.com"], {})
    assert asked["query"] == "recursion (site:ocw.mit.edu OR site:programiz.com)" and asked["backend"] == "duckduckgo"
    assert out == [{"url": "https://ocw.mit.edu/a", "title": "A", "content": "about recursion"}]


def test_allowlist():
    domains = ["geeksforgeeks.org", "ocw.mit.edu"]
    assert trusted.allowed("https://www.geeksforgeeks.org/x", domains)
    assert trusted.allowed("https://ocw.mit.edu/courses/y", domains)
    assert not trusted.allowed("https://geeksforgeeks.org.evil.com/x", domains)
    assert not trusted.allowed("https://mit.edu/x", domains)
    assert not trusted.allowed("ftp://geeksforgeeks.org/x", domains)
    assert not trusted.allowed("https://evilgeeksforgeeks.org/x", domains)


def test_extra_explanations_keep_slides_and_sites_apart(built, monkeypatch):
    page = (
        "The banker's algorithm avoids deadlock by checking for a safe state. Before a request is granted the "
        "algorithm pretends to allocate the resources and checks whether every process can still finish using "
        "the available, max and need matrices. If no safe sequence exists the request waits. "
    ) * 2
    monkeypatch.setattr(
        trusted,
        "search",
        lambda q, d: [
            {"url": "https://www.geeksforgeeks.org/bankers", "title": "Banker's Algorithm", "content": ""},
            {"url": "https://example.com/bankers", "title": "not allowed", "content": page},
        ],
    )
    monkeypatch.setattr(trusted, "fetch_text", lambda url, d: page if trusted.allowed(url, d) else None)
    reply = {
        "found": True,
        "answer": GOOD,
        "layers": {
            "intuition": [{"text": "Like a careful bank that never lends out all its cash.", "cite": ["W1"]}],
            "example": [{"text": "Two processes asking for the last instance must wait.", "cite": ["S1"]}],
            "common_mistakes": [{"text": "Confusing avoidance with prevention.", "cite": []}],
        },
        "related_topics": ["Deadlock Conditions", "Not a topic"],
    }
    monkeypatch.setattr(llm, "chat_json", FakeLLM(reply))
    r = grounded.answer(built, "Explain the bankers algorithm", marks=2, deeper=True, use_cache=False)
    assert r["web"] and all("geeksforgeeks" in w["url"] for w in r["web"])
    layers = r["layers"]
    assert layers["intuition"][0]["beyond"] and layers["intuition"][0]["links"][0]["site"] == "geeksforgeeks.org"
    assert not layers["example"][0]["beyond"]
    assert "common_mistakes" not in layers  # no valid citation: left out
    assert r["checks"]["beyond_points"] == 2 and r["checks"]["beyond_with_valid_link"] == 1
    assert r["checks"].get("self_checked")
    md = grounded.to_markdown(r)
    assert "### Beyond your slides" in md and "geeksforgeeks.org" in md and "### More from your slides" in md
    assert all(t != "Not a topic" for t in r["related_topics"])


def test_gate_rejects_unrelated_passages(monkeypatch):
    slides = ["Paging divides memory into fixed size frames and pages mapped by a page table."]
    cands = [
        trusted.WebPassage("u1", "t", "s", "Paging maps pages to frames using a page table in memory."),
        trusted.WebPassage("u2", "t", "s", "The French revolution began in 1789 with the storming of the Bastille."),
    ]
    kept, rejected = trusted.gate("what is paging", slides, cands, reranker=lambda q, ts: [0.9, 0.8])
    assert [k.url for k in kept] == ["u1"] and [r.url for r in rejected] == ["u2"]
    kept, rejected = trusted.gate("what is paging", slides, cands[:1], reranker=lambda q, ts: [0.1])
    assert not kept  # about the slides but not about the question


def test_feedback_is_logged(built, monkeypatch):
    monkeypatch.setattr(llm, "chat_json", FakeLLM({"found": True, "answer": GOOD}))
    r = grounded.answer(built, "What does the bankers algorithm check?", marks=2)
    fid = grounded.feedback(built, r["id"], -1, "missing the safety algorithm steps")
    assert fid > 0
    with pytest.raises(ValueError):
        grounded.feedback(built, r["id"], 5)


def test_answer_evaluation_report(built, monkeypatch):
    from examprep.ingest import ledger

    qs = {
        "questions": [
            {
                "id": "q1",
                "question": "What does the bankers algorithm check?",
                "pages": [{"file": "os_slides.pdf", "page": 6}],
            }
        ],
        "outside": [{"id": "o1", "question": "Who painted the Mona Lisa?"}],
    }
    d = ledger.subject_dir(built) / "eval"
    d.mkdir(parents=True, exist_ok=True)
    (d / "questions.yaml").write_text(yaml.safe_dump(qs), encoding="utf-8")

    def reply(messages):
        return {"found": "Mona" not in messages[1]["content"], "answer": GOOD}

    monkeypatch.setattr(llm, "chat_json", FakeLLM(reply))
    from examprep.answer import ragas_eval

    sent = []
    monkeypatch.setattr(ragas_eval, "available", lambda: True)
    monkeypatch.setattr(
        ragas_eval,
        "score",
        lambda items, log: (
            sent.extend(items)
            or {
                "judge": "groq/m",
                "rows": [{"faithfulness": 1.0, "response_groundedness": 0.5, "context_relevance": 1.0}],
                "errors": [],
            }
        ),
    )
    rep = answer_eval.run(built, deeper=0, log=lambda m: None)
    assert rep["ragas"]["status"] == "ok" and rep["ragas"]["faithfulness"] == 1.0
    assert sent and "[S1]" not in sent[0]["answer"] and sent[0]["contexts"]
    assert rep["answered"] == 1.0 and rep["valid_citations"] == 1.0 and rep["cites_source_page"] == 1.0
    assert rep["correct_refusals"] == 1.0
    saved = json.loads((d / "answer_report.json").read_text(encoding="utf-8"))
    assert saved["questions"] == 1 and (d / "answer_report.md").exists()
    # A repeat run with the same inputs can be skipped; other counts or edited questions cannot.
    assert answer_eval.is_current(built, deeper=0)
    assert not answer_eval.is_current(built, deeper=1)
    qs["questions"][0]["question"] = "What does the bankers algorithm decide?"
    (d / "questions.yaml").write_text(yaml.safe_dump(qs), encoding="utf-8")
    assert not answer_eval.is_current(built, deeper=0)


def test_api_ask(built, monkeypatch):
    from fastapi.testclient import TestClient

    from examprep.api.main import app

    monkeypatch.setattr(llm, "chat_json", FakeLLM({"found": True, "answer": GOOD}))
    client = TestClient(app)
    r = client.post(f"/ask/{built}", json={"question": "What does the bankers algorithm check?", "marks": 2})
    assert r.status_code == 200 and r.json()["in_slides"] and r.json()["sources"]
    assert client.post(f"/ask/{built}", json={"question": ""}).status_code == 422
    assert client.post("/ask/nosuchsubject", json={"question": "x"}).status_code == 404


def _content_setting(monkeypatch, on: bool):
    from examprep import config

    models = dict(config.models())
    models["tracing"] = {"log_content": on}
    monkeypatch.setattr(config, "models", lambda: models)


class _FakeLangfuse:
    """Records what the app sends to Langfuse."""

    def __init__(self):
        self.started, self.updates, self.scores = [], [], []

    def start_as_current_observation(self, **kw):
        fake = self

        class Obs:
            trace_id = "t-1"

            def update(self, **u):
                fake.updates.append((kw["name"], u))

        class CM:
            def __enter__(self):
                fake.started.append(kw)
                return Obs()

            def __exit__(self, *a):
                return False

        return CM()

    def score_current_trace(self, **kw):
        self.scores.append(kw)

    def create_score(self, **kw):
        self.scores.append(kw)

    def get_trace_url(self, trace_id):
        return f"https://cloud.langfuse.com/trace/{trace_id}"


def test_tracing_is_off_without_keys_and_can_hide_content(monkeypatch):
    from examprep import tracing

    monkeypatch.setattr(tracing, "_tried", False)
    monkeypatch.setattr(tracing, "_client", None)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    with tracing.observe("x", input="slide text") as obs:
        assert obs is None
    with tracing.trace("answer-question", input="q", tags=["ask"]) as root:
        assert root is None
    tracing.score("in-slides", 1)  # no client: nothing happens
    assert tracing.content("slide text") == "slide text"  # content is shown by default
    _content_setting(monkeypatch, False)
    assert tracing.content("secret slide text") == {"chars": 17}

    fake = _FakeLangfuse()
    monkeypatch.setattr(tracing, "_tried", True)
    monkeypatch.setattr(tracing, "_client", fake)
    with tracing.observe("call-llm", as_type="generation", input=[{"content": "slide"}], model="m") as obs:
        tracing.update(obs, output="answer text", usage_details={"input": 3, "output": 4}, level=None)
    assert fake.started[0]["input"] == {"chars": len(repr([{"content": "slide"}]))}
    name, sent = fake.updates[0]
    assert sent["output"] == {"chars": 11} and sent["usage_details"]["output"] == 4 and "level" not in sent


def test_tracing_errors_in_the_app_are_recorded_and_raised(monkeypatch):
    from examprep import tracing

    fake = _FakeLangfuse()
    monkeypatch.setattr(tracing, "_tried", True)
    monkeypatch.setattr(tracing, "_client", fake)
    with pytest.raises(ValueError):
        with tracing.observe("search-material", as_type="retriever"):
            raise ValueError("boom")
    assert fake.updates[0][1]["level"] == "ERROR"


def test_redact_removes_personal_data_and_keys():
    from examprep import tracing

    text = "mail me at a.b@college.edu or +91 98765 43210, key tvly-abcdefghijklmnop, page 12, year 2024"
    out = tracing.redact(text)
    assert "college.edu" not in out and "98765" not in out and "abcdefghijklmnop" not in out
    assert "[email]" in out and "[phone]" in out and "[key]" in out
    assert "page 12" in out and "2024" in out  # ordinary numbers stay


def test_answer_trace_names_attributes_and_scores(built, monkeypatch):
    from examprep import tracing

    fake = _FakeLangfuse()
    monkeypatch.setattr(tracing, "_tried", True)
    monkeypatch.setattr(tracing, "_client", fake)
    attrs = {}

    class Propagate:
        def __init__(self, **kw):
            attrs.update(kw)

        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    import langfuse

    monkeypatch.setattr(langfuse, "propagate_attributes", Propagate)
    llm_names = []
    reply = FakeLLM({"found": True, "answer": GOOD})

    def fake_llm(messages, temperature=0.1, max_tokens=4096, trace_name="call-llm", **kw):
        llm_names.append(trace_name)
        return reply(messages)

    monkeypatch.setattr(llm, "chat_json", fake_llm)
    r = grounded.answer(built, "What does the bankers algorithm check?", 2, use_cache=False, session_id="chat-1")
    names = [x["name"] for x in fake.started]
    assert names[0] == "answer-question" and "search-material" in names and "check-answer" in names
    assert llm_names == ["write-answer"]
    assert attrs["session_id"] == "chat-1" and attrs["tags"] == ["ask"] and attrs["metadata"]["marks"] == "2"
    assert r["trace_id"] == "t-1"
    assert {s["name"] for s in fake.scores} >= {"in-slides", "checks-passed", "attempts"}
    grounded.feedback(built, r["id"], -1, "too short")
    fb = fake.scores[-1]
    assert fb["name"] == "user-feedback" and fb["value"] == -1.0 and fb["trace_id"] == "t-1"


def test_ragas_waits_once_then_switches_judge(monkeypatch):
    import sys
    import types

    from examprep.answer import ragas_eval

    class RateLimitError(Exception):
        pass

    class BadRequestError(Exception):
        pass

    calls = []

    def metric(name):
        class Metric:
            def __init__(self, llm):
                self.judge = llm

            async def ascore(self, **kw):
                calls.append((self.judge, name))
                if self.judge == "groq/a":
                    raise RateLimitError("429 too many requests")
                if self.judge == "groq/b":
                    raise BadRequestError("400 bad request")
                return types.SimpleNamespace(value=0.8)

        return Metric

    fake = types.ModuleType("ragas.metrics.collections")
    fake.Faithfulness = metric("faithfulness")
    fake.ResponseGroundedness = metric("response_groundedness")
    fake.ContextRelevance = metric("context_relevance")
    monkeypatch.setitem(sys.modules, "ragas", types.ModuleType("ragas"))
    monkeypatch.setitem(sys.modules, "ragas.metrics", types.ModuleType("ragas.metrics"))
    monkeypatch.setitem(sys.modules, "ragas.metrics.collections", fake)
    chain = [("groq/a", {}), ("groq/b", {}), ("nvidia/c", {})]
    monkeypatch.setattr(ragas_eval, "_judges", lambda: chain)
    monkeypatch.setattr(ragas_eval, "_make_judge", lambda model_id, p: model_id)
    cfg = dict(ragas_eval.config.answer())
    cfg["evaluation"] = {**cfg["evaluation"], "ragas_rate_limit_wait": 0}
    monkeypatch.setattr(ragas_eval.config, "answer", lambda: cfg)
    item = {"question": "q", "answer": "a", "contexts": ["c"]}
    res = ragas_eval.score([item, item], log=lambda m: None)
    assert res["rows"] == [{"faithfulness": 0.8, "response_groundedness": 0.8, "context_relevance": 0.8}] * 2
    # groq/a: rate limited twice (waited once), groq/b: refused, then nvidia/c for everything after.
    assert [j for j, _ in calls[:3]] == ["groq/a", "groq/a", "groq/b"] and calls[3][0] == "nvidia/c"
    assert res["judge"] == "groq/a, groq/b, nvidia/c" and any("rate" in e for e in res["errors"])
