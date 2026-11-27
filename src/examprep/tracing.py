"""Langfuse tracing (Python SDK v5), following Langfuse's instrumentation best practices.

- One trace per unit of work: answering one question (`answer-question`), making the notes
  of one topic (`make-topic-notes`), one answer evaluation run, one standalone search.
- Steps are nested observations with their most specific type: `retriever` for searches,
  `generation` for every LLM call (model, parameters, token usage, cache hits), `evaluator`
  for checks done by a model, `guardrail` for the code checks, `chain` for the rest.
- Names are verb-first and fixed (no ids or counters in names), so dashboards and filters
  can rely on them.
- Trace attributes (session id, tags, metadata such as subject and marks, version) are set
  with propagate_attributes so every observation in the trace carries them; the environment
  comes from LANGFUSE_TRACING_ENVIRONMENT (laptop, kaggle, space ...).
- Scores: whether an answer passed its checks, whether it came from the slides, user
  thumbs up or down (POST /feedback) and the RAGAS figures from the answer evaluation are
  attached to the answer's trace.
- Masking: e-mail addresses, phone numbers and API keys are removed from everything sent
  (export-stage masking). With tracing.log_content false in config/models.yaml, inputs and
  outputs are replaced by their sizes, so no course text leaves the machine.

Tracing is on only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set (LANGFUSE_HOST or
LANGFUSE_BASE_URL for a server other than Langfuse Cloud EU). Without them, or if Langfuse
fails, every function here does nothing: tracing never stops the app.
"""

from __future__ import annotations

import atexit
import re
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any

from examprep import __version__, config

_client: Any = None
_tried = False

# Personal data and secrets that must never reach a trace, whatever the content setting.
_SECRETS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"(?<![\w.])\+?\d(?:[\s-]?\d){9,13}(?![\w.])"), "[phone]"),
    (re.compile(r"\b(?:sk|pk|tvly|gsk|nvapi|hf|AIza|exa)[-_][A-Za-z0-9_-]{12,}"), "[key]"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._-]{12,}"), r"\1[key]"),
]


def redact(text: str) -> str:
    for pattern, repl in _SECRETS:
        text = pattern.sub(repl, text)
    return text


def _mask_otel_spans(*, params: Any) -> Any:
    """Export-stage masking: redact every string attribute of every span in the batch."""
    from langfuse.types import MaskOtelSpansResult, OtelSpanPatch

    patches = {}
    for ident, span in params.spans.items():
        changed: dict[str, Any] = {}
        for key, value in (span.attributes or {}).items():
            if isinstance(value, str):
                masked = redact(value)
                if masked != value:
                    changed[key] = masked
            elif isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
                masked_list = [redact(v) for v in value]
                if masked_list != list(value):
                    changed[key] = masked_list
        if changed:
            patches[ident] = OtelSpanPatch(set_attributes=changed)
    return MaskOtelSpansResult(span_patches=patches)


def _settings() -> dict:
    return config.models().get("tracing") or {}


def client():
    """The Langfuse client, or None when tracing is off or not installed."""
    global _client, _tried
    if _tried:
        return _client
    _tried = True
    if not (config.secret("LANGFUSE_PUBLIC_KEY") and config.secret("LANGFUSE_SECRET_KEY")):
        return None
    try:
        from langfuse import Langfuse

        rate = _settings().get("sample_rate")
        _client = Langfuse(
            release=__version__,
            sample_rate=float(rate) if rate is not None else None,
            mask_otel_spans=_mask_otel_spans,
        )
    except Exception:  # noqa: BLE001 (tracing must never stop the app)
        _client = None
        return None
    atexit.register(flush)
    return _client


def flush() -> None:
    if _client is not None:
        try:
            _client.flush()
        except Exception:  # noqa: BLE001
            pass


def _log_content() -> bool:
    return bool(_settings().get("log_content", True))


def content(value: Any) -> Any:
    """The value itself when content logging is on, else only its size."""
    if value is None or _log_content():
        return value
    text = value if isinstance(value, str) else repr(value)
    return {"chars": len(text)}


@contextmanager
def _guard(cm: Any) -> Iterator[Any]:
    """Enter a Langfuse context manager without letting a Langfuse error reach the app;
    errors raised by the app's own code inside the block are recorded and re-raised."""
    try:
        obj = cm.__enter__()
    except Exception:  # noqa: BLE001
        yield None
        return
    try:
        yield obj
    except BaseException as e:
        try:
            if obj is not None and hasattr(obj, "update"):
                obj.update(level="ERROR", status_message=f"{e.__class__.__name__}: {str(e)[:200]}")
        except Exception:  # noqa: BLE001
            pass
        try:
            cm.__exit__(type(e), e, e.__traceback__)
        except Exception:  # noqa: BLE001
            pass
        raise
    try:
        cm.__exit__(None, None, None)
    except Exception:  # noqa: BLE001
        pass


@contextmanager
def observe(name: str, as_type: str = "span", input: Any = None, **kw) -> Iterator[Any]:  # noqa: A002
    """A Langfuse observation around the block (None when tracing is off). Nested calls
    become child observations; outside any trace this starts a new trace."""
    c = client()
    if c is None:
        yield None
        return
    try:
        cm = c.start_as_current_observation(name=name, as_type=as_type, input=content(input), **kw)
    except Exception:  # noqa: BLE001
        yield None
        return
    with _guard(cm) as obs:
        yield obs


@contextmanager
def trace(
    name: str,
    input: Any = None,  # noqa: A002
    *,
    as_type: str = "chain",
    session_id: str | None = None,
    user_id: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    version: str | None = None,
) -> Iterator[Any]:
    """The root observation of a unit of work, with trace attributes propagated to every
    observation inside it. Inside an existing trace (e.g. an answer made during an
    evaluation run) it is a nested observation that adds its attributes."""
    with ExitStack() as stack:
        obs = stack.enter_context(observe(name, as_type=as_type, input=input))
        if obs is not None:
            try:
                from langfuse import propagate_attributes

                meta = {k: str(v)[:200] for k, v in (metadata or {}).items() if v is not None}
                attrs: dict[str, Any] = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "tags": tags,
                    "version": version,
                    "metadata": meta,
                }
                stack.enter_context(_guard(propagate_attributes(**{k: v for k, v in attrs.items() if v})))
            except Exception:  # noqa: BLE001
                pass
        yield obs


def update(obs: Any, output: Any = None, **kw) -> None:
    if obs is None:
        return
    try:
        kw = {k: v for k, v in kw.items() if v is not None}
        if output is not None:
            kw["output"] = content(output)
        obs.update(**kw)
    except Exception:  # noqa: BLE001
        pass


def trace_id(obs: Any) -> str | None:
    """The trace id of an observation (stored with answers so later scores can find it)."""
    try:
        tid = getattr(obs, "trace_id", None) if obs is not None else None
    except Exception:  # noqa: BLE001
        return None
    return tid if isinstance(tid, str) else None


def trace_url(tid: str | None) -> str | None:
    c = client()
    if c is None or not tid:
        return None
    try:
        return c.get_trace_url(trace_id=tid)
    except Exception:  # noqa: BLE001
        return None


def score(
    name: str,
    value: float | str,
    *,
    trace_id: str | None = None,
    data_type: str | None = None,
    comment: str | None = None,
) -> None:
    """A score on a trace (by id) or, without an id, on the current trace."""
    c = client()
    if c is None:
        return
    try:
        if trace_id:
            c.create_score(name=name, value=value, trace_id=trace_id, data_type=data_type, comment=comment)
        else:
            c.score_current_trace(name=name, value=value, data_type=data_type, comment=comment)
    except Exception:  # noqa: BLE001
        pass
