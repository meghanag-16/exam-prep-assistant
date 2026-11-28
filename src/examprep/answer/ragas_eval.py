"""RAGAS scores for grounded answers (examprep-answer-eval).

Three RAGAS metrics that need no reference answer, so they work on any subject's
evaluation questions:
- faithfulness: share of the answer's claims supported by the passages given to the model.
- response groundedness: how well the answer as a whole is grounded in those passages.
- context relevance: how relevant the retrieved passages are to the question.

RAGAS calls its judge model itself (through an OpenAI-compatible client): evaluation.ragas_provider
in config/answer.yaml when set, else the providers of config/models.yaml in order. A rate limit
is waited out (for as long as the provider asks, "try again in 7.5s", up to
evaluation.ragas_rate_limit_retries times); a daily limit, a refusal, or a judge still limited
after that is replaced by the next judge. Instructor's own retry messages are not printed: the
waits and judge changes are logged instead.
A judgement that fails for another reason (for example a reply that is not valid JSON) is
tried once more with the next judge, for that answer only. Judge replies may be up to
evaluation.ragas_max_tokens tokens: RAGAS's own default (1024) cuts off faithfulness, which
lists every claim of the answer with a verdict, and reasoning models spend part of the
budget on thinking.
RAGAS is in the `eval` extra (installed on Kaggle); without it these
scores are skipped and the report says so. RAGAS's anonymous usage tracking is turned off.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
from collections.abc import Callable

from examprep import config, llm

Log = Callable[[str], None]


def available() -> bool:
    try:
        import ragas  # noqa: F401
    except ImportError:
        return False
    return True


def _judges() -> list[tuple[str, dict]]:
    """("provider/model", provider settings) for every judge model that can be tried, in order:
    evaluation.ragas_provider first (when set), then the providers of config/models.yaml that
    have a key, two models each. When one judge is refused or rate-limited, the next is used."""
    cfg = config.models()["llm"]
    wanted = config.answer()["evaluation"].get("ragas_provider")
    order = ([wanted] if wanted else []) + [n for n in llm.available_providers() if n != wanted]
    out = []
    for name in order:
        p = cfg["providers"].get(name or "")
        if p and config.secret(p["api_key_env"]):
            out += [(f"{name}/{m}", p) for m in llm.provider_models(p)[:2]]
    if not out:
        raise RuntimeError("no LLM provider key is set for the RAGAS judge")
    return out


def _make_judge(model_id: str, p: dict):
    import instructor
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    timeout = float(config.models()["llm"].get("timeout_seconds", 60))
    client = AsyncOpenAI(base_url=p["base_url"], api_key=config.secret(p["api_key_env"]), timeout=timeout)
    # MD_JSON asks for JSON in the reply text instead of the response_format parameter, which
    # not every free provider and model accepts.
    max_tokens = int(config.answer()["evaluation"].get("ragas_max_tokens", 4096))
    return llm_factory(
        model_id.split("/", 1)[1],
        provider="openai",
        client=client,
        mode=instructor.Mode.MD_JSON,
        temperature=0.0,
        max_tokens=max_tokens,
    )


def _kind(e: BaseException) -> str:
    """ "rate" for rate limits and quota, "refused" for requests the model rejects, else "other"."""
    seen: set[int] = set()
    while e is not None and id(e) not in seen:  # instructor wraps the provider's error
        seen.add(id(e))
        text = f"{e.__class__.__name__} {e}"
        if "RateLimit" in text or "429" in text or "quota" in text.lower():
            return "rate"
        if "BadRequest" in text or "400" in text or "NotFound" in text or "404" in text:
            return "refused"
        e = e.__cause__ or e.__context__  # type: ignore[assignment]
    return "other"


def _limit_text(e: BaseException) -> str:
    seen: set[int] = set()
    parts = []
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        parts.append(f"{e.__class__.__name__} {e}")
        resp = getattr(e, "response", None)
        after = getattr(getattr(resp, "headers", None), "get", lambda _k: None)("retry-after")
        if after:
            parts.append(f"retry-after {after}")
        e = e.__cause__ or e.__context__  # type: ignore[assignment]
    return " ".join(parts)


def retry_after(e: BaseException) -> float | None:
    """Seconds the provider asks to wait ("try again in 1m2.5s", "in 250ms", Retry-After)."""
    text = _limit_text(e)
    m = re.search(r"try again in (?:(\d+)m(?!s))?\s*([\d.]+)\s*(ms|s)\b", text, re.IGNORECASE)
    if m:
        secs = float(m.group(2)) / (1000 if m.group(3).lower() == "ms" else 1)
        return secs + 60 * int(m.group(1) or 0)
    m = re.search(r"retry-after ([\d.]+)", text)
    return float(m.group(1)) if m else None


def daily_limit(e: BaseException) -> bool:
    """True for a limit that will not clear within minutes (tokens or requests per day)."""
    text = _limit_text(e).lower()
    return any(k in text for k in ("per day", "tokens per day", "requests per day", "(tpd)", "(rpd)", "daily"))


def _clean(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else round(v, 3)


def score(items: list[dict], log: Log = print) -> dict:
    """items: [{"question", "answer", "contexts": [passage text, ...]}].
    Returns {"judge", "rows": [{"faithfulness", "response_groundedness", "context_relevance"}], "errors"}."""
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    logging.getLogger("instructor").setLevel(logging.ERROR)  # rate limits are handled and logged here
    from ragas.metrics.collections import ContextRelevance, Faithfulness, ResponseGroundedness

    chain = _judges()
    state = {"at": 0}
    metric_sets: dict[int, dict] = {}
    ecfg = config.answer()["evaluation"]
    wait = float(ecfg.get("ragas_rate_limit_wait", 30))
    tries = int(ecfg.get("ragas_rate_limit_retries", 3))

    def metrics(at: int) -> dict:
        if at not in metric_sets:
            judge = _make_judge(*chain[at])
            metric_sets[at] = {
                "faithfulness": Faithfulness(llm=judge),
                "response_groundedness": ResponseGroundedness(llm=judge),
                "context_relevance": ContextRelevance(llm=judge),
            }
        return metric_sets[at]

    async def one(key: str, args: dict, errors: list[str]) -> float | None:
        waited = 0
        while state["at"] < len(chain):
            try:
                return _clean((await metrics(state["at"])[key].ascore(**args)).value)
            except Exception as e:  # noqa: BLE001 (one failed judgement must not stop the evaluation)
                kind = _kind(e)
                errors.append(f"{chain[state['at']][0]} {key}: {kind}: {e.__class__.__name__}: {str(e)[:160]}")
                if kind == "rate" and waited < tries and not daily_limit(e):
                    waited += 1
                    pause = min(retry_after(e) or wait, 120.0) + 1.0  # per-minute limits clear soon
                    log(f"  RAGAS judge {chain[state['at']][0]} rate-limited: waiting {pause:.0f}s")
                    await asyncio.sleep(pause)
                    continue
                if kind == "other":
                    # A bad reply for this answer: the judge itself is fine. Ask the next judge once.
                    alt = state["at"] + 1
                    if alt < len(chain):
                        try:
                            return _clean((await metrics(alt)[key].ascore(**args)).value)
                        except Exception as e2:  # noqa: BLE001
                            errors.append(
                                f"{chain[alt][0]} {key}: {_kind(e2)}: {e2.__class__.__name__}: {str(e2)[:160]}"
                            )
                    return None
                state["at"] += 1  # refused, daily limit or still limited: use the next judge from now on
                waited = 0
                if state["at"] < len(chain):
                    log(f"  RAGAS judge switched to {chain[state['at']][0]}")
        return None

    async def run() -> tuple[list[dict], list[str]]:
        rows: list[dict] = []
        errors: list[str] = []
        for n, it in enumerate(items, start=1):
            started = time.monotonic()
            row: dict = {}
            for key in ("faithfulness", "response_groundedness", "context_relevance"):
                args = {"user_input": it["question"], "response": it["answer"], "retrieved_contexts": it["contexts"]}
                if key == "response_groundedness":
                    args.pop("user_input")
                if key == "context_relevance":
                    args.pop("response")
                row[key] = await one(key, args, errors)
            rows.append(row)
            log(f"  RAGAS {n}/{len(items)}: " + ", ".join(f"{k} {v}" for k, v in row.items()))
            if state["at"] >= len(chain):
                log("  RAGAS stopped: every judge model was refused or rate-limited.")
                break
            # Each metric makes about two judge calls; stay under the judge's requests per minute,
            # counting the time the calls themselves took.
            rpm = float(chain[state["at"]][1].get("requests_per_minute") or 0)
            if rpm and n < len(items):
                await asyncio.sleep(max(0.0, 6 * 60.0 / rpm - (time.monotonic() - started)))
        return rows, errors

    rows, errors = asyncio.run(run())
    for err in list(dict.fromkeys(errors))[:3]:
        log(f"  RAGAS error: {err}")
    used = [chain[i][0] for i in sorted(metric_sets)]
    return {"judge": ", ".join(used) or chain[0][0], "rows": rows, "errors": list(dict.fromkeys(errors))[:10]}


def means(rows: list[dict]) -> dict:
    out = {}
    for key in ("faithfulness", "response_groundedness", "context_relevance"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = round(sum(vals) / len(vals), 3) if vals else None
    return out
