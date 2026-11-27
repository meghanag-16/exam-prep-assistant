"""RAGAS scores for grounded answers (examprep-answer-eval).

Three RAGAS metrics that need no reference answer, so they work on any subject's
evaluation questions:
- faithfulness: share of the answer's claims supported by the passages given to the model.
- response groundedness: how well the answer as a whole is grounded in those passages.
- context relevance: how relevant the retrieved passages are to the question.

RAGAS calls its judge model itself (through an OpenAI-compatible client), using the first
text provider in config/models.yaml that has a key, or evaluation.ragas_provider in
config/answer.yaml. RAGAS is in the `eval` extra (installed on Kaggle); without it these
scores are skipped and the report says so. RAGAS's anonymous usage tracking is turned off.
"""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Callable

from examprep import config, llm

Log = Callable[[str], None]


def available() -> bool:
    try:
        import ragas  # noqa: F401
    except ImportError:
        return False
    return True


def _judge():
    """(RAGAS LLM, "provider/model") for the judge model."""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    cfg = config.models()["llm"]
    wanted = config.answer()["evaluation"].get("ragas_provider")
    names = [wanted] if wanted else [n for n in llm.available_providers() if not cfg["providers"][n].get("optional")]
    names = [n for n in names if n and config.secret(cfg["providers"][n]["api_key_env"])]
    if not names:
        names = llm.available_providers()
    if not names:
        raise RuntimeError("no LLM provider key is set for the RAGAS judge")
    p = cfg["providers"][names[0]]
    model = llm.provider_models(p)[0]
    client = AsyncOpenAI(
        base_url=p["base_url"], api_key=config.secret(p["api_key_env"]), timeout=float(cfg.get("timeout_seconds", 60))
    )
    return llm_factory(model, provider="openai", client=client, temperature=0.0), f"{names[0]}/{model}", p


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
    from ragas.metrics.collections import ContextRelevance, Faithfulness, ResponseGroundedness

    judge, name, provider = _judge()
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "response_groundedness": ResponseGroundedness(llm=judge),
        "context_relevance": ContextRelevance(llm=judge),
    }
    # Each metric makes about two judge calls; keep under the provider's requests per minute.
    rpm = float(provider.get("requests_per_minute") or 0)
    gap = 6 * 60.0 / rpm if rpm else 0.0

    async def run() -> tuple[list[dict], list[str]]:
        rows, errors = [], []
        for n, it in enumerate(items, start=1):
            row: dict = {}
            for key, metric in metrics.items():
                args = {"user_input": it["question"], "response": it["answer"], "retrieved_contexts": it["contexts"]}
                if key == "response_groundedness":
                    args.pop("user_input")
                if key == "context_relevance":
                    args.pop("response")
                try:
                    row[key] = _clean((await metric.ascore(**args)).value)
                except Exception as e:  # noqa: BLE001 (one failed judgement must not stop the evaluation)
                    row[key] = None
                    errors.append(f"{key}: {e.__class__.__name__}: {str(e)[:120]}")
            rows.append(row)
            log(f"  RAGAS {n}/{len(items)}: " + ", ".join(f"{k} {v}" for k, v in row.items()))
            if gap and n < len(items):
                await asyncio.sleep(gap)
        return rows, errors

    rows, errors = asyncio.run(run())
    return {"judge": name, "rows": rows, "errors": sorted(set(errors))[:10]}


def means(rows: list[dict]) -> dict:
    out = {}
    for key in ("faithfulness", "response_groundedness", "context_relevance"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = round(sum(vals) / len(vals), 3) if vals else None
    return out
