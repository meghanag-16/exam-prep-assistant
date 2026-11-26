"""Setup check: `uv run examprep-check` (add --live to spend one Tavily credit).

Never prints secret values, only whether they are set and whether they work.
"""

from __future__ import annotations

import argparse
import shutil
import sys

import httpx

from examprep import config, llm
from examprep.ingest.convert import available_converter

OK, WARN, FAIL, INFO = "OK  ", "WARN", "FAIL", "INFO"


def secret_status() -> dict[str, bool]:
    return {name: config.secret(name) is not None for name in config.SECRET_ENV_VARS}


def _model_ids(payload: dict) -> set[str]:
    ids = set()
    for item in payload.get("data", []):
        mid = str(item.get("id", ""))
        ids.add(mid)
        ids.add(mid.removeprefix("models/"))
    return ids


def check_llm_provider(name: str, cfg: dict, timeout: float) -> tuple[str, str]:
    key = config.secret(cfg["api_key_env"])
    if not key:
        return FAIL, f"{cfg['api_key_env']} is not set"
    url = cfg["base_url"].rstrip("/") + "/models"
    try:
        r = httpx.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
    except httpx.HTTPError as e:
        return FAIL, f"could not reach {url}: {e.__class__.__name__}"
    if r.status_code in (401, 403):
        return FAIL, f"key rejected (HTTP {r.status_code})"
    if r.status_code != 200:
        return FAIL, f"HTTP {r.status_code} from {url}"
    ids = _model_ids(r.json())
    models = llm.provider_models(cfg)
    missing = [m for m in models if m not in ids]
    if not missing:
        return OK, f"key works, model(s) available: {', '.join(models)}"
    raw = sorted(i for i in ids if not i.startswith("models/"))
    if name == "openrouter":
        raw = [i for i in raw if i.endswith(":free")]
    shown = ", ".join(raw[:25]) + (" ..." if len(raw) > 25 else "")
    return WARN, (
        f"key works, but model(s) {', '.join(missing)} were not in the provider's list. "
        f"Pick from these in config/models.yaml: {shown}"
    )


def check_hf(timeout: float) -> list[tuple[str, str, str]]:
    out = []
    token = config.secret("HF_TOKEN")
    if not token:
        return [("Hugging Face", FAIL, "HF_TOKEN is not set")]
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        user = api.whoami().get("name", "?")
        out.append(("Hugging Face", OK, f"token works (user: {user})"))
        repo = config.secret("HF_DATASET_REPO")
        if not repo:
            out.append(("HF dataset repo", WARN, "HF_DATASET_REPO is not set yet"))
        elif api.repo_exists(repo, repo_type="dataset"):
            out.append(("HF dataset repo", OK, f"{repo} exists"))
        else:
            out.append(("HF dataset repo", FAIL, f"{repo} not found (create it as a private dataset)"))
    except Exception as e:  # noqa: BLE001
        out.append(("Hugging Face", FAIL, f"{e.__class__.__name__}: {e}"))
    return out


def check_tavily(live: bool, timeout: float) -> tuple[str, str]:
    key = config.secret("TAVILY_API_KEY")
    if not key:
        return FAIL, "TAVILY_API_KEY is not set"
    if not live:
        return OK, "key is set (run with --live to test it, costs 1 search credit)"
    try:
        r = httpx.post(
            "https://api.tavily.com/search",
            json={
                "query": "deadlock prevention",
                "max_results": 1,
                "include_domains": config.allowed_domains(),
            },
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        return FAIL, f"could not reach Tavily: {e.__class__.__name__}"
    if r.status_code == 200:
        return OK, "key works"
    return FAIL, f"HTTP {r.status_code} from Tavily"


def check_tools() -> list[tuple[str, str, str]]:
    rows = []
    rows.append(
        ("Tesseract OCR", OK, "found")
        if shutil.which("tesseract")
        else ("Tesseract OCR", INFO, "not installed locally (expected: OCR runs on Kaggle and the Space)")
    )
    conv = available_converter()
    rows.append(
        (".ppt converter", OK, conv)
        if conv
        else (".ppt converter", INFO, "none locally (expected: .ppt conversion runs on Kaggle and the Space)")
    )
    return rows


def run(live: bool = False) -> int:
    timeout = float(config.models()["llm"].get("timeout_seconds", 60))
    rows: list[tuple[str, str, str]] = []
    for name, cfg in config.models()["llm"]["providers"].items():
        rows.append((f"LLM: {name}", *check_llm_provider(name, cfg, timeout)))
    rows.append(("Tavily search", *check_tavily(live, timeout)))
    rows += check_hf(timeout)
    rows += check_tools()

    width = max(len(r[0]) for r in rows)
    for label, status, msg in rows:
        print(f"[{status}] {label.ljust(width)}  {msg}")

    llm_ok = any(s == OK for label, s, _ in rows if label.startswith("LLM:"))
    failures = [r for r in rows if r[1] == FAIL]
    print()
    if not llm_ok:
        print("No LLM provider is working yet.")
    print(f"{len(failures)} failing, {sum(r[1] == WARN for r in rows)} warnings.")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check examprep setup")
    parser.add_argument("--live", action="store_true", help="also test the Tavily key (1 credit)")
    args = parser.parse_args()
    sys.exit(run(live=args.live))


if __name__ == "__main__":
    main()
