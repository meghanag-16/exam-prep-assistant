"""Settings: secrets come from environment variables (.env locally, Space/Kaggle
secrets when hosted); everything else comes from the YAML files in config/."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.getenv("EXAMPREP_CONFIG_DIR", PROJECT_ROOT / "config"))

load_dotenv(PROJECT_ROOT / ".env", override=False)

SECRET_ENV_VARS = (
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "OPENROUTER_API_KEY",
    "TAVILY_API_KEY",
    "HF_TOKEN",
)


def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache
def models() -> dict[str, Any]:
    return _load_yaml("models.yaml")


@lru_cache
def exam_pattern() -> dict[str, Any]:
    return _load_yaml("exam_pattern.yaml")


@lru_cache
def trusted_sites() -> dict[str, Any]:
    return _load_yaml("trusted_sites.yaml")


@lru_cache
def ingest() -> dict[str, Any]:
    return _load_yaml("ingest.yaml")


@lru_cache
def structure() -> dict[str, Any]:
    return _load_yaml("structure.yaml")


@lru_cache
def formulas() -> dict[str, Any]:
    return _load_yaml("formulas.yaml")


@lru_cache
def search() -> dict[str, Any]:
    return _load_yaml("search.yaml")


def secret(name: str) -> str | None:
    """Return a secret from the environment, or None if unset/blank."""
    value = os.getenv(name, "").strip()
    return value or None


def data_dir() -> Path:
    path = Path(os.getenv("EXAMPREP_DATA_DIR", PROJECT_ROOT / "data"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def allowed_domains(subject: str | None = None) -> list[str]:
    cfg = trusted_sites()
    domains = list(cfg.get("default") or [])
    if subject:
        domains += list((cfg.get("subjects") or {}).get(subject) or [])
    return sorted(set(domains))
