"""Sync a subject's ledger with the Hugging Face dataset repo.

Layout in the dataset repo:  subjects/<subject>/ledger.sqlite plus the JSON/YAML reports,
and cache/llm_cache.sqlite (saved LLM replies, shared by all subjects)

- pull: download the ledger before ingesting, so a fresh Kaggle session knows
  what was already processed and skips it.
- push: upload the ledger after ingesting, so the app (and the next run) sees it.
"""

from __future__ import annotations

import os

from examprep import config
from examprep.ingest import ledger


class SyncError(RuntimeError):
    pass


def _repo_and_token() -> tuple[str, str]:
    repo = os.getenv("HF_DATASET_REPO", "").strip()
    token = config.secret("HF_TOKEN")
    if not repo or not token:
        raise SyncError("HF_DATASET_REPO and HF_TOKEN must both be set to sync with Hugging Face.")
    return repo, token


CACHE_FILE = "cache/llm_cache.sqlite"


def remote_dir(subject: str) -> str:
    return f"subjects/{ledger.safe_subject(subject)}"


def pull(subject: str) -> bool:
    """Download the subject's ledger if the repo has one. Returns True if downloaded.
    An existing local ledger is replaced by the remote copy."""
    from huggingface_hub import hf_hub_download

    try:
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError
    except ImportError:  # older huggingface_hub
        from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    repo, token = _repo_and_token()
    try:
        path = hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=f"{remote_dir(subject)}/ledger.sqlite",
            token=token,
            cache_dir=config.data_dir() / ".hf-cache",  # keep downloads inside the project's data folder
        )
    except EntryNotFoundError:
        return False
    except RepositoryNotFoundError as e:
        raise SyncError(f"dataset repo {repo} not found, or the token cannot access it") from e
    with open(path, "rb") as f:
        ledger.db_path(subject).write_bytes(f.read())
    pull_cache()
    return True


def pull_cache() -> bool:
    """Merge the saved LLM replies from the dataset repo into the local cache."""
    import sqlite3

    from huggingface_hub import hf_hub_download

    from examprep import llm

    repo, token = _repo_and_token()
    try:
        path = hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=CACHE_FILE,
            token=token,
            cache_dir=config.data_dir() / ".hf-cache",
        )
    except Exception:
        return False  # no cache uploaded yet, or not reachable: the run works without it
    try:
        llm._cache_db().close()  # make sure the local table exists
        con = sqlite3.connect(llm.cache_path(), timeout=30)
        try:
            con.execute("ATTACH DATABASE ? AS remote", (str(path),))
            con.execute("INSERT OR IGNORE INTO replies SELECT key, reply, model, created FROM remote.replies")
            con.commit()
        finally:
            con.close()
    except sqlite3.Error:
        return False
    return True


def push_cache() -> None:
    from huggingface_hub import HfApi

    from examprep import llm

    if not llm.cache_path().exists():
        return
    repo, token = _repo_and_token()
    HfApi(token=token).upload_file(
        repo_id=repo,
        repo_type="dataset",
        path_or_fileobj=str(llm.cache_path()),
        path_in_repo=CACHE_FILE,
        commit_message="update LLM reply cache",
    )


def push(subject: str, message: str | None = None) -> str:
    """Upload the subject's ledger and reports. Returns the repo path."""
    from huggingface_hub import HfApi

    repo, token = _repo_and_token()
    folder = ledger.subject_dir(subject)
    HfApi(token=token).upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(folder),
        path_in_repo=remote_dir(subject),
        allow_patterns=[
            "ledger.sqlite",
            "report.json",
            "topics.yaml",
            "topics_report.json",
            "syllabus_report.json",
            "formulas.md",
            "formulas_report.json",
        ],
        commit_message=message or f"ingest {ledger.safe_subject(subject)}",
    )
    push_cache()
    return f"{repo}/{remote_dir(subject)}"
