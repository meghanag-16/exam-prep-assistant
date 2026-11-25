"""Sync a subject's ledger with the Hugging Face dataset repo.

Layout in the dataset repo:  subjects/<subject>/ledger.sqlite plus the JSON/YAML reports

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
    return True


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
        allow_patterns=["ledger.sqlite", "report.json", "topics.yaml", "topics_report.json", "syllabus_report.json"],
        commit_message=message or f"ingest {ledger.safe_subject(subject)}",
    )
    return f"{repo}/{remote_dir(subject)}"
