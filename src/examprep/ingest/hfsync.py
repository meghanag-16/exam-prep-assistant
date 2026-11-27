"""Sync a subject's ledger with the Hugging Face dataset repo.

Layout in the dataset repo:  subjects/<subject>/ledger.sqlite plus the JSON/YAML reports,
the search index (index/), evaluation files (eval/) and search_settings.json,
notes/ (notes, revision sheets and their reports), and cache/llm_cache.sqlite and
cache/web_cache.sqlite (saved LLM replies and trusted-site pages, shared by all subjects)

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
WEB_CACHE_FILE = "cache/web_cache.sqlite"


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
    pull_web_cache()
    pull_search(subject)
    return True


SEARCH_FILES = ["index/**", "eval/*", "search_settings.json", "notes/*"]


def pull_search(subject: str, overwrite_questions: bool = False) -> int:
    """Download the search index, evaluation files and chosen search settings, if uploaded.
    A local eval/questions.yaml is kept (it may hold your edits) unless overwrite_questions.
    Returns the number of files written."""
    import shutil
    from pathlib import Path

    from huggingface_hub import snapshot_download

    repo, token = _repo_and_token()
    remote = remote_dir(subject)
    try:
        root = snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            allow_patterns=[f"{remote}/{p}" for p in SEARCH_FILES],
            token=token,
            cache_dir=config.data_dir() / ".hf-cache",
        )
    except Exception:
        return 0  # nothing uploaded yet, or not reachable: search can be rebuilt with examprep-index
    src = Path(root) / remote
    dst = ledger.subject_dir(subject)
    written = 0
    if (src / "index").exists():
        shutil.rmtree(dst / "index", ignore_errors=True)
    for f in src.rglob("*"):
        if not f.is_file():
            continue
        rel = f.relative_to(src)
        target = dst / rel
        if rel.as_posix() == "eval/questions.yaml" and target.exists() and not overwrite_questions:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f, target)
        written += 1
    return written


def push_questions(subject: str) -> str:
    """Upload only eval/questions.yaml (after editing it by hand)."""
    from huggingface_hub import HfApi

    repo, token = _repo_and_token()
    local = ledger.subject_dir(subject) / "eval" / "questions.yaml"
    if not local.exists():
        raise SyncError(f"{local} does not exist.")
    HfApi(token=token).upload_file(
        repo_id=repo,
        repo_type="dataset",
        path_or_fileobj=str(local),
        path_in_repo=f"{remote_dir(subject)}/eval/questions.yaml",
        commit_message=f"update evaluation questions for {ledger.safe_subject(subject)}",
    )
    return f"{repo}/{remote_dir(subject)}/eval/questions.yaml"


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


def pull_web_cache() -> bool:
    """Merge the saved trusted-site searches and pages from the dataset repo into the local cache."""
    import sqlite3

    from huggingface_hub import hf_hub_download

    from examprep.answer import trusted

    repo, token = _repo_and_token()
    try:
        path = hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=WEB_CACHE_FILE,
            token=token,
            cache_dir=config.data_dir() / ".hf-cache",
        )
    except Exception:
        return False
    try:
        trusted._db().close()
        con = sqlite3.connect(trusted.cache_path(), timeout=30)
        try:
            con.execute("ATTACH DATABASE ? AS remote", (str(path),))
            con.execute("INSERT OR IGNORE INTO searches SELECT key, query, results, created FROM remote.searches")
            con.execute("INSERT OR IGNORE INTO pages SELECT url, text, created FROM remote.pages")
            con.commit()
        finally:
            con.close()
    except sqlite3.Error:
        return False
    return True


def push_cache() -> None:
    from huggingface_hub import HfApi

    from examprep import llm
    from examprep.answer import trusted

    repo, token = _repo_and_token()
    for local, remote, what in (
        (llm.cache_path(), CACHE_FILE, "LLM reply cache"),
        (trusted.cache_path(), WEB_CACHE_FILE, "trusted-site cache"),
    ):
        if local.exists():
            HfApi(token=token).upload_file(
                repo_id=repo,
                repo_type="dataset",
                path_or_fileobj=str(local),
                path_in_repo=remote,
                commit_message=f"update {what}",
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
            *SEARCH_FILES,
        ],
        ignore_patterns=["**/.lock", "**/*.lock"],
        commit_message=message or f"ingest {ledger.safe_subject(subject)}",
    )
    push_cache()
    return f"{repo}/{remote_dir(subject)}"
