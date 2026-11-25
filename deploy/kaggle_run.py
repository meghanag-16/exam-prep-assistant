"""Kaggle launcher: a plain script, no notebook.

On Kaggle: create a new notebook, switch it to Script if offered, and paste this file.
Where the code comes from (pick one):
  a) CODE_REPO_URL below set to your GitHub repo URL (after it is pushed), or
  b) the project folder uploaded as a private Kaggle dataset (Add Input).
Course files: a private Kaggle dataset with one subject's PDFs/PPTs/TXTs/zips (Add Input).
Secrets (Add-ons > Secrets): GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY,
TAVILY_API_KEY, HF_TOKEN, HF_DATASET_REPO. Internet must be On. Ingestion needs no GPU.

All logic lives in the examprep package; this file only sets up the machine
and calls the CLI.
"""

import glob
import os
import shutil
import subprocess
import sys

SUBJECT = os.environ.get("EXAMPREP_SUBJECT", "my_subject")  # CHANGE THIS per subject, e.g. "operating_systems"
# Leave empty to use the code dataset. For a private repo, also add a GITHUB_TOKEN secret.
CODE_REPO_URL = ""
SECRETS = ["GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "TAVILY_API_KEY", "HF_TOKEN", "HF_DATASET_REPO"]
WORK = "/kaggle/working"


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def install_system_tools() -> None:
    sh("apt-get -qq update && apt-get -qq install -y tesseract-ocr libreoffice-impress > /dev/null")


def install_project() -> tuple[str, str | None]:
    """Returns (installed project dir, code dataset dir or None when cloned from GitHub)."""
    dst = os.path.join(WORK, "examprep")
    if CODE_REPO_URL:
        url = CODE_REPO_URL
        try:
            from kaggle_secrets import UserSecretsClient

            token = UserSecretsClient().get_secret("GITHUB_TOKEN")
            url = url.replace("https://", f"https://{token}@")
        except Exception:
            pass  # public repo, no token needed
        shutil.rmtree(dst, ignore_errors=True)
        subprocess.run(["git", "clone", "-q", "--depth", "1", url, dst], check=True)
        sh(f"pip install -q -e {dst}")
        return dst, None
    hits = [h for h in glob.glob("/kaggle/input/**/pyproject.toml", recursive=True)]
    if not hits:
        sys.exit("Project code not found: add the private code dataset as an input.")
    src = os.path.dirname(hits[0])
    shutil.copytree(src, dst, dirs_exist_ok=True)
    sh(f"pip install -q -e {dst}")
    return dst, src


def load_secrets() -> None:
    from kaggle_secrets import UserSecretsClient

    client = UserSecretsClient()
    for name in SECRETS:
        try:
            os.environ[name] = client.get_secret(name)
            print(f"{name}: set")
        except Exception:
            print(f"{name}: MISSING")
    os.environ["EXAMPREP_DATA_DIR"] = os.path.join(WORK, "data")


def course_inputs(code_dir: str | None) -> list[str]:
    """Everything attached under /kaggle/input except the code dataset itself."""
    found: list[str] = []

    def visit(d: str) -> None:
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if code_dir and p == code_dir:
                continue
            if code_dir and code_dir.startswith(p + os.sep):
                visit(p)  # the code sits somewhere inside: look at its siblings only
            else:
                found.append(p)

    if os.path.isdir("/kaggle/input"):
        visit("/kaggle/input")
    return found


def main() -> None:
    install_system_tools()
    project_dir, code_dir = install_project()
    load_secrets()
    sh(f"cd {project_dir} && examprep-check")
    inputs = course_inputs(code_dir)
    if inputs:
        # Downloads the existing ledger from Hugging Face first, skips files already
        # done, OCRs on all CPU cores, then uploads the updated ledger.
        sh(f"examprep-ingest --subject {SUBJECT} --push " + " ".join(f'"{p}"' for p in inputs))
    else:
        print("No course-file dataset attached; setup check only.")


if __name__ == "__main__":
    main()
