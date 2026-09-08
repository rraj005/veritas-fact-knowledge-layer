"""One-shot deploy of Veritas to a Hugging Face Docker Space.

Creates (or reuses) a Space named "veritas" under your account and uploads the
repo. HF then builds the Dockerfile and serves the app (API + UI) on port 7860.

Usage (from the repo root):

    # 1. Create a WRITE token at https://huggingface.co/settings/tokens
    # 2. Provide it via env var, then run this script with the project venv:

    #   PowerShell:
    #     $env:HF_TOKEN = "hf_xxx"
    #     .venv\\Scripts\\python.exe deploy\\huggingface\\deploy_space.py
    #   bash:
    #     export HF_TOKEN=hf_xxx
    #     python deploy/huggingface/deploy_space.py

The LLM is bring-your-own-key (entered in the UI at runtime), so NO secrets are
uploaded or stored in the Space.
"""

from __future__ import annotations

import os
import sys

from huggingface_hub import HfApi

SPACE_NAME = "veritas"

# Everything the running app does not need in the Space repo.
IGNORE = [
    ".git*",
    ".git/*",
    ".venv/*",
    "venv/*",
    "env/*",
    "data/*",
    "**/__pycache__/*",
    "*.egg-info/*",
    ".pytest_cache/*",
    ".ruff_cache/*",
    ".mypy_cache/*",
    "starter-datasets.zip",
    ".superpowers/*",
    ".env",
    ".env.*",
    # The Space README must carry the Docker front-matter, so skip the project
    # README here and upload the dedicated one below.
    "README.md",
]


def main() -> int:
    token = os.getenv("HF_TOKEN")
    if not token:
        print(
            "ERROR: set HF_TOKEN to a Hugging Face WRITE token first.\n"
            "  Create one at https://huggingface.co/settings/tokens",
            file=sys.stderr,
        )
        return 1

    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/{SPACE_NAME}"

    print(f"Creating/reusing Docker Space: {repo_id}")
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    print("Uploading project files ...")
    api.upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=IGNORE,
        commit_message="Deploy Veritas fact knowledge layer",
    )

    print("Uploading Space README (Docker front-matter) ...")
    api.upload_file(
        path_or_fileobj="deploy/huggingface/README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
        commit_message="Add Space configuration",
    )

    print(f"\nDone. HF is now building the image.\n  https://huggingface.co/spaces/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
