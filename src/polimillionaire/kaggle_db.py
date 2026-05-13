"""Kaggle-Dataset round-trip for the question log.

Kaggle has no persistent writable filesystem like Drive — `/kaggle/working/`
is wiped at session end. To make the SQLite log survive across sessions,
we treat a private Kaggle Dataset as the canonical store:

    session start: pull the latest dataset version into /kaggle/working/
    session end:   WAL-checkpoint, then push a new dataset version

`pull_db` and `push_db` are the two functions the notebook uses. They
import the `kaggle` package lazily so the rest of the project doesn't
need it as a hard dep — it lives in the `kaggle` optional group, and
the Kaggle environment has it pre-installed anyway.

One-time setup (run locally once to seed the dataset from the current
`data/questions.sqlite`): see `scripts/kaggle_db_init.py`.

Authentication:
- Locally: ~/.kaggle/kaggle.json (username + key).
- On Kaggle: enable "Internet" in the kernel and add KAGGLE_USERNAME +
  KAGGLE_KEY as Kernel Secrets; the kaggle package picks them up.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path

DEFAULT_DATASET_SLUG = "polimillionaire-question-log"
DEFAULT_DB_FILENAME = "questions.sqlite"


def _kaggle_api():
    """Lazy import + authenticate. Raises a friendly error if the kaggle
    package isn't installed or credentials are missing."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError(
            "kaggle package not installed. On Kaggle it's pre-installed; "
            "locally: `pip install kaggle` and place an API token at "
            "~/.kaggle/kaggle.json (Account → Create New Token)."
        ) from e
    api = KaggleApi()
    try:
        api.authenticate()
    except OSError as e:
        raise RuntimeError(
            "kaggle authentication failed. Put your API token at "
            "~/.kaggle/kaggle.json (locally) or set KAGGLE_USERNAME + "
            "KAGGLE_KEY (Kaggle Secrets)."
        ) from e
    return api


def resolve_dataset_id(explicit: str | None = None) -> str:
    """Resolve the full `owner/dataset` slug.

    Priority: explicit arg > POLIMILLIONAIRE_KAGGLE_DATASET env var >
    `<kaggle_username>/polimillionaire-question-log` (auto-prefixed from
    the kaggle.json username).
    """
    if explicit:
        return explicit
    env = os.environ.get("POLIMILLIONAIRE_KAGGLE_DATASET")
    if env:
        return env
    api = _kaggle_api()
    username = api.config_values.get("username")
    if not username:
        raise RuntimeError(
            "no Kaggle username in config; pass dataset_id explicitly or "
            "set POLIMILLIONAIRE_KAGGLE_DATASET=<owner>/<slug>"
        )
    return f"{username}/{DEFAULT_DATASET_SLUG}"


def pull_db(
    dataset_id: str | None = None,
    *,
    target_path: str | Path = "/kaggle/working/questions.sqlite",
    db_filename: str = DEFAULT_DB_FILENAME,
) -> Path:
    """Download the dataset's current version and copy the DB to `target_path`.

    Returns the target path (always absolute). Sets up the parent dir.
    Idempotent: re-running overwrites the target with the latest version.
    """
    dataset = resolve_dataset_id(dataset_id)
    target = Path(target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    api = _kaggle_api()
    # `dataset_download_files` writes a zip by default and unzips it.
    # We download to a temp staging dir to avoid clobbering other files
    # in target.parent.
    staging = target.parent / "_kaggle_db_pull"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    api.dataset_download_files(dataset, path=str(staging), unzip=True, quiet=False)

    src = staging / db_filename
    if not src.exists():
        # Some downloads keep the zip if the API decides not to unzip.
        zip_path = staging / f"{dataset.split('/')[-1]}.zip"
        raise FileNotFoundError(
            f"{db_filename} not found in dataset {dataset}. "
            f"Staging dir contents: {sorted(p.name for p in staging.iterdir())}. "
            f"Zip path checked: {zip_path}"
        )
    shutil.copy(src, target)
    shutil.rmtree(staging)
    return target


def push_db(
    db_path: str | Path,
    dataset_id: str | None = None,
    *,
    version_notes: str = "session update",
    db_filename: str = DEFAULT_DB_FILENAME,
) -> None:
    """WAL-checkpoint `db_path`, then upload it as a new dataset version.

    The dataset must already exist (one-time create via
    `scripts/kaggle_db_init.py`). For a fresh dataset, use
    `create_db_dataset` instead.
    """
    dataset = resolve_dataset_id(dataset_id)
    db = Path(db_path).resolve()
    if not db.exists():
        raise FileNotFoundError(f"no DB at {db}")

    _checkpoint_wal(db)

    api = _kaggle_api()
    staging = db.parent / "_kaggle_db_push"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    shutil.copy(db, staging / db_filename)
    (staging / "dataset-metadata.json").write_text(
        json.dumps({"id": dataset, "title": dataset.split("/")[-1]})
    )
    api.dataset_create_version(
        str(staging),
        version_notes=version_notes,
        dir_mode="zip",
    )
    shutil.rmtree(staging)


def create_db_dataset(
    db_path: str | Path,
    dataset_id: str | None = None,
    *,
    title: str | None = None,
    public: bool = False,
    db_filename: str = DEFAULT_DB_FILENAME,
) -> str:
    """Create a NEW Kaggle Dataset seeded with the current DB.

    One-time setup. Use `push_db` for subsequent updates.
    Returns the resolved dataset id so the caller can record it.
    """
    dataset = resolve_dataset_id(dataset_id)
    db = Path(db_path).resolve()
    if not db.exists():
        raise FileNotFoundError(f"no DB at {db}")

    _checkpoint_wal(db)

    api = _kaggle_api()
    staging = db.parent / "_kaggle_db_create"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    shutil.copy(db, staging / db_filename)
    (staging / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset,
                "title": title or dataset.split("/")[-1],
                "licenses": [{"name": "CC0-1.0"}],
            }
        )
    )
    api.dataset_create_new(
        str(staging),
        public=public,
        quiet=False,
        dir_mode="zip",
    )
    shutil.rmtree(staging)
    return dataset


def _checkpoint_wal(db_path: Path) -> None:
    """Fold the WAL sidecar into the main DB file so the upload is
    self-contained. Without this, an active WAL can leave uncommitted
    rows out of the snapshot."""
    con = sqlite3.connect(str(db_path))
    try:
        con.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    finally:
        con.close()
