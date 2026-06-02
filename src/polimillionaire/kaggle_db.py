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

    Uses Kaggle's per-file download endpoint (`dataset_download_file`) rather
    than the whole-dataset zip. The whole-dataset endpoint returns a 22-byte
    empty zip when the caller authenticates with an OAuth-style
    ``credentials.json`` token whose scope is dataset-read only — silently
    breaking pulls from a laptop. The per-file path works for both OAuth and
    legacy ``kaggle.json`` tokens.

    Returns the target path (always absolute). Sets up the parent dir.
    Idempotent: re-running overwrites the target with the latest version.
    """
    dataset = resolve_dataset_id(dataset_id)
    target = Path(target_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    api = _kaggle_api()
    staging = target.parent / "_kaggle_db_pull"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    try:
        api.dataset_download_file(dataset, db_filename, path=str(staging), force=True, quiet=False)
    except Exception as e:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(f"failed to download {db_filename} from dataset {dataset}: {e}") from e

    # `dataset_download_file` writes either `<file>` or `<file>.zip` depending
    # on whether the server pre-zipped the response. Resolve both.
    candidates = [staging / db_filename, staging / f"{db_filename}.zip"]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        contents = sorted(p.name for p in staging.iterdir())
        shutil.rmtree(staging, ignore_errors=True)
        raise FileNotFoundError(
            f"{db_filename} not found in dataset {dataset} after download. "
            f"Staging dir contents: {contents}"
        )

    if src.suffix == ".zip":
        # Unzip just our file out of the wrapper.
        import zipfile

        with zipfile.ZipFile(src) as zf:
            zf.extract(db_filename, path=staging)
        src = staging / db_filename

    shutil.copy(src, target)
    shutil.rmtree(staging)
    return target


def push_db(
    db_path: str | Path,
    dataset_id: str | None = None,
    *,
    version_notes: str = "session update",
    db_filename: str = DEFAULT_DB_FILENAME,
) -> dict[str, int]:
    """Pull the latest remote DB, merge our new local rows in, then push.

    Cooperative concurrency for a handful of simultaneous Kaggle kernels:

      1. Pull the current dataset version into a tempdir.
      2. Insert any local rows missing from the remote -- dedup key is
         (account_username, session_id, level, mode), so rows from
         different players or different sessions never collide.
      3. Push the merged DB as a new version. Then replace the local
         file with the merged copy so this kernel also sees other
         players' rows (helps db_retrieval cache hits going forward).

    Not transactional: if two pushers race within seconds, the second
    one wins the version slot but loses the first's rows. That first
    player will re-merge on their *next* push (pull-then-merge is
    unconditional), so the system converges. Fine for a few concurrent
    players; do not rely on it for high-rate writers.

    Only the `predictions` table is merged. The `meta` side-table is
    left at the target's value (the `index_valid` sentinel can be
    re-set by any client that observes a collision; not worth a
    bespoke merge rule here).

    For the rare case where the remote dataset has no DB file yet
    (first push after `create_db_dataset`), falls back to a plain
    overwrite seed.

    Returns a stats dict {"inserted": N, "skipped": M} so callers can
    log how much was merged in.
    """
    local = Path(db_path).resolve()
    if not local.exists():
        raise FileNotFoundError(f"no local DB at {local}")

    _checkpoint_wal(local)
    dataset = resolve_dataset_id(dataset_id)
    api = _kaggle_api()

    pull_dir = local.parent / "_kaggle_db_merge_pull"
    push_dir = local.parent / "_kaggle_db_merge_push"
    for d in (pull_dir, push_dir):
        if d.exists():
            shutil.rmtree(d)
    pull_dir.mkdir()

    try:
        api.dataset_download_files(dataset, path=str(pull_dir), unzip=True, quiet=False)
    except Exception as e:
        shutil.rmtree(pull_dir)
        raise RuntimeError(f"failed to pull remote dataset {dataset}: {e}") from e

    remote_db = pull_dir / db_filename
    if not remote_db.exists():
        # Dataset exists but has no DB file -- nothing to merge into.
        # Fall back to seeding it. (Rare; should only happen once.)
        shutil.rmtree(pull_dir)
        _push_db_overwrite(
            local,
            dataset_id,
            version_notes=version_notes,
            db_filename=db_filename,
        )
        return {"inserted": 0, "skipped": 0}

    stats = _merge_predictions(local, remote_db)
    _checkpoint_wal(remote_db)

    push_dir.mkdir()
    shutil.copy(remote_db, push_dir / db_filename)
    (push_dir / "dataset-metadata.json").write_text(
        json.dumps({"id": dataset, "title": dataset.split("/")[-1]})
    )
    notes = f"{version_notes} (merge +{stats['inserted']} new, {stats['skipped']} dup)"
    api.dataset_create_version(str(push_dir), version_notes=notes, dir_mode="zip")

    # Adopt the merged DB locally so subsequent sessions -- or any
    # continuation of this one -- see other players' rows.
    shutil.copy(remote_db, local)

    shutil.rmtree(pull_dir)
    shutil.rmtree(push_dir)
    return stats


def _push_db_overwrite(
    db_path: str | Path,
    dataset_id: str | None = None,
    *,
    version_notes: str = "session update",
    db_filename: str = DEFAULT_DB_FILENAME,
) -> None:
    """Upload `db_path` verbatim as a new dataset version, overwriting
    whatever was there. Internal helper for the first-push case; normal
    pushes go through `push_db` which pulls + merges first.
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


def _merge_predictions(local_db: Path, target_db: Path) -> dict[str, int]:
    """Insert rows from local.predictions into target.predictions that
    don't already match a target row on (account_username, session_id,
    level, mode). Returns {"inserted", "skipped"}.

    Target gets the current schema applied first (CREATE IF NOT EXISTS +
    migrate) so older remote snapshots gain any new columns before the
    insert. The autoincrement `id` is dropped from the insert list --
    target's autoincrement assigns fresh ids.
    """
    # Inspect local schema in a separate connection: dodges any quirks with
    # PRAGMA on attached databases.
    with sqlite3.connect(str(local_db)) as lc:
        local_cols = {r[1] for r in lc.execute("PRAGMA table_info(predictions)").fetchall()}

    target_con = sqlite3.connect(str(target_db))
    try:
        # Defensive: bring target up to current schema before merging.
        # SCHEMA is CREATE TABLE IF NOT EXISTS, so it's a no-op on a healthy
        # remote and a recovery for a corrupted/empty one.
        from polimillionaire.recording import SCHEMA, _migrate

        target_con.executescript(SCHEMA)
        _migrate(target_con)

        target_cols = [
            r[1]
            for r in target_con.execute("PRAGMA table_info(predictions)").fetchall()
            if r[1] != "id"
        ]
        cols = [c for c in target_cols if c in local_cols]
        if not cols:
            return {"inserted": 0, "skipped": 0}
        col_csv = ", ".join(cols)

        before = target_con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

        target_con.execute("ATTACH DATABASE ? AS local", (str(local_db),))
        total_local = target_con.execute("SELECT COUNT(*) FROM local.predictions").fetchone()[0]
        target_con.execute(
            f"""
            INSERT INTO predictions ({col_csv})
            SELECT {col_csv}
            FROM local.predictions L
            WHERE NOT EXISTS (
                SELECT 1 FROM predictions T
                WHERE T.account_username = L.account_username
                  AND T.session_id       = L.session_id
                  AND T.level            = L.level
                  AND T.mode             = L.mode
            )
            """  # noqa: S608 -- col_csv is built from PRAGMA, not user input
        )
        target_con.commit()
        after = target_con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        target_con.execute("DETACH DATABASE local")

        inserted = after - before
        skipped = total_local - inserted
        return {"inserted": inserted, "skipped": skipped}
    finally:
        target_con.close()


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
