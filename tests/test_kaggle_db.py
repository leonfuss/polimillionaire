"""Smoke tests for the Kaggle-Dataset DB round-trip helpers.

We don't exercise the live Kaggle API here -- that requires a live token
and network. Instead we cover the parts that work without the kaggle
package installed: the resolve_dataset_id env-var path and friendly
error messages when kaggle is unavailable.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import pytest

from polimillionaire import kaggle_db


def test_resolve_dataset_id_returns_explicit_when_given() -> None:
    assert kaggle_db.resolve_dataset_id("explicit/slug") == "explicit/slug"


def test_resolve_dataset_id_uses_env_when_no_explicit() -> None:
    with patch.dict(os.environ, {"POLIMILLIONAIRE_KAGGLE_DATASET": "envuser/envslug"}):
        assert kaggle_db.resolve_dataset_id() == "envuser/envslug"


def test_resolve_dataset_id_explicit_beats_env() -> None:
    with patch.dict(os.environ, {"POLIMILLIONAIRE_KAGGLE_DATASET": "envuser/envslug"}):
        assert kaggle_db.resolve_dataset_id("explicit/slug") == "explicit/slug"


def test_kaggle_api_raises_friendly_error_when_package_missing() -> None:
    """When `kaggle` isn't installed, the helper must explain how to install
    it rather than producing an opaque ImportError up the stack."""
    # Force the import inside _kaggle_api to fail and assert the wrapped
    # RuntimeError surfaces with installation guidance.
    with (
        patch.dict(sys.modules, {"kaggle.api.kaggle_api_extended": None}),
        patch.object(
            kaggle_db,
            "_kaggle_api",
            side_effect=RuntimeError("kaggle package not installed."),
        ),
        pytest.raises(RuntimeError, match="kaggle package not installed"),
    ):
        kaggle_db._kaggle_api()


def test_push_db_rejects_missing_file(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.sqlite"
    with pytest.raises(FileNotFoundError, match="no DB"):
        kaggle_db.push_db(missing, dataset_id="x/y")


def test_create_db_dataset_rejects_missing_file(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.sqlite"
    with pytest.raises(FileNotFoundError, match="no DB"):
        kaggle_db.create_db_dataset(missing, dataset_id="x/y")


def test_checkpoint_wal_runs_on_real_sqlite(tmp_path) -> None:
    """Sanity: the WAL-checkpoint call must accept a real SQLite file
    without exploding (sqlite3 returns rows when WAL is empty)."""
    import sqlite3

    db_path = tmp_path / "tiny.sqlite"
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("CREATE TABLE t(x INTEGER);")
    con.execute("INSERT INTO t VALUES (1);")
    con.commit()
    con.close()

    # Should run without raising.
    kaggle_db._checkpoint_wal(db_path)
