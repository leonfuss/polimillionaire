"""Tests for the DB path resolver in play.py.

Regression: running auto_play_loop from a subdirectory used to drop the
SQLite log into `<cwd>/data/questions.sqlite` instead of the project root,
splitting the corpus across multiple databases. The resolver must always
anchor relative paths to the repo root.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from polimillionaire.play import _PROJECT_ROOT, _resolve_db_path


def test_default_path_is_project_root_data_dir() -> None:
    out = _resolve_db_path(None)
    assert Path(out) == _PROJECT_ROOT / "data" / "questions.sqlite"


def test_relative_path_arg_anchors_to_project_root() -> None:
    out = _resolve_db_path("data/elsewhere.sqlite")
    assert Path(out) == _PROJECT_ROOT / "data" / "elsewhere.sqlite"


def test_absolute_path_arg_is_returned_as_is() -> None:
    out = _resolve_db_path("/tmp/explicit.sqlite")
    assert Path(out) == Path("/tmp/explicit.sqlite")


def test_env_var_used_when_arg_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLIMILLIONAIRE_DB_PATH", "data/from_env.sqlite")
    out = _resolve_db_path(None)
    # Even env-derived relative paths anchor to project root.
    assert Path(out) == _PROJECT_ROOT / "data" / "from_env.sqlite"


def test_resolution_independent_of_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("POLIMILLIONAIRE_DB_PATH", raising=False)
    out = _resolve_db_path(None)
    # Must NOT be `tmp_path / "data" / "questions.sqlite"`.
    assert os.fspath(tmp_path) not in out
    assert Path(out).is_absolute()
    assert "questions.sqlite" in out
