"""Smoke tests: run on every commit via the pre-commit / CI gate."""

import sqlite3

from polimillionaire.recording import PredictionRecord, QuestionLog
from polimillionaire.strategies.base import AnswerDecision, Context


def test_question_log_roundtrip(tmp_path):
    log = QuestionLog(tmp_path / "test.sqlite")
    rec = PredictionRecord(
        account_username="alice",
        session_id=1,
        competition_id=2,
        level=3,
        question_id=42,
        question_text="What is 2+2?",
        options=[{"id": 0, "text": "3"}, {"id": 1, "text": "4"}],
        predicted_option_id=1,
        correct_option_id_if_known=1,
        strategy_name="zero_shot",
        model_name="dummy",
        prompt_version="v1",
        confidence=0.9,
        rationale="basic arithmetic",
        latency_ms=10,
    )
    log.record(rec)

    assert log.has_question(42)
    assert not log.has_question(99)


def test_answer_decision_defaults():
    d = AnswerDecision(option_id=2)
    assert d.option_id == 2
    assert d.confidence is None
    assert d.rationale is None


def test_context_fields():
    ctx = Context(competition_id=1, level=5)
    assert ctx.competition_id == 1
    assert ctx.level == 5


def _record(**overrides):
    base = dict(
        account_username="alice",
        session_id=1,
        competition_id=2,
        level=3,
        question_id=42,
        question_text="What is 2+2?",
        options=[{"id": 0, "text": "3"}, {"id": 1, "text": "4"}],
        predicted_option_id=1,
        correct_option_id_if_known=1,
        strategy_name="zero_shot",
        model_name="dummy",
        prompt_version="v1",
        confidence=0.9,
        rationale="basic arithmetic",
        latency_ms=10,
    )
    base.update(overrides)
    return PredictionRecord(**base)


def test_record_persists_generated_answer_flag(tmp_path):
    log = QuestionLog(tmp_path / "gen.sqlite")
    log.record(_record(generated_answer=True))
    log.record(_record(question_id=43, generated_answer=False))

    con = sqlite3.connect(tmp_path / "gen.sqlite")
    rows = con.execute(
        "SELECT question_id, generated_answer FROM predictions ORDER BY question_id"
    ).fetchall()
    con.close()
    assert rows == [(42, 1), (43, 0)]


def test_migration_adds_generated_answer_to_pre_existing_db(tmp_path):
    """A DB created before generated_answer existed must gain the column
    on next QuestionLog construction, with the default 0 for old rows."""
    db_path = tmp_path / "old.sqlite"
    # Hand-create a v1 schema (no generated_answer) and seed a row.
    con = sqlite3.connect(db_path)
    con.executescript(
        """
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            account_username TEXT NOT NULL,
            session_id INTEGER NOT NULL,
            competition_id INTEGER NOT NULL,
            level INTEGER NOT NULL,
            question_id INTEGER NOT NULL,
            question_text TEXT NOT NULL,
            options_json TEXT NOT NULL,
            predicted_option_id INTEGER NOT NULL,
            correct_option_id_if_known INTEGER,
            strategy_name TEXT NOT NULL,
            model_name TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            confidence REAL,
            rationale TEXT,
            latency_ms INTEGER NOT NULL
        );
        INSERT INTO predictions (timestamp, account_username, session_id,
            competition_id, level, question_id, question_text, options_json,
            predicted_option_id, correct_option_id_if_known, strategy_name,
            model_name, prompt_version, confidence, rationale, latency_ms)
        VALUES ('2026-05-01T00:00:00Z', 'a', 1, 2, 3, 99, 'old', '[]',
                0, NULL, 's', 'm', 'v1', 0.5, 'r', 100);
        """
    )
    con.commit()
    con.close()

    # Construct QuestionLog -> migration runs.
    QuestionLog(db_path)

    con = sqlite3.connect(db_path)
    cols = {row[1] for row in con.execute("PRAGMA table_info(predictions)").fetchall()}
    assert "generated_answer" in cols
    # Old row defaults to 0.
    flag = con.execute("SELECT generated_answer FROM predictions WHERE question_id = 99").fetchone()
    con.close()
    assert flag == (0,)
