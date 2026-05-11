"""Tests for replay_records() and the ReplayResult type."""

from __future__ import annotations

import pytest

from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.eval.replay import _aggregate, replay_records
from polimillionaire.eval.results import ReplayResult, to_polars
from polimillionaire.recording import PredictionRecord, QuestionLog
from polimillionaire.strategies.base import AnswerDecision, Context


def _pred(**overrides) -> PredictionRecord:
    base = dict(
        account_username="alice",
        session_id=1,
        competition_id=0,
        level=1,
        question_id=1,
        question_text="What is 1+1?",
        options=[{"id": 0, "text": "1"}, {"id": 1, "text": "2"}, {"id": 2, "text": "3"}],
        predicted_option_id=1,
        correct_option_id_if_known=1,
        strategy_name="stub",
        model_name="stub-model",
        prompt_version="v0",
        confidence=None,
        rationale=None,
        latency_ms=0,
        generated_answer=False,
    )
    base.update(overrides)
    return PredictionRecord(**base)


class _FixedStrategy:
    """Always answers with a preset option_id, recording calls made."""

    def __init__(self, option_id: int) -> None:
        self._option_id = option_id
        self.calls: list[tuple[Question, Context]] = []

    def __call__(self, question: Question, ctx: Context) -> AnswerDecision:
        self.calls.append((question, ctx))
        return AnswerDecision(
            option_id=self._option_id,
            strategy_name="fixed",
            model_name="stub-model",
            prompt_version="v1",
            confidence=0.8,
            latency_ms=5,
        )


def _populate(tmp_path, records: list[PredictionRecord]):
    log = QuestionLog(tmp_path / "test.sqlite")
    for rec in records:
        log.record(rec)
    return tmp_path / "test.sqlite"


def test_replay_records_returns_one_result_per_labelled_question(tmp_path):
    db = _populate(
        tmp_path,
        [
            _pred(question_id=1, correct_option_id_if_known=1),
            _pred(question_id=2, correct_option_id_if_known=2),
        ],
    )
    strategy = _FixedStrategy(option_id=1)
    results = replay_records(strategy, db)

    assert len(results) == 2
    assert len(strategy.calls) == 2
    assert all(isinstance(r, ReplayResult) for r in results)


def test_replay_records_excludes_unlabelled_questions(tmp_path):
    db = _populate(
        tmp_path,
        [
            _pred(question_id=1, correct_option_id_if_known=1),
            _pred(question_id=2, correct_option_id_if_known=None),
        ],
    )
    strategy = _FixedStrategy(option_id=1)
    results = replay_records(strategy, db)

    assert len(results) == 1
    assert results[0].question_id == 1


def test_replay_records_correct_flag_reflects_decision(tmp_path):
    db = _populate(
        tmp_path,
        [
            _pred(question_id=1, correct_option_id_if_known=1),
            _pred(question_id=2, question_text="Other?", correct_option_id_if_known=0),
        ],
    )
    # strategy always picks 1; question 1 is right, question 2 is wrong
    strategy = _FixedStrategy(option_id=1)
    results = replay_records(strategy, db)
    by_qid = {r.question_id: r for r in results}

    assert by_qid[1].correct is True
    assert by_qid[2].correct is False


def test_replay_records_decision_metadata_propagates(tmp_path):
    db = _populate(tmp_path, [_pred(question_id=1, correct_option_id_if_known=1)])
    strategy = _FixedStrategy(option_id=1)
    results = replay_records(strategy, db)

    r = results[0]
    assert r.strategy_name == "fixed"
    assert r.model_name == "stub-model"
    assert r.prompt_version == "v1"
    assert r.confidence == 0.8
    assert r.latency_ms == 5


def test_replay_records_competition_filter(tmp_path):
    db = _populate(
        tmp_path,
        [
            _pred(question_id=1, competition_id=0, correct_option_id_if_known=1),
            _pred(
                question_id=2,
                competition_id=1,
                correct_option_id_if_known=1,
                question_text="Another?",
            ),
        ],
    )
    strategy = _FixedStrategy(option_id=1)
    results = replay_records(strategy, db, competition_id=0)

    assert len(results) == 1
    assert results[0].competition_id == 0


def test_replay_records_generated_answer_flag(tmp_path):
    db = _populate(
        tmp_path,
        [
            _pred(question_id=1, generated_answer=True, correct_option_id_if_known=1),
            _pred(
                question_id=2,
                generated_answer=False,
                correct_option_id_if_known=1,
                question_text="Another?",
            ),
        ],
    )
    strategy = _FixedStrategy(option_id=1)
    results = replay_records(strategy, db)
    by_qid = {r.question_id: r for r in results}

    assert by_qid[1].generated_answer is True
    assert by_qid[2].generated_answer is False


def test_aggregate_combined_counts(tmp_path):
    db = _populate(
        tmp_path,
        [
            _pred(question_id=1, competition_id=0, correct_option_id_if_known=1),
            _pred(
                question_id=2, competition_id=0, correct_option_id_if_known=0, question_text="Hard?"
            ),
        ],
    )
    # picks 1 always: question 1 right (correct=1), question 2 wrong (correct=0)
    strategy = _FixedStrategy(option_id=1)
    records = replay_records(strategy, db)
    summary = _aggregate(records)

    assert summary["total"] == 2
    assert summary["correct"] == 1
    assert abs(summary["accuracy"] - 0.5) < 1e-9


def test_aggregate_hand_labeled_vs_server_confirmed_split(tmp_path):
    db = _populate(
        tmp_path,
        [
            _pred(question_id=1, generated_answer=True, correct_option_id_if_known=1),
            _pred(
                question_id=2,
                generated_answer=False,
                correct_option_id_if_known=1,
                question_text="Server q?",
            ),
        ],
    )
    strategy = _FixedStrategy(option_id=1)
    records = replay_records(strategy, db)
    summary = _aggregate(records)

    assert summary["hand_labeled"]["total"] == 1
    assert summary["server_confirmed"]["total"] == 1
    assert summary["hand_labeled"]["correct"] == 1
    assert summary["server_confirmed"]["correct"] == 1


def test_aggregate_matches_old_replay_output_shape(tmp_path):
    """_aggregate must produce the same top-level keys that replay() always has."""
    from polimillionaire.eval.replay import replay

    db = _populate(
        tmp_path,
        [
            _pred(question_id=1, competition_id=0, correct_option_id_if_known=1),
            _pred(
                question_id=2, competition_id=1, correct_option_id_if_known=1, question_text="Q2?"
            ),
        ],
    )
    strategy = _FixedStrategy(option_id=1)

    records = replay_records(strategy, db)
    via_aggregate = _aggregate(records)

    strategy2 = _FixedStrategy(option_id=1)
    via_replay = replay(strategy2, db)

    assert set(via_aggregate.keys()) == set(via_replay.keys())
    assert via_aggregate["total"] == via_replay["total"]
    assert via_aggregate["correct"] == via_replay["correct"]
    assert via_aggregate["accuracy"] == via_replay["accuracy"]
    # by_competition shape must survive intact -- keyed by int competition_id,
    # each value carrying {name, correct, total, accuracy}.
    assert "by_competition" in via_aggregate
    assert set(via_aggregate["by_competition"]) == {0, 1}
    sample = via_aggregate["by_competition"][0]
    assert {"name", "correct", "total", "accuracy"} <= sample.keys()
    assert via_aggregate["by_competition"] == via_replay["by_competition"]


def test_to_polars_produces_expected_columns(tmp_path):
    pytest.importorskip("polars")

    db = _populate(
        tmp_path,
        [
            _pred(question_id=1, correct_option_id_if_known=1),
            _pred(question_id=2, correct_option_id_if_known=1, question_text="Q2?"),
        ],
    )
    strategy = _FixedStrategy(option_id=1)
    records = replay_records(strategy, db)
    df = to_polars(records)

    expected_cols = {
        "strategy_name",
        "model_name",
        "prompt_version",
        "question_id",
        "competition_id",
        "level",
        "predicted_option_id",
        "correct_option_id",
        "correct",
        "confidence",
        "latency_ms",
        "generated_answer",
    }
    assert set(df.columns) == expected_cols
    assert len(df) == 2


def test_to_polars_values_match_records(tmp_path):
    pytest.importorskip("polars")

    db = _populate(tmp_path, [_pred(question_id=7, correct_option_id_if_known=1)])
    strategy = _FixedStrategy(option_id=1)
    records = replay_records(strategy, db)
    df = to_polars(records)

    row = df.row(0, named=True)
    assert row["question_id"] == 7
    assert row["correct"] is True
    assert row["strategy_name"] == "fixed"
