"""Smoke tests: run on every commit via the pre-commit / CI gate."""

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


def test_context_extras_default_empty():
    ctx = Context(competition_id=1, level=5)
    assert ctx.extras == {}
