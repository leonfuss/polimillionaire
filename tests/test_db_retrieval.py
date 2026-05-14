"""Tests for DbRetrievalStrategy + the QuestionLog lookup helpers it depends on.

The strategy is the riskier surface: it owns the drift-detection logic
that flips a sticky `index_valid` flag, and the timeout/avoidance code
that rewrites the inner strategy's answer. The tests below pin both,
plus the lookup helpers in QuestionLog so a future schema tweak can't
silently break the retrieval path.
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from polimillionaire._vendor.millionaire_client.models import Option, Question
from polimillionaire.recording import PredictionRecord, QuestionLog
from polimillionaire.strategies.base import AnswerDecision, Context
from polimillionaire.strategies.db_retrieval import DbRetrievalStrategy


def _q(qid: int = 1, text: str = "what's 2+2?") -> Question:
    return Question(
        id=qid,
        text=text,
        options=[
            Option(id=0, text="3"),
            Option(id=1, text="4"),
            Option(id=2, text="5"),
            Option(id=3, text="22"),
        ],
        level=1,
    )


def _ctx() -> Context:
    return Context(competition_id=0, level=1)


def _decision(option_id: int, **kw) -> AnswerDecision:
    return AnswerDecision(
        option_id=option_id,
        confidence=kw.get("confidence", 0.8),
        rationale=kw.get("rationale", "inner says so"),
        model_name=kw.get("model_name", "inner-llm"),
        strategy_name=kw.get("strategy_name", "inner"),
        prompt_version=kw.get("prompt_version", "v1"),
        latency_ms=kw.get("latency_ms", 100),
    )


class _FixedInner:
    strategy_name = "inner"
    model_name = "inner-llm"
    prompt_version = "v1"

    def __init__(self, decision: AnswerDecision) -> None:
        self._decision = decision
        self.calls = 0

    def __call__(self, question: Question, ctx: Context) -> AnswerDecision:  # noqa: ARG002
        self.calls += 1
        return self._decision


class _SlowInner:
    """Sleeps longer than the wrapper's timeout to exercise the timeout path."""

    strategy_name = "slow_inner"
    model_name = "inner-llm"
    prompt_version = "v1"

    def __init__(self, sleep_s: float, decision: AnswerDecision) -> None:
        self._sleep_s = sleep_s
        self._decision = decision

    def __call__(self, question: Question, ctx: Context) -> AnswerDecision:  # noqa: ARG002
        time.sleep(self._sleep_s)
        return self._decision


def _record(
    log: QuestionLog,
    *,
    qid: int,
    text: str,
    predicted: int,
    correct: int | None,
    generated: bool = False,
    session_id: int = 1,
) -> None:
    log.record(
        PredictionRecord(
            account_username="tester",
            session_id=session_id,
            competition_id=0,
            level=1,
            question_id=qid,
            question_text=text,
            options=[{"id": i, "text": str(i)} for i in range(4)],
            predicted_option_id=predicted,
            correct_option_id_if_known=correct,
            strategy_name="seed",
            model_name="seed",
            prompt_version="v0",
            confidence=None,
            rationale=None,
            latency_ms=0,
            generated_answer=generated,
        )
    )


# ---- recording helpers ----------------------------------------------------


def test_meta_get_set_roundtrip(tmp_path: Path) -> None:
    log = QuestionLog(tmp_path / "q.sqlite")
    assert log.get_meta("missing") is None
    log.set_meta("k", "v1")
    assert log.get_meta("k") == "v1"
    log.set_meta("k", "v2")  # upsert
    assert log.get_meta("k") == "v2"


def test_lookup_known_correct_ignores_generated_answers(tmp_path: Path) -> None:
    """Only server-confirmed truth (generated_answer=0) counts. A reasoned
    answer from offline replay must not influence live play."""
    log = QuestionLog(tmp_path / "q.sqlite")
    _record(log, qid=42, text="Q", predicted=1, correct=1, generated=True)
    assert log.lookup_known_correct(42, "Q") is None
    _record(log, qid=42, text="Q", predicted=1, correct=1, generated=False)
    assert log.lookup_known_correct(42, "Q") == 1


def test_lookup_failed_options_excludes_correct_pick(tmp_path: Path) -> None:
    log = QuestionLog(tmp_path / "q.sqlite")
    _record(log, qid=7, text="Q", predicted=2, correct=None)  # unconfirmed; ignored
    _record(log, qid=7, text="Q", predicted=0, correct=1)  # wrong: 0 is failed
    _record(log, qid=7, text="Q", predicted=3, correct=1)  # wrong: 3 is failed
    _record(log, qid=7, text="Q", predicted=1, correct=1)  # right: 1 is NOT failed
    assert log.lookup_failed_options(7, "Q") == {0, 3}


def test_find_text_mismatch_normalises_whitespace(tmp_path: Path) -> None:
    log = QuestionLog(tmp_path / "q.sqlite")
    _record(log, qid=5, text="What  is the  capital?", predicted=0, correct=None)
    # whitespace-only difference is not drift
    assert log.find_text_mismatch(5, "What is the capital?") is None
    # material change is drift
    assert log.find_text_mismatch(5, "Who is the prime minister?") is not None


# ---- strategy behaviour ---------------------------------------------------


def test_returns_db_known_correct_without_calling_inner(tmp_path: Path) -> None:
    log_path = tmp_path / "q.sqlite"
    log = QuestionLog(log_path)
    _record(log, qid=1, text="what's 2+2?", predicted=1, correct=1)

    inner = _FixedInner(_decision(option_id=2))  # would be wrong if invoked
    sleeps: list[float] = []
    strat = DbRetrievalStrategy(
        inner,
        str(log_path),
        sleep_min=7.0,
        sleep_max=18.0,
        sleeper=sleeps.append,
        rng=random.Random(0),
    )
    out = strat(_q(), _ctx())

    assert out.option_id == 1
    assert out.strategy_name == "db_retrieval"
    assert inner.calls == 0
    # pacing delay applied and inside the configured range
    assert len(sleeps) == 1
    assert 7.0 <= sleeps[0] <= 18.0


def test_drift_invalidates_index_and_falls_through(tmp_path: Path) -> None:
    log_path = tmp_path / "q.sqlite"
    log = QuestionLog(log_path)
    # logged text doesn't match what the question now says
    _record(log, qid=1, text="OLD TEXT", predicted=1, correct=1)

    inner = _FixedInner(_decision(option_id=2))
    sleeps: list[float] = []
    strat = DbRetrievalStrategy(inner, str(log_path), sleeper=sleeps.append, rng=random.Random(0))
    out = strat(_q(text="NEW TEXT"), _ctx())

    # no DB sleep -- we fell through to the inner LLM
    assert sleeps == []
    assert inner.calls == 1
    assert out.option_id == 2
    # flag stuck for future calls
    assert log.get_meta("index_valid") == "0"

    # second call: DB is now considered invalid; never consult it again
    inner.calls = 0
    out2 = strat(_q(text="NEW TEXT"), _ctx())
    assert inner.calls == 1
    assert out2.option_id == 2
    assert sleeps == []


def test_post_invalidation_lookup_failed_options_skipped(tmp_path: Path) -> None:
    """Once index_valid=0, we don't filter by historical failed picks either --
    those rows are by assumption from a stale question pool."""
    log_path = tmp_path / "q.sqlite"
    log = QuestionLog(log_path)
    log.set_meta("index_valid", "0")
    _record(log, qid=1, text="what's 2+2?", predicted=1, correct=2)  # would block 1

    inner = _FixedInner(_decision(option_id=1))  # picks the "blocked" option
    strat = DbRetrievalStrategy(inner, str(log_path), sleeper=lambda _s: None)
    out = strat(_q(), _ctx())
    # inner's pick is preserved (no rewrite) because the index is invalid
    assert out.option_id == 1


def test_blocks_previously_failed_option_with_random_rewrite(tmp_path: Path) -> None:
    """If the inner picks an option already known to be wrong, rewrite to a
    random remaining one. We seed failed=[0, 2] with `correct=99` -- a
    phantom id not in the option set -- so lookup_known_correct returns
    nothing and the rewrite path is exercised."""
    log_path = tmp_path / "q.sqlite"
    log = QuestionLog(log_path)
    _record(log, qid=1, text="what's 2+2?", predicted=0, correct=99)
    _record(log, qid=1, text="what's 2+2?", predicted=2, correct=99)

    inner = _FixedInner(_decision(option_id=0))  # known-wrong pick
    strat = DbRetrievalStrategy(inner, str(log_path), sleeper=lambda _s: None, rng=random.Random(1))
    out = strat(_q(), _ctx())
    assert out.option_id in {1, 3}
    assert out.option_id != 0
    # the rationale must surface the inner's blocked pick for debugging
    assert "0" in (out.rationale or "")


def test_timeout_falls_back_to_random_remaining(tmp_path: Path) -> None:
    log_path = tmp_path / "q.sqlite"
    log = QuestionLog(log_path)
    _record(log, qid=1, text="what's 2+2?", predicted=0, correct=99)  # 0 failed

    inner = _SlowInner(sleep_s=2.0, decision=_decision(option_id=1))
    strat = DbRetrievalStrategy(
        inner,
        str(log_path),
        llm_timeout_s=0.2,
        sleeper=lambda _s: None,
        rng=random.Random(42),
    )
    out = strat(_q(), _ctx())
    # timed out -- pick from {1, 2, 3}, not 0
    assert out.option_id in {1, 2, 3}
    assert out.option_id != 0
    assert out.confidence == 0.0
    assert "did not return in time" in (out.rationale or "")


def test_inner_exception_falls_back_cleanly(tmp_path: Path) -> None:
    log_path = tmp_path / "q.sqlite"
    QuestionLog(log_path)  # init schema

    class _Boom:
        strategy_name = "boom"
        model_name = "x"
        prompt_version = "v0"

        def __call__(self, q, c):  # noqa: ANN001, ARG002
            raise RuntimeError("inner exploded")

    strat = DbRetrievalStrategy(
        _Boom(), str(log_path), sleeper=lambda _s: None, rng=random.Random(0)
    )
    out = strat(_q(), _ctx())
    assert out.option_id in {0, 1, 2, 3}
    assert out.confidence == 0.0


def test_db_hit_filtered_when_option_no_longer_present(tmp_path: Path) -> None:
    """The DB has option_id 99 as correct, but the current question only has
    options 0-3. Don't return 99 -- fall through to the inner."""
    log_path = tmp_path / "q.sqlite"
    log = QuestionLog(log_path)
    _record(log, qid=1, text="what's 2+2?", predicted=99, correct=99)

    inner = _FixedInner(_decision(option_id=2))
    strat = DbRetrievalStrategy(inner, str(log_path), sleeper=lambda _s: None)
    out = strat(_q(), _ctx())
    assert out.option_id == 2
    assert inner.calls == 1


def test_all_options_failed_returns_first_option(tmp_path: Path) -> None:
    """Pathological case: every option has a confirmed-wrong record. We still
    have to send something; defaulting to option 0 is the recorded behaviour."""
    log_path = tmp_path / "q.sqlite"
    log = QuestionLog(log_path)
    for oid in (0, 1, 2, 3):
        # use a phantom 'correct' so these don't also count as known-correct
        _record(log, qid=1, text="what's 2+2?", predicted=oid, correct=99)

    inner = _FixedInner(_decision(option_id=2))
    strat = DbRetrievalStrategy(inner, str(log_path), sleeper=lambda _s: None)
    out = strat(_q(), _ctx())
    assert out.option_id == 0
    assert inner.calls == 0
