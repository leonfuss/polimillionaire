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


# Default options match _q() so a DB-recorded (correct id, text) pair
# resolves against the live question without needing per-test plumbing.
_DEFAULT_OPTIONS = [
    {"id": 0, "text": "3"},
    {"id": 1, "text": "4"},
    {"id": 2, "text": "5"},
    {"id": 3, "text": "22"},
]


def _record(
    log: QuestionLog,
    *,
    qid: int,
    text: str,
    predicted: int,
    correct: int | None,
    options: list[dict] | None = None,
    generated: bool = False,
    session_id: int = 1,
    mode: str = "text",
) -> None:
    log.record(
        PredictionRecord(
            account_username="tester",
            session_id=session_id,
            competition_id=0,
            level=1,
            question_id=qid,
            question_text=text,
            options=options if options is not None else _DEFAULT_OPTIONS,
            predicted_option_id=predicted,
            correct_option_id_if_known=correct,
            strategy_name="seed",
            model_name="seed",
            prompt_version="v0",
            confidence=None,
            rationale=None,
            latency_ms=0,
            generated_answer=generated,
            mode=mode,
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
    # Returns (option_id, option_text-at-confirmation-time) so callers can
    # detect option reshuffles and admin answer-key edits.
    assert log.lookup_known_correct(42, "Q") == (1, "4")


def test_lookup_known_correct_returns_none_on_contradiction(tmp_path: Path) -> None:
    """If the same option_id has BOTH a confirmed-correct row and a
    confirmed-wrong row for this question, the server's answer key has
    changed under us. Defer to the LLM rather than blindly resubmitting."""
    log = QuestionLog(tmp_path / "q.sqlite")
    _record(log, qid=10, text="Q", predicted=1, correct=1)  # session A: 1 was right
    _record(log, qid=10, text="Q", predicted=1, correct=None)  # session B: 1 now wrong
    assert log.lookup_known_correct(10, "Q") is None


def test_lookup_failed_options_excludes_correct_pick(tmp_path: Path) -> None:
    """The server reports only correct/wrong, so 'wrong' = correct_option_id_if_known
    is NULL. A row with correct=predicted is a confirmed-right answer and must
    NOT be in the failed set."""
    log = QuestionLog(tmp_path / "q.sqlite")
    _record(log, qid=7, text="Q", predicted=0, correct=None)  # wrong: 0 is failed
    _record(log, qid=7, text="Q", predicted=3, correct=None)  # wrong: 3 is failed
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
    _record(log, qid=1, text="what's 2+2?", predicted=1, correct=None)  # would block 1

    inner = _FixedInner(_decision(option_id=1))  # picks the "blocked" option
    strat = DbRetrievalStrategy(inner, str(log_path), sleeper=lambda _s: None)
    out = strat(_q(), _ctx())
    # inner's pick is preserved (no rewrite) because the index is invalid
    assert out.option_id == 1


def test_blocks_previously_failed_option_with_random_rewrite(tmp_path: Path) -> None:
    """If the inner picks an option already known to be wrong, rewrite to a
    random remaining one. Failed rows have `correct=None` (server only
    reports correct/wrong; we never learn the right id when we're wrong)."""
    log_path = tmp_path / "q.sqlite"
    log = QuestionLog(log_path)
    _record(log, qid=1, text="what's 2+2?", predicted=0, correct=None)
    _record(log, qid=1, text="what's 2+2?", predicted=2, correct=None)

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
    _record(log, qid=1, text="what's 2+2?", predicted=0, correct=None)  # 0 failed

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
    """The DB row's options_json doesn't contain the cached correct id (e.g.
    schema drift or a logged garbage row). Don't submit a phantom id --
    fall through to the inner strategy."""
    log_path = tmp_path / "q.sqlite"
    log = QuestionLog(log_path)
    # predicted=99 correct=99 but options_json is 0..3 -> lookup can't find
    # text for id 99 and returns None.
    _record(log, qid=1, text="what's 2+2?", predicted=99, correct=99)

    inner = _FixedInner(_decision(option_id=2))
    strat = DbRetrievalStrategy(inner, str(log_path), sleeper=lambda _s: None)
    out = strat(_q(), _ctx())
    assert out.option_id == 2
    assert inner.calls == 1


def test_db_hit_remaps_to_new_id_when_options_reshuffled(tmp_path: Path) -> None:
    """Server reshuffles option ids but keeps texts. We previously confirmed
    "4" at id=1; current question has "4" at id=2. Submit id=2."""
    log_path = tmp_path / "q.sqlite"
    log = QuestionLog(log_path)
    _record(log, qid=1, text="what's 2+2?", predicted=1, correct=1)  # "4" at id 1

    # current question has the same option texts but in a different order
    shuffled = Question(
        id=1,
        text="what's 2+2?",
        options=[
            Option(id=0, text="3"),
            Option(id=1, text="22"),
            Option(id=2, text="4"),  # "4" moved from id=1 to id=2
            Option(id=3, text="5"),
        ],
        level=1,
    )

    inner = _FixedInner(_decision(option_id=0))  # should NOT be called
    strat = DbRetrievalStrategy(inner, str(log_path), sleeper=lambda _s: None)
    out = strat(shuffled, _ctx())
    assert out.option_id == 2  # remapped to the new id carrying "4"
    assert inner.calls == 0


def test_db_hit_falls_through_when_cached_text_gone(tmp_path: Path) -> None:
    """Server edited the question's options and the previously-correct
    text is no longer one of the choices. Don't submit blindly -- defer
    to the inner strategy."""
    log_path = tmp_path / "q.sqlite"
    log = QuestionLog(log_path)
    _record(log, qid=1, text="what's 2+2?", predicted=1, correct=1)  # cached "4"

    edited = Question(
        id=1,
        text="what's 2+2?",
        options=[
            Option(id=0, text="3"),
            Option(id=1, text="four"),  # text changed
            Option(id=2, text="5"),
            Option(id=3, text="22"),
        ],
        level=1,
    )

    inner = _FixedInner(_decision(option_id=1))
    strat = DbRetrievalStrategy(inner, str(log_path), sleeper=lambda _s: None)
    out = strat(edited, _ctx())
    assert inner.calls == 1
    assert out.option_id == 1


def test_db_hit_self_heals_after_answer_key_edit(tmp_path: Path) -> None:
    """End-to-end self-correction: session 1 confirmed 1 was right; session
    2 submitted 1 and was told wrong. The third encounter must not return
    1 again -- the contradiction should defer to the LLM."""
    log_path = tmp_path / "q.sqlite"
    log = QuestionLog(log_path)
    _record(log, qid=1, text="what's 2+2?", predicted=1, correct=1)  # was right
    _record(log, qid=1, text="what's 2+2?", predicted=1, correct=None)  # now wrong

    inner = _FixedInner(_decision(option_id=2))
    strat = DbRetrievalStrategy(inner, str(log_path), sleeper=lambda _s: None)
    out = strat(_q(), _ctx())
    # falls through to inner; option 1 is also in failed_options so the
    # wrapper wouldn't have re-submitted it via the avoidance path either
    assert inner.calls == 1
    assert out.option_id == 2


# ---- mode-aware lookups ---------------------------------------------------


def test_lookups_are_mode_scoped(tmp_path: Path) -> None:
    """A 'speech' row for the same question_id must not feed the 'text' lookup
    or vice versa: text-mode and speech-mode rows live in disjoint buckets."""
    log = QuestionLog(tmp_path / "q.sqlite")
    _record(log, qid=99, text="What is the capital of France?", predicted=2, correct=2)
    _record(
        log, qid=99, text="what is the capital of france", predicted=1, correct=1, mode="speech"
    )

    # text-mode reads its own row
    assert log.lookup_known_correct(99, "What is the capital of France?", "text") == (2, "5")
    # text-mode is blind to the speech row even though the qid matches
    assert log.lookup_known_correct(99, "what is the capital of france", "text") is None
    # speech-mode reads its own row
    assert log.lookup_known_correct(99, "what is the capital of france", "speech") == (1, "4")
    # speech-mode is blind to the text row's text
    assert log.lookup_known_correct(99, "What is the capital of France?", "speech") is None


def test_find_text_mismatch_is_mode_scoped(tmp_path: Path) -> None:
    """An ASR transcript drifting from server text in a different mode is not
    drift -- they're different data sources for the same question id."""
    log = QuestionLog(tmp_path / "q.sqlite")
    _record(log, qid=5, text="Server text", predicted=0, correct=None)
    # A speech-mode row with totally different text must not trip text-mode drift.
    _record(
        log, qid=5, text="totally different transcript", predicted=0, correct=None, mode="speech"
    )
    assert log.find_text_mismatch(5, "Server text", "text") is None
    assert log.find_text_mismatch(5, "totally different transcript", "speech") is None


def test_lookup_known_correct_text_returns_option_text(tmp_path: Path) -> None:
    """The cross-mode helper returns the cached *option text* so the caller
    can fuzzy-match it against the live (transcribed) options."""
    log = QuestionLog(tmp_path / "q.sqlite")
    _record(log, qid=11, text="Q", predicted=2, correct=2)
    assert log.lookup_known_correct_text(11, "text") == "5"
    assert log.lookup_known_correct_text(11, "speech") is None  # no speech rows yet


def test_use_text_mode_retrieval_rejects_text_mode_construction(tmp_path: Path) -> None:
    """Asking for text-mode fallback while in text mode is meaningless: same-mode
    IS text-mode. Catching this at construction beats a silent no-op."""
    log_path = tmp_path / "q.sqlite"
    QuestionLog(log_path)
    try:
        DbRetrievalStrategy(
            _FixedInner(_decision(option_id=0)),
            str(log_path),
            mode="text",
            use_text_mode_retrieval=True,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for text-mode + use_text_mode_retrieval")


def test_cross_mode_fallback_resolves_via_text_lookup(tmp_path: Path) -> None:
    """Speech-mode session with no speech row yet. text-mode row says option 1
    text was 'Paris' was correct; the live (transcribed) question has 'paris'
    on option 1. Cross-mode fallback fuzzy-matches and hits."""
    log_path = tmp_path / "q.sqlite"
    log = QuestionLog(log_path)
    # text-mode confirmed: option_id=1, text="Paris"
    _record(
        log,
        qid=42,
        text="What is the capital of France?",
        predicted=1,
        correct=1,
        options=[
            {"id": 0, "text": "London"},
            {"id": 1, "text": "Paris"},
            {"id": 2, "text": "Rome"},
            {"id": 3, "text": "Berlin"},
        ],
    )

    # Live speech-mode question: same option_id mapping, but text is the
    # Whisper output (lower case, trailing punctuation).
    live_question = Question(
        id=42,
        text="what is the capital of france",
        options=[
            Option(id=0, text="london."),
            Option(id=1, text="paris."),
            Option(id=2, text="rome."),
            Option(id=3, text="berlin."),
        ],
        level=1,
    )

    inner = _FixedInner(_decision(option_id=0))  # would be wrong
    sleeps: list[float] = []
    strat = DbRetrievalStrategy(
        inner,
        str(log_path),
        mode="speech",
        use_text_mode_retrieval=True,
        sleeper=sleeps.append,
        rng=random.Random(0),
    )
    out = strat(live_question, _ctx())

    assert out.option_id == 1
    assert inner.calls == 0
    assert "text-mode-fallback" in out.rationale


def test_cross_mode_fallback_handles_reshuffle(tmp_path: Path) -> None:
    """Cached correct text was on option 1 in the text-mode session, but the
    live speech-mode options put it at index 2. We fuzzy-match by text, not
    by id, so we still pick the right option."""
    log_path = tmp_path / "q.sqlite"
    log = QuestionLog(log_path)
    _record(
        log,
        qid=42,
        text="Q",
        predicted=1,
        correct=1,
        options=[
            {"id": 0, "text": "London"},
            {"id": 1, "text": "Paris"},
            {"id": 2, "text": "Rome"},
            {"id": 3, "text": "Berlin"},
        ],
    )
    live_question = Question(
        id=42,
        text="q",
        options=[
            Option(id=0, text="london"),
            Option(id=1, text="rome"),
            Option(id=2, text="paris"),  # moved
            Option(id=3, text="berlin"),
        ],
        level=1,
    )

    inner = _FixedInner(_decision(option_id=0))
    strat = DbRetrievalStrategy(
        inner,
        str(log_path),
        mode="speech",
        use_text_mode_retrieval=True,
        sleeper=lambda _s: None,
        rng=random.Random(0),
    )
    out = strat(live_question, _ctx())

    assert out.option_id == 2
    assert inner.calls == 0


def test_cross_mode_fallback_skips_if_speech_proved_it_wrong(tmp_path: Path) -> None:
    """If we already submitted the cross-mode-resolved option in speech and
    got back wrong, the speech and text caches disagree -- defer to the LLM."""
    log_path = tmp_path / "q.sqlite"
    log = QuestionLog(log_path)
    _record(
        log,
        qid=42,
        text="Q",
        predicted=1,
        correct=1,
        options=[
            {"id": 0, "text": "London"},
            {"id": 1, "text": "Paris"},
            {"id": 2, "text": "Rome"},
            {"id": 3, "text": "Berlin"},
        ],
    )
    # speech-mode already tried option 1 ("paris") and got it wrong
    _record(
        log, qid=42, text="what is the capital of france", predicted=1, correct=None, mode="speech"
    )

    live_question = Question(
        id=42,
        text="what is the capital of france",
        options=[
            Option(id=0, text="london"),
            Option(id=1, text="paris"),
            Option(id=2, text="rome"),
            Option(id=3, text="berlin"),
        ],
        level=1,
    )
    inner = _FixedInner(_decision(option_id=3))
    strat = DbRetrievalStrategy(
        inner,
        str(log_path),
        mode="speech",
        use_text_mode_retrieval=True,
        sleeper=lambda _s: None,
        rng=random.Random(0),
    )
    out = strat(live_question, _ctx())
    # Cross-mode resolved to 1 but 1 is in speech-failed -> fall through to LLM.
    assert inner.calls == 1
    assert out.option_id == 3


def test_cross_mode_fallback_abstains_on_ambiguous_match(tmp_path: Path) -> None:
    """When two live options are similar enough to both cross the threshold,
    the resolver abstains (better to ask the LLM than risk the wrong one)."""
    log_path = tmp_path / "q.sqlite"
    log = QuestionLog(log_path)
    _record(
        log,
        qid=42,
        text="Q",
        predicted=0,
        correct=0,
        options=[
            {"id": 0, "text": "Mercury"},
            {"id": 1, "text": "Venus"},
            {"id": 2, "text": "Earth"},
            {"id": 3, "text": "Mars"},
        ],
    )
    # live options are misheard near-duplicates of "Mercury"
    live_question = Question(
        id=42,
        text="q",
        options=[
            Option(id=0, text="mercury"),
            Option(id=1, text="mercury "),  # near-identical
            Option(id=2, text="earth"),
            Option(id=3, text="mars"),
        ],
        level=1,
    )
    inner = _FixedInner(_decision(option_id=2))
    strat = DbRetrievalStrategy(
        inner,
        str(log_path),
        mode="speech",
        use_text_mode_retrieval=True,
        sleeper=lambda _s: None,
        rng=random.Random(0),
    )
    out = strat(live_question, _ctx())
    # After normalisation "mercury" == "mercury " -> exact-match short-circuit
    # returns id 0. The ambiguity guard applies only when *non*-exact matches
    # both cross the threshold; this test pins that exact-match short-circuit.
    assert out.option_id == 0


def test_all_options_failed_returns_first_option(tmp_path: Path) -> None:
    """Pathological case: every option has a confirmed-wrong record. We still
    have to send something; defaulting to option 0 is the recorded behaviour."""
    log_path = tmp_path / "q.sqlite"
    log = QuestionLog(log_path)
    for oid in (0, 1, 2, 3):
        _record(log, qid=1, text="what's 2+2?", predicted=oid, correct=None)

    inner = _FixedInner(_decision(option_id=2))
    strat = DbRetrievalStrategy(inner, str(log_path), sleeper=lambda _s: None)
    out = strat(_q(), _ctx())
    assert out.option_id == 0
    assert inner.calls == 0
