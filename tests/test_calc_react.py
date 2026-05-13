"""Unit tests for `CalcReactStrategy`.

We script the FakeLLM with a queue of canned responses; each `complete_json`
call pops the next one. This lets us simulate direct-answer, calc-then-answer,
error-then-retry, and step-cap-fallback flows without loading llama-cpp.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from polimillionaire._vendor.millionaire_client.models import Option, Question
from polimillionaire.llm import LLM
from polimillionaire.strategies._common import make_action_schema
from polimillionaire.strategies.base import Context
from polimillionaire.strategies.calc_react import CalcReactStrategy


class _ScriptedLLM:
    """Fake LLM that returns canned responses in order, recording every call.

    A response of `ValueError(...)` is *raised* instead of returned, so we can
    simulate `complete_json` failing to parse the model's output.
    """

    name = "fake-model"

    def __init__(self, responses: list[dict[str, Any] | ValueError]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def complete_json(
        self, messages: list[dict[str, str]], schema: dict[str, Any], **_: Any
    ) -> dict[str, Any]:
        self.calls.append(([dict(m) for m in messages], schema))
        if not self._responses:
            raise AssertionError("script exhausted: strategy made an unexpected extra call")
        head = self._responses.pop(0)
        if isinstance(head, ValueError):
            raise head
        return head


def _make_question() -> Question:
    return Question(
        id=99,
        text="What is 12 * 17?",
        options=[
            Option(id=1, text="184"),
            Option(id=2, text="204"),
            Option(id=3, text="214"),
            Option(id=4, text="224"),
        ],
        level=2,
    )


def test_make_action_schema_has_two_branches() -> None:
    schema = make_action_schema(_make_question())
    assert "oneOf" in schema
    assert len(schema["oneOf"]) == 2
    branches = {b["properties"]["action"]["const"]: b for b in schema["oneOf"]}
    assert set(branches) == {"calculate", "answer"}
    # Answer branch must restrict answer_id to this question's option ids.
    assert branches["answer"]["properties"]["answer_id"]["enum"] == [1, 2, 3, 4]
    # Answer branch field order: rationale, confidence, answer_id (rationale-first CoT).
    answer_keys = list(branches["answer"]["properties"].keys())
    assert answer_keys == ["action", "rationale", "confidence", "answer_id"]


def test_strategy_answers_directly_without_calling_calculator() -> None:
    fake = _ScriptedLLM(
        [
            {
                "action": "answer",
                "rationale": "obvious",
                "confidence": 0.9,
                "answer_id": 2,
            }
        ]
    )
    strategy = CalcReactStrategy(cast(LLM, fake))
    decision = strategy(_make_question(), Context(competition_id=0, level=2))

    assert len(fake.calls) == 1
    assert decision.option_id == 2
    assert decision.confidence == 0.9
    assert decision.strategy_name == "calc_react"
    assert decision.model_name == "fake-model"


def test_strategy_uses_calculator_then_answers() -> None:
    fake = _ScriptedLLM(
        [
            {"action": "calculate", "expression": "12 * 17"},
            {
                "action": "answer",
                "rationale": "computed it",
                "confidence": 0.99,
                "answer_id": 2,
            },
        ]
    )
    # max_steps=2 pinned: this test exercises one calc + one answer; the
    # production default (max_steps=1) is tested separately by the step-cap
    # test below.
    strategy = CalcReactStrategy(cast(LLM, fake), max_steps=2)
    decision = strategy(_make_question(), Context(competition_id=0, level=2))

    assert len(fake.calls) == 2
    assert decision.option_id == 2

    # Second call must include the calc tool turn in history.
    second_messages = fake.calls[1][0]
    tool_turn = second_messages[-1]
    assert tool_turn["role"] == "user"
    assert "12 * 17" in tool_turn["content"]
    assert "204" in tool_turn["content"]


def test_strategy_recovers_from_calculator_error() -> None:
    fake = _ScriptedLLM(
        [
            {"action": "calculate", "expression": "this is not math"},
            {"action": "calculate", "expression": "12 * 17"},
            {
                "action": "answer",
                "rationale": "ok now",
                "confidence": 0.8,
                "answer_id": 2,
            },
        ]
    )
    # max_steps=3 pinned: this test exercises the retry-after-calc-error
    # path, which needs at least two calc steps + one answer. The production
    # default (max_steps=1) is tested separately by the step-cap test below.
    strategy = CalcReactStrategy(cast(LLM, fake), max_steps=3)
    decision = strategy(_make_question(), Context(competition_id=0, level=2))

    assert len(fake.calls) == 3
    assert decision.option_id == 2

    # First retry must have surfaced the ERROR string back to the model.
    second_messages = fake.calls[1][0]
    assert "ERROR" in second_messages[-1]["content"]


def test_strategy_force_answers_after_step_cap() -> None:
    fake = _ScriptedLLM(
        [
            {"action": "calculate", "expression": "1 + 1"},
            {"action": "calculate", "expression": "1 + 1"},
            # Forced-answer call uses the answer-only schema (no `action` key).
            {"rationale": "fine", "confidence": 0.5, "answer_id": 3},
        ]
    )
    strategy = CalcReactStrategy(cast(LLM, fake), max_steps=2)
    decision = strategy(_make_question(), Context(competition_id=0, level=2))

    assert len(fake.calls) == 3
    assert decision.option_id == 3

    forced_schema = fake.calls[-1][1]
    assert "oneOf" not in forced_schema
    assert forced_schema["properties"]["answer_id"]["enum"] == [1, 2, 3, 4]
    forced_messages = fake.calls[-1][0]
    assert "Step limit reached" in forced_messages[-1]["content"]


def test_strategy_short_circuits_on_duplicate_calc_expression() -> None:
    """If the model emits the SAME calc expression two steps in a row (a
    pattern we've seen when it ignores an ERROR result), abandon the action
    loop and force an answer rather than burning the remaining budget on
    the same broken expression."""
    fake = _ScriptedLLM(
        [
            {"action": "calculate", "expression": "broken_thing"},
            # Same expression again -- model ignored the previous result.
            {"action": "calculate", "expression": "broken_thing"},
            # Forced-answer call (answer-only schema, no `action` key).
            {"rationale": "fallback", "confidence": 0.5, "answer_id": 2},
        ]
    )
    # max_steps=4 gives plenty of room; the short-circuit should fire on the
    # duplicate at step 2 and skip directly to the forced-answer call.
    strategy = CalcReactStrategy(cast(LLM, fake), max_steps=4)
    decision = strategy(_make_question(), Context(competition_id=0, level=2))

    # Three calls total: the two duplicate action steps and one forced answer.
    # If duplicate detection didn't fire we'd see four action calls.
    assert len(fake.calls) == 3
    assert decision.option_id == 2

    forced_messages = fake.calls[-1][0]
    assert "Step limit reached" in forced_messages[-1]["content"]


def test_strategy_rejects_zero_max_steps() -> None:
    fake = _ScriptedLLM([])
    with pytest.raises(ValueError, match="max_steps"):
        CalcReactStrategy(cast(LLM, fake), max_steps=0)


def test_strategy_recovers_when_action_step_overflows_max_tokens() -> None:
    """Regression for the i^259 sum crash: when an action-step output blows
    through max_tokens and complete_json raises ValueError, the strategy must
    skip remaining steps and force an answer rather than killing the game."""
    fake = _ScriptedLLM(
        [
            ValueError("grammar-constrained output did not parse: ..."),
            # Forced-answer call (answer-only schema) succeeds.
            {"rationale": "fallback", "confidence": 0.3, "answer_id": 1},
        ]
    )
    strategy = CalcReactStrategy(cast(LLM, fake))
    decision = strategy(_make_question(), Context(competition_id=0, level=2))

    assert len(fake.calls) == 2
    assert decision.option_id == 1

    # Forced-answer call must use the answer-only schema.
    forced_schema = fake.calls[-1][1]
    assert "oneOf" not in forced_schema


def test_math_tir_prompt_variant_renders_with_specialist_system_message() -> None:
    """The math-tir prompt swaps the generalist 'trivia player' system message
    for an explicit math-specialist framing, but reuses the same JSON action
    schema and exemplars so the strategy machinery is unchanged.
    """
    from polimillionaire.prompts import calc_react as prompt

    assert "math-tir" in prompt.PROMPTS
    fake = _ScriptedLLM(
        [
            {"action": "answer", "rationale": "x", "confidence": 0.9, "answer_id": 2},
        ]
    )
    strategy = CalcReactStrategy(cast(LLM, fake), prompt_version="math-tir")
    decision = strategy(_make_question(), Context(competition_id=3, level=2))

    # The system message of the first LLM call should be the math-specialist
    # one, not the generalist v2 one. v2 opens with "expert trivia player";
    # math-tir opens with "math specialist".
    system_msg = fake.calls[0][0][0]
    assert system_msg["role"] == "system"
    assert "math specialist" in system_msg["content"]
    assert "trivia player" not in system_msg["content"]
    assert decision.prompt_version == "math-tir"
    assert decision.option_id == 2


def test_strategy_falls_back_to_first_option_when_forced_answer_also_fails() -> None:
    """Last-resort: if even the answer-only schema fails to parse, return a
    valid AnswerDecision pointing at the first option (confidence 0) so the
    game submits *something* and continues to the next question."""
    fake = _ScriptedLLM(
        [
            ValueError("grammar-constrained output did not parse: ..."),
            ValueError("grammar-constrained output did not parse: ..."),
        ]
    )
    strategy = CalcReactStrategy(cast(LLM, fake))
    question = _make_question()
    decision = strategy(question, Context(competition_id=0, level=2))

    assert decision.option_id == question.options[0].id
    assert decision.confidence == 0.0
    assert decision.rationale and "fail" in decision.rationale.lower()
