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
    """Fake LLM that returns canned responses in order, recording every call."""

    name = "fake-model"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    def complete_json(
        self, messages: list[dict[str, str]], schema: dict[str, Any], **_: Any
    ) -> dict[str, Any]:
        self.calls.append(([dict(m) for m in messages], schema))
        if not self._responses:
            raise AssertionError("script exhausted: strategy made an unexpected extra call")
        return self._responses.pop(0)


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
    strategy = CalcReactStrategy(cast(LLM, fake))
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
    strategy = CalcReactStrategy(cast(LLM, fake))
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


def test_strategy_rejects_zero_max_steps() -> None:
    fake = _ScriptedLLM([])
    with pytest.raises(ValueError, match="max_steps"):
        CalcReactStrategy(cast(LLM, fake), max_steps=0)
