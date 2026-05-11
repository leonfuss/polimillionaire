"""Unit tests for `run_react_loop`.

The helper owns the bounded calculate-or-answer loop shared by all ReAct
strategies. These tests exercise it in isolation using the same _ScriptedLLM
pattern as test_calc_react.py.
"""

from __future__ import annotations

from typing import Any, cast

from polimillionaire._vendor.millionaire_client.models import Option, Question
from polimillionaire.llm import LLM
from polimillionaire.strategies._common import run_react_loop


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
        id=42,
        text="What is 6 * 7?",
        options=[
            Option(id=1, text="36"),
            Option(id=2, text="42"),
            Option(id=3, text="48"),
            Option(id=4, text="54"),
        ],
        level=1,
    )


def _run(llm: LLM, question: Question | None = None, *, max_steps: int = 3) -> Any:
    if question is None:
        question = _make_question()
    # minimal initial message list -- the helper doesn't inspect content
    initial = [{"role": "user", "content": question.text}]
    return run_react_loop(
        llm,
        initial,
        question,
        max_steps=max_steps,
        model_name="fake-model",
        strategy_name="test_react",
        prompt_version="v1",
    )


def test_immediate_answer_returns_decision() -> None:
    fake = _ScriptedLLM(
        [
            {
                "action": "answer",
                "rationale": "obvious",
                "confidence": 0.95,
                "answer_id": 2,
            }
        ]
    )
    decision = _run(cast(LLM, fake))

    assert len(fake.calls) == 1
    assert decision.option_id == 2
    assert decision.confidence == 0.95
    assert decision.strategy_name == "test_react"
    assert decision.model_name == "fake-model"
    assert decision.prompt_version == "v1"


def test_calculate_then_answer_feeds_result_back() -> None:
    fake = _ScriptedLLM(
        [
            {"action": "calculate", "expression": "6 * 7"},
            {
                "action": "answer",
                "rationale": "computed it",
                "confidence": 0.99,
                "answer_id": 2,
            },
        ]
    )
    decision = _run(cast(LLM, fake))

    assert len(fake.calls) == 2
    assert decision.option_id == 2

    # second call must include the calculator result in message history
    second_messages = fake.calls[1][0]
    tool_turn = second_messages[-1]
    assert tool_turn["role"] == "user"
    assert "6 * 7" in tool_turn["content"]
    assert "42" in tool_turn["content"]


def test_step_cap_triggers_forced_answer() -> None:
    fake = _ScriptedLLM(
        [
            {"action": "calculate", "expression": "1 + 1"},
            # forced-answer call uses answer-only schema (no `action` key)
            {"rationale": "giving up", "confidence": 0.4, "answer_id": 3},
        ]
    )
    decision = _run(cast(LLM, fake), max_steps=1)

    assert len(fake.calls) == 2
    assert decision.option_id == 3

    # forced-answer schema must not have oneOf
    forced_schema = fake.calls[-1][1]
    assert "oneOf" not in forced_schema
    assert forced_schema["properties"]["answer_id"]["enum"] == [1, 2, 3, 4]

    # step-limit message must be present
    forced_messages = fake.calls[-1][0]
    assert "Step limit reached" in forced_messages[-1]["content"]


def test_double_parse_failure_defaults_to_first_option() -> None:
    """last-resort: if both the action step and the forced answer fail to
    parse, return a valid AnswerDecision pointing at option 0 (confidence 0)
    so the game continues."""
    fake = _ScriptedLLM(
        [
            ValueError("grammar-constrained output did not parse: ..."),
            ValueError("grammar-constrained output did not parse: ..."),
        ]
    )
    question = _make_question()
    decision = _run(cast(LLM, fake), question)

    assert decision.option_id == question.options[0].id
    assert decision.confidence == 0.0
    assert decision.rationale is not None
    assert "fail" in decision.rationale.lower()
