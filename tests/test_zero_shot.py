"""Unit tests for the zero-shot strategy.

We don't load a real LLM — we use a tiny duck-typed stub so the test runs
without llama-cpp-python or a 5 GB GGUF download.
"""

from __future__ import annotations

from typing import Any, cast

from polimillionaire._vendor.millionaire_client.models import Option, Question
from polimillionaire.llm import LLM
from polimillionaire.prompts._common import render_question_block
from polimillionaire.strategies._common import make_schema
from polimillionaire.strategies.base import Context
from polimillionaire.strategies.zero_shot import ZeroShotStrategy


class _FakeLLM:
    """Records the call args and returns a canned response."""

    name = "fake-model"

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.last_messages: list[dict[str, str]] | None = None
        self.last_schema: dict[str, Any] | None = None

    def complete_json(
        self, messages: list[dict[str, str]], schema: dict[str, Any], **_: Any
    ) -> dict[str, Any]:
        self.last_messages = messages
        self.last_schema = schema
        return self.response


def _make_question() -> Question:
    return Question(
        id=42,
        text="What is the capital of France?",
        options=[
            Option(id=1, text="Berlin"),
            Option(id=2, text="Madrid"),
            Option(id=3, text="Paris"),
            Option(id=4, text="Rome"),
        ],
        level=1,
    )


def test_render_question_block_uses_actual_option_ids() -> None:
    block = render_question_block(_make_question())
    assert "[1] Berlin" in block
    assert "[3] Paris" in block
    assert "Q: What is the capital of France?" in block


def test_make_schema_field_order_puts_rationale_first() -> None:
    schema = make_schema(_make_question())
    keys = list(schema["properties"].keys())
    assert keys == ["rationale", "confidence", "answer_id"]
    assert schema["properties"]["answer_id"]["enum"] == [1, 2, 3, 4]


def test_make_schema_can_drop_rationale_for_two_stage() -> None:
    schema = make_schema(_make_question(), include_rationale=False)
    assert "rationale" not in schema["properties"]
    assert "rationale" not in schema["required"]


def test_zero_shot_strategy_returns_decision_with_metadata() -> None:
    fake = _FakeLLM(
        response={
            "rationale": "Paris is the capital of France.",
            "confidence": 0.95,
            "answer_id": 3,
        }
    )
    strategy = ZeroShotStrategy(cast(LLM, fake))
    decision = strategy(_make_question(), Context(competition_id=0, level=1))

    assert decision.option_id == 3
    assert decision.confidence == 0.95
    assert decision.rationale == "Paris is the capital of France."
    assert decision.model_name == "fake-model"
    assert decision.strategy_name == "zero_shot"
    assert decision.prompt_version == "v1"
    assert decision.latency_ms >= 0


def test_zero_shot_strategy_passes_per_question_schema() -> None:
    fake = _FakeLLM(response={"rationale": "x", "confidence": 0.5, "answer_id": 1})
    strategy = ZeroShotStrategy(cast(LLM, fake))
    strategy(_make_question(), Context(competition_id=0, level=1))

    assert fake.last_schema is not None
    assert fake.last_schema["properties"]["answer_id"]["enum"] == [1, 2, 3, 4]
    assert fake.last_messages is not None
    assert fake.last_messages[0]["role"] == "system"
    assert fake.last_messages[1]["role"] == "user"
    assert "[3] Paris" in fake.last_messages[1]["content"]
