"""Tests for RoutedStrategy."""

from __future__ import annotations

from typing import Any, cast

from polimillionaire._vendor.millionaire_client.models import Option, Question
from polimillionaire.llm import LLM
from polimillionaire.strategies.base import AnswerDecision, Context
from polimillionaire.strategies.routed import RoutedStrategy
from polimillionaire.strategies.zero_shot import ZeroShotStrategy


class _FakeLLM:
    name = "fake-model"

    def __init__(self, answer_id: int = 1) -> None:
        self._answer_id = answer_id

    def complete_json(self, messages: list[dict], schema: dict, **_: Any) -> dict[str, Any]:
        return {"rationale": "test", "confidence": 0.7, "answer_id": self._answer_id}


class _FixedStrategy:
    """Returns a predetermined AnswerDecision regardless of input."""

    def __init__(self, decision: AnswerDecision) -> None:
        self._decision = decision
        self.strategy_name = decision.strategy_name
        self.model_name = decision.model_name
        self.prompt_version = decision.prompt_version

    def __call__(self, question: Any, ctx: Context) -> AnswerDecision:
        return self._decision


def _make_question() -> Question:
    return Question(
        id=1,
        text="What is the capital of France?",
        options=[Option(id=1, text="Berlin"), Option(id=2, text="Paris")],
        level=1,
    )


def _decision(option_id: int, strategy_name: str = "zero_shot") -> AnswerDecision:
    return AnswerDecision(
        option_id=option_id,
        confidence=0.9,
        rationale="test",
        model_name="fake-model",
        strategy_name=strategy_name,
        prompt_version="v1",
        latency_ms=0,
    )


def test_dispatches_to_correct_route() -> None:
    route0 = _FixedStrategy(_decision(option_id=1, strategy_name="wiki_rag"))
    route3 = _FixedStrategy(_decision(option_id=2, strategy_name="rag_calc_react"))
    default = _FixedStrategy(_decision(option_id=1, strategy_name="zero_shot"))

    routed = RoutedStrategy(routes={0: route0, 3: route3}, default=default)

    result = routed(_make_question(), Context(competition_id=0, level=1))
    assert result.option_id == 1
    assert result.strategy_name == "wiki_rag"

    result = routed(_make_question(), Context(competition_id=3, level=1))
    assert result.option_id == 2
    assert result.strategy_name == "rag_calc_react"


def test_falls_back_to_default_for_unrouted_competition() -> None:
    default = _FixedStrategy(_decision(option_id=1, strategy_name="zero_shot"))
    routed = RoutedStrategy(routes={3: _FixedStrategy(_decision(2, "calc_react"))}, default=default)

    # competition_id=1 is not in routes -> should hit default
    result = routed(_make_question(), Context(competition_id=1, level=1))
    assert result.strategy_name == "zero_shot"


def test_returns_sub_strategy_decision_unchanged() -> None:
    """RoutedStrategy must not relabel the decision as 'routed'."""
    inner_decision = _decision(option_id=2, strategy_name="wiki_rag")
    route = _FixedStrategy(inner_decision)
    default = ZeroShotStrategy(cast(LLM, _FakeLLM()))
    routed = RoutedStrategy(routes={0: route}, default=default)

    result = routed(_make_question(), Context(competition_id=0, level=1))
    assert result.strategy_name == "wiki_rag"
    assert result is inner_decision  # exact same object, not a copy


def test_strategy_name_attribute_is_routed() -> None:
    default = ZeroShotStrategy(cast(LLM, _FakeLLM()))
    routed = RoutedStrategy(routes={}, default=default)
    assert routed.strategy_name == "routed"


def test_model_name_and_prompt_version_are_routed_sentinel() -> None:
    default = ZeroShotStrategy(cast(LLM, _FakeLLM()))
    routed = RoutedStrategy(routes={}, default=default)
    assert routed.model_name == "routed"
    assert routed.prompt_version == "routed"


def test_routes_property_exposes_internal_dict() -> None:
    route = _FixedStrategy(_decision(1))
    routed = RoutedStrategy(routes={0: route}, default=route)
    assert routed.routes == {0: route}
