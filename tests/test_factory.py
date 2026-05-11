"""Tests for the strategy factory and registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from polimillionaire._vendor.millionaire_client.models import Option, Question
from polimillionaire.llm import LLM
from polimillionaire.strategies.calc_react import CalcReactStrategy
from polimillionaire.strategies.factory import available, make_strategy, register
from polimillionaire.strategies.routed import RoutedStrategy
from polimillionaire.strategies.zero_shot import ZeroShotStrategy


class _FakeLLM:
    name = "fake-model"

    def __init__(self) -> None:
        self.calls: list = []

    def complete_json(self, messages: list[dict], schema: dict, **_: Any) -> dict[str, Any]:
        self.calls.append((messages, schema))
        # return a valid answer for whatever options are in the schema
        enum = schema.get("properties", {}).get("answer_id", {}).get("enum", [1])
        return {"rationale": "test", "confidence": 0.5, "answer_id": enum[0]}


def _make_question() -> Question:
    return Question(
        id=1,
        text="What is 2+2?",
        options=[Option(id=1, text="3"), Option(id=2, text="4")],
        level=1,
    )


def _fake_llm() -> LLM:
    return cast(LLM, _FakeLLM())


def test_make_strategy_zero_shot_returns_zero_shot_strategy() -> None:
    s = make_strategy("zero_shot", _fake_llm())
    assert isinstance(s, ZeroShotStrategy)


def test_make_strategy_calc_react_returns_calc_react_strategy() -> None:
    s = make_strategy("calc_react", _fake_llm())
    assert isinstance(s, CalcReactStrategy)


def test_make_strategy_unknown_raises_key_error() -> None:
    with pytest.raises(KeyError, match="unknown strategy"):
        make_strategy("no_such_strategy", _fake_llm())


def test_make_strategy_rag_calc_react_falls_back_without_index(tmp_path: Path) -> None:
    # tmp_path has no data/index/math directory, so should fall back to CalcReactStrategy
    s = make_strategy("rag_calc_react", _fake_llm(), project_root=tmp_path)
    assert isinstance(s, CalcReactStrategy)


def test_available_returns_expected_names() -> None:
    names = available()
    assert sorted(names) == names  # must be sorted
    assert set(names) >= {"zero_shot", "calc_react", "rag_calc_react", "wiki_rag", "auto"}


def test_register_duplicate_raises_value_error() -> None:
    # use a uuid-suffixed name so re-imports of this test file under pytest
    # plugin reload don't fight over the same registry slot
    import uuid

    name = f"_test_duplicate_{uuid.uuid4().hex}"

    @register(name)
    def _builder_a(llm: LLM, **kw: Any) -> ZeroShotStrategy:
        return ZeroShotStrategy(llm)

    with pytest.raises(ValueError, match="already registered"):

        @register(name)
        def _builder_b(llm: LLM, **kw: Any) -> ZeroShotStrategy:
            return ZeroShotStrategy(llm)


def test_make_strategy_zero_shot_accepts_and_ignores_extra_kwargs() -> None:
    """ZeroShotStrategy doesn't take verbose/max_steps -- the builder must absorb them."""
    s = make_strategy("zero_shot", _fake_llm(), verbose=True, max_steps=99)
    assert isinstance(s, ZeroShotStrategy)


def test_make_strategy_wiki_rag_falls_back_to_zero_shot_without_index(tmp_path: Path) -> None:
    """Wiki index missing -> degrade to ZeroShotStrategy; extra kwargs don't blow it up."""
    s = make_strategy(
        "wiki_rag",
        _fake_llm(),
        competition_id=0,
        project_root=tmp_path,
        verbose=True,
    )
    assert isinstance(s, ZeroShotStrategy)


def test_make_strategy_auto_no_competition_returns_routed_with_four_routes(tmp_path: Path) -> None:
    # no indexes exist in tmp_path, so all routes degrade to fallbacks,
    # but we still get a RoutedStrategy with all four route keys
    s = make_strategy("auto", _fake_llm(), project_root=tmp_path)
    assert isinstance(s, RoutedStrategy)
    assert set(s.routes.keys()) == {0, 1, 2, 3}


def test_make_strategy_auto_with_competition_id_builds_only_that_route(tmp_path: Path) -> None:
    # competition_id=2 -> only key 2 in routes; others are placeholders (ZeroShotStrategy)
    s = make_strategy("auto", _fake_llm(), competition_id=2, project_root=tmp_path)
    assert isinstance(s, RoutedStrategy)
    assert 2 in s.routes
    # route 2 degrades to ZeroShotStrategy because no index exists in tmp_path
    assert isinstance(s.routes[2], ZeroShotStrategy)
    # routes 0 and 1 should not appear (they are handled by the default)
    assert 0 not in s.routes
    assert 1 not in s.routes


def test_make_strategy_auto_math_competition_builds_only_route_3(tmp_path: Path) -> None:
    s = make_strategy("auto", _fake_llm(), competition_id=3, project_root=tmp_path)
    assert isinstance(s, RoutedStrategy)
    assert set(s.routes.keys()) == {3}
    # falls back to CalcReactStrategy since there's no math index in tmp_path
    assert isinstance(s.routes[3], CalcReactStrategy)
