"""Parametrised checks that every strategy honours the `prompt_version=` kwarg.

Constructed with the strategy's prompts module: each strategy must validate
the version against its module's PROMPTS dict, raise on unknown versions, and
fall back to the module's LATEST when no kwarg is passed.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from polimillionaire.llm import LLM
from polimillionaire.prompts import calc_react as calc_react_prompt
from polimillionaire.prompts import wiki_rag as wiki_rag_prompt
from polimillionaire.prompts import zero_shot as zero_shot_prompt
from polimillionaire.strategies.calc_react import CalcReactStrategy
from polimillionaire.strategies.wiki_rag import WikiRagStrategy
from polimillionaire.strategies.zero_shot import ZeroShotStrategy


class _StubLLM:
    name = "stub"

    def complete_json(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        **_: Any,
    ) -> dict[str, Any]:
        return {"rationale": "x", "confidence": 0.5, "answer_id": 1}


# A no-op constructor function per strategy, with the prompts module that backs it.
# wiki_rag also needs three fake retrieval deps; the validation happens before any
# of them are touched, so None placeholders are fine for the construction tests.
_CASES = [
    ("zero_shot", lambda llm, pv: ZeroShotStrategy(llm, prompt_version=pv), zero_shot_prompt),
    ("calc_react", lambda llm, pv: CalcReactStrategy(llm, prompt_version=pv), calc_react_prompt),
    (
        "wiki_rag",
        lambda llm, pv: WikiRagStrategy(llm, None, None, None, prompt_version=pv),  # type: ignore[arg-type]
        wiki_rag_prompt,
    ),
]


@pytest.mark.parametrize("name,builder,module", _CASES)
def test_default_prompt_version_matches_module_latest(name, builder, module) -> None:
    strategy = builder(cast(LLM, _StubLLM()), module.LATEST)
    assert strategy.prompt_version == module.LATEST
    assert module.LATEST in module.PROMPTS, f"{name}: LATEST must be a key in PROMPTS"


@pytest.mark.parametrize("name,builder,module", _CASES)
def test_unknown_prompt_version_raises(name, builder, module) -> None:  # noqa: ARG001
    with pytest.raises(ValueError, match="unknown prompt version"):
        builder(cast(LLM, _StubLLM()), "definitely-not-a-real-version")
