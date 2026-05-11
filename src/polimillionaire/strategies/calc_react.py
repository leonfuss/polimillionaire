"""ReAct-style strategy with a calculator tool.

Each step the model emits either `{"action":"calculate", ...}` or
`{"action":"answer", ...}`. Calculate results are appended to the message
list and the loop continues. At `max_steps` the action schema is swapped for
the answer-only schema, forcing a commit.
"""

from __future__ import annotations

import json
import time

from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.llm import LLM, Message
from polimillionaire.prompts import calc_react as prompt
from polimillionaire.strategies._common import build_decision, make_action_schema, make_schema
from polimillionaire.strategies.base import AnswerDecision, Context
from polimillionaire.tools import calc


class CalcReactStrategy:
    """Single LLM, calculator tool, bounded ReAct loop.

    `max_steps` caps the total number of `complete_json` calls *before* the
    forced-answer fallback, so worst case is `max_steps + 1` calls per question.
    """

    strategy_name = "calc_react"

    def __init__(
        self,
        llm: LLM,
        *,
        max_steps: int = 3,
        verbose: bool = False,
        prompt_version: str = prompt.LATEST,
    ) -> None:
        if max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {max_steps}")
        if prompt_version not in prompt.PROMPTS:
            raise ValueError(
                f"unknown prompt version {prompt_version!r}; "
                f"available: {sorted(prompt.PROMPTS)}"
            )
        self._llm = llm
        self._max_steps = max_steps
        self._verbose = verbose
        self._variant = prompt.PROMPTS[prompt_version]

    @property
    def model_name(self) -> str:
        return self._llm.name

    @property
    def prompt_version(self) -> str:
        return self._variant.version

    def __call__(self, question: Question, ctx: Context) -> AnswerDecision:  # noqa: ARG002
        messages: list[Message] = list(self._variant.render(question))
        action_schema = make_action_schema(question)
        start = time.perf_counter()

        for _ in range(self._max_steps):
            try:
                out = self._llm.complete_json(messages, action_schema)
            except ValueError:
                # Output blew through max_tokens and didn't parse as JSON.
                # Skip remaining steps and force an answer with whatever
                # context we have so the game continues.
                if self._verbose:
                    print("   [calc-react] action step failed to parse — forcing answer")
                break
            if out["action"] == "answer":
                return self._build(out, start)
            expression = out["expression"]
            result = calc(expression)
            if self._verbose:
                print(f'   [calc-react] calc("{expression}") -> {result}')
            messages.append({"role": "assistant", "content": json.dumps(out)})
            messages.append({"role": "user", "content": f"Calculator: `{expression}` = {result}"})

        # Step cap hit OR parse failure — force a commit on the answer-only schema.
        messages.append(
            {"role": "user", "content": "Step limit reached. Answer now using the answer schema."}
        )
        try:
            out = self._llm.complete_json(messages, make_schema(question))
        except ValueError:
            # Even the forced answer didn't parse. Default to the first option
            # at zero confidence so the game submits *something* and continues.
            if self._verbose:
                print("   [calc-react] forced answer also failed — defaulting to option 0")
            return self._build(
                {
                    "rationale": "Model output failed to parse; defaulting to first option.",
                    "answer_id": question.options[0].id,
                    "confidence": 0.0,
                },
                start,
            )
        return self._build(out, start)

    def _build(self, out: dict, start: float) -> AnswerDecision:
        return build_decision(
            out,
            start,
            model_name=self.model_name,
            strategy_name=self.strategy_name,
            prompt_version=self.prompt_version,
        )
