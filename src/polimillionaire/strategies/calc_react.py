"""ReAct-style strategy with a calculator tool.

Each step the model emits either `{"action":"calculate", ...}` or
`{"action":"answer", ...}`. Calculate results are appended to the message
list and the loop continues. At `max_steps` the action schema is swapped for
the answer-only schema, forcing a commit.
"""

from __future__ import annotations

from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.llm import LLM
from polimillionaire.prompts import calc_react as prompt
from polimillionaire.strategies._common import run_react_loop
from polimillionaire.strategies.base import AnswerDecision, Context


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
        messages = list(self._variant.render(question))
        return run_react_loop(
            self._llm,
            messages,
            question,
            max_steps=self._max_steps,
            model_name=self.model_name,
            strategy_name=self.strategy_name,
            prompt_version=self.prompt_version,
            verbose=self._verbose,
            log_prefix="calc-react",
        )
