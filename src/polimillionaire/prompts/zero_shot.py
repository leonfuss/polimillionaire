"""Zero-shot prompt variants."""

from __future__ import annotations

from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.llm import Message
from polimillionaire.prompts._common import PromptVariant, render_question_block

_V1_SYSTEM = (
    "You are an expert trivia player. For each multiple-choice question, "
    "first write a brief rationale (no more than three sentences) weighing "
    "the options, then commit to the option_id you believe is correct. "
    "Only commit to an answer that is consistent with your rationale."
)


def _render_v1(question: Question) -> list[Message]:
    return [
        {"role": "system", "content": _V1_SYSTEM},
        {"role": "user", "content": render_question_block(question)},
    ]


PROMPTS: dict[str, PromptVariant] = {
    "v1": PromptVariant(version="v1", render=_render_v1),
}

LATEST = "v1"

# legacy module-level aliases so existing callers that do `prompt.render(...)` keep working
PROMPT_VERSION = LATEST
SYSTEM = _V1_SYSTEM
render = PROMPTS[LATEST].render
