"""Zero-shot prompt template.

Bump `PROMPT_VERSION` whenever the wording changes; the version string is
written into the predictions log so we can attribute accuracy shifts to
prompt changes vs model changes.
"""

from __future__ import annotations

from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.llm import Message
from polimillionaire.strategies._common import render_question_block

PROMPT_VERSION = "v1"

SYSTEM = (
    "You are an expert trivia player. For each multiple-choice question, "
    "first write a brief rationale (no more than three sentences) weighing "
    "the options, then commit to the option_id you believe is correct. "
    "Only commit to an answer that is consistent with your rationale."
)


def render(question: Question) -> list[Message]:
    """Build the message list for a single zero-shot turn."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": render_question_block(question)},
    ]
