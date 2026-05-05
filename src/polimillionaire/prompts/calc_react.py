"""Prompt for the calculator-equipped ReAct strategy.

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
    "You are an expert trivia player with access to a calculator tool.\n"
    "\n"
    "On every turn, output JSON matching exactly one of these two shapes:\n"
    '  - {"action": "calculate", "expression": "<sympy expression>"}\n'
    '  - {"action": "answer", "rationale": "...", "confidence": <0..1>, "answer_id": <int>}\n'
    "\n"
    "Calculator syntax is sympy. Examples: sqrt(2), pi, factorial(10), 1/3, "
    "2**32, log(100, 10), Rational(355, 113). After a calculate step you will "
    "see the result and may calculate again or commit to an answer.\n"
    "\n"
    "Use the calculator whenever arithmetic, exponents, factorials, square "
    "roots, logarithms, or unit conversions are involved -- do not compute "
    "those mentally. For non-numeric questions, answer directly.\n"
    "\n"
    "When you answer, write the rationale first (no more than three sentences) "
    "and only commit to an answer_id consistent with that rationale."
)


def render(question: Question) -> list[Message]:
    """Build the initial message list for a calc-react turn."""
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": render_question_block(question)},
    ]
