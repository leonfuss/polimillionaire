"""Helpers shared by every strategy.

`render_question_block` and `make_schema` are designed to be reused by both
the current single-stage `ZeroShotStrategy` and a future two-stage
reason-then-extract strategy, so prompt formatting and the per-question JSON
schema only live in one place.
"""

from __future__ import annotations

from typing import Any

from polimillionaire._vendor.millionaire_client.models import Question


def render_question_block(question: Question) -> str:
    """Format the question + numbered options for the user turn.

    The bracketed integer is the actual server-side option id, which is what
    the model commits to in `answer_id` — no letter-to-id mapping layer.
    """
    lines = [f"Q: {question.text}", "", "Options:"]
    for opt in question.options:
        lines.append(f"[{opt.id}] {opt.text}")
    return "\n".join(lines)


def make_schema(question: Question, *, include_rationale: bool = True) -> dict[str, Any]:
    """Build the per-question JSON schema for `LLM.complete_json`.

    Property order matters: GBNF generates fields in declaration order, so
    `rationale` is listed first to give the model a chain-of-thought window
    before it commits to `answer_id`.

    `include_rationale=False` is for the future two-stage flow where the
    rationale has already been produced in a prior free-form turn.
    """
    option_ids = [opt.id for opt in question.options]
    properties: dict[str, Any] = {}
    required: list[str] = []
    if include_rationale:
        properties["rationale"] = {"type": "string"}
        required.append("rationale")
    properties["confidence"] = {"type": "number", "minimum": 0, "maximum": 1}
    properties["answer_id"] = {"type": "integer", "enum": option_ids}
    required.extend(["confidence", "answer_id"])
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
