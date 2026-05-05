"""Schema helpers shared across strategies.

Prompt-formatting helpers (`render_question_block`) live under `prompts/`
to avoid `prompts/* -> strategies/*` imports that would form a cycle.
"""

from __future__ import annotations

from typing import Any

from polimillionaire._vendor.millionaire_client.models import Question


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


def make_action_schema(question: Question) -> dict[str, Any]:
    """Schema for a ReAct-style step: either call a tool, or commit to an answer.

    Top-level `oneOf` with a `const`-typed `action` discriminator. The answer
    branch mirrors `make_schema` (rationale first, then confidence, then
    `answer_id` constrained to this question's option ids) so the eval/replay
    layer sees identical fields regardless of which strategy produced them.
    """
    option_ids = [opt.id for opt in question.options]
    return {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "const": "calculate"},
                    "expression": {"type": "string"},
                },
                "required": ["action", "expression"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "const": "answer"},
                    "rationale": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "answer_id": {"type": "integer", "enum": option_ids},
                },
                "required": ["action", "rationale", "confidence", "answer_id"],
                "additionalProperties": False,
            },
        ]
    }
