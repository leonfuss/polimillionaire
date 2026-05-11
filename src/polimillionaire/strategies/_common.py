"""Schema helpers shared across strategies.

Prompt-formatting helpers (`render_question_block`) live under `prompts/`
to avoid `prompts/* -> strategies/*` imports that would form a cycle.
"""

from __future__ import annotations

import json
import time
from typing import Any

from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.llm import LLM, Message
from polimillionaire.strategies.base import AnswerDecision
from polimillionaire.tools import calc


def build_decision(
    out: dict,
    start: float,
    *,
    model_name: str,
    strategy_name: str,
    prompt_version: str,
) -> AnswerDecision:
    """Build an AnswerDecision from a parsed model output and a perf_counter start time.

    All strategies that produce `{rationale, confidence, answer_id}` output share
    this shape -- keeping it here avoids three identical 10-line copies.
    """
    latency_ms = int((time.perf_counter() - start) * 1000)
    return AnswerDecision(
        option_id=int(out["answer_id"]),
        confidence=float(out["confidence"]),
        rationale=out.get("rationale"),
        model_name=model_name,
        strategy_name=strategy_name,
        prompt_version=prompt_version,
        latency_ms=latency_ms,
    )


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


def run_react_loop(
    llm: LLM,
    initial_messages: list[Message],
    question: Question,
    *,
    max_steps: int,
    model_name: str,
    strategy_name: str,
    prompt_version: str,
    start: float | None = None,
    verbose: bool = False,
    log_prefix: str = "react",
) -> AnswerDecision:
    """Bounded calculate-or-answer ReAct loop. Forces an answer on step cap or
    parse failure; defaults to option 0 at confidence 0 if even that fails.

    Pass `start` from before any retrieval the caller did so the returned
    AnswerDecision.latency_ms reflects total wall-clock; defaults to "now"
    for callers (like CalcReactStrategy) that have no retrieval prefix.
    """
    messages: list[Message] = list(initial_messages)
    action_schema = make_action_schema(question)
    if start is None:
        start = time.perf_counter()

    for _ in range(max_steps):
        try:
            out = llm.complete_json(messages, action_schema)
        except ValueError:
            # output blew through max_tokens and didn't parse as JSON;
            # skip remaining steps and force an answer with whatever context we have
            if verbose:
                print(f"   [{log_prefix}] action step failed to parse — forcing answer")
            break
        if out["action"] == "answer":
            return build_decision(
                out,
                start,
                model_name=model_name,
                strategy_name=strategy_name,
                prompt_version=prompt_version,
            )
        expression = out["expression"]
        result = calc(expression)
        if verbose:
            print(f'   [{log_prefix}] calc("{expression}") -> {result}')
        messages.append({"role": "assistant", "content": json.dumps(out)})
        messages.append({"role": "user", "content": f"Calculator: `{expression}` = {result}"})

    # step cap hit or parse failure — force a commit on the answer-only schema
    messages.append(
        {"role": "user", "content": "Step limit reached. Answer now using the answer schema."}
    )
    try:
        out = llm.complete_json(messages, make_schema(question))
    except ValueError:
        # even the forced answer didn't parse; default to the first option at zero
        # confidence so the game submits something and continues
        if verbose:
            print(f"   [{log_prefix}] forced answer also failed — defaulting to option 0")
        return build_decision(
            {
                "rationale": "Model output failed to parse; defaulting to first option.",
                "answer_id": question.options[0].id,
                "confidence": 0.0,
            },
            start,
            model_name=model_name,
            strategy_name=strategy_name,
            prompt_version=prompt_version,
        )
    return build_decision(
        out,
        start,
        model_name=model_name,
        strategy_name=strategy_name,
        prompt_version=prompt_version,
    )
