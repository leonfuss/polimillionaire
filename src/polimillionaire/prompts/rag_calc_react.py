"""Prompt for RAG-augmented calc-react.

Reuses `calc_react.SYSTEM` and the four hand-crafted ReAct exemplars
verbatim -- those teach the JSON action format the model has to emit.
On top, this module formats the k retrieved MATH problems as natural-
language reference solutions and appends them to the system message.
The model thus reads:

    [SYSTEM: calc-react instructions]
    [SYSTEM tail: "Below are similar problems and their solutions..."]
    [hand-crafted ReAct exemplars: action JSON format]
    [actual question]

The references go in the system rather than as a fake user/assistant
pair because they're context, not a turn -- splitting them across roles
would invite the model to "respond" to them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.llm import Message
from polimillionaire.prompts._common import render_question_block
from polimillionaire.prompts.calc_react import EXEMPLAR_MESSAGES, SYSTEM

if TYPE_CHECKING:
    # `Passage` is only used as a type annotation; deferred so this prompt
    # module imports cleanly without the `[rag]` deps.
    from polimillionaire.retrieval.retriever import Passage

PROMPT_VERSION = "rag-v1"

# Cap each retrieved solution. The MATH dataset's solutions average ~400
# chars but tail out past 1500 on the hardest problems; left uncapped
# they bloat the prompt past the model's useful attention span.
_MAX_SOLUTION_CHARS = 600


def _format_reference(p: Passage, idx: int) -> str:
    subject = p.metadata.get("subject", "math")
    solution = p.metadata.get("solution", "")
    if len(solution) > _MAX_SOLUTION_CHARS:
        solution = solution[:_MAX_SOLUTION_CHARS] + " [...]"
    return (
        f"Reference {idx} ({subject}, similarity={p.score:.2f}):\n"
        f"Problem: {p.text}\n"
        f"Solution: {solution}"
    )


def _format_reference_block(passages: list[Passage]) -> str:
    if not passages:
        return ""
    body = "\n\n".join(_format_reference(p, i + 1) for i, p in enumerate(passages))
    return (
        "Below are similar problems and their step-by-step solutions, "
        "retrieved from a problem bank for reference. Use them to "
        "recognise the pattern, then solve the actual question with the "
        "calculator -- do not copy a reference's answer if the actual "
        "question's numbers differ.\n"
        "\n"
        f"{body}"
    )


def render(question: Question, references: list[Passage]) -> list[Message]:
    """Build the initial message list for a RAG-augmented calc-react turn."""
    system = SYSTEM
    block = _format_reference_block(references)
    if block:
        system = SYSTEM + "\n\n---\n\n" + block
    return [
        {"role": "system", "content": system},
        *EXEMPLAR_MESSAGES,
        {"role": "user", "content": render_question_block(question)},
    ]
