"""Prompt for the Wikipedia-RAG strategy: numbered excerpts + question, single turn.

Bump `PROMPT_VERSION` whenever wording changes so accuracy shifts attribute
to the prompt, not the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.llm import Message
from polimillionaire.prompts._common import render_question_block

if TYPE_CHECKING:
    # Passage lives in the optional [rag] group; deferred so this module
    # imports cleanly on a base install.
    from polimillionaire.retrieval.retriever import Passage

PROMPT_VERSION = "wiki_rag/v1"

# ~300 words ≈ 1800 chars for an average English passage. Long wiki
# paragraphs can run 600+ words; truncating keeps the prompt manageable.
_MAX_PASSAGE_CHARS = 1800

SYSTEM = (
    "You are an expert trivia player. Wikipedia excerpts relevant to the "
    "question are provided below. Use them when they are helpful; rely on "
    "your own knowledge when they are not. First write a brief rationale "
    "(no more than three sentences), then commit to the option you believe "
    "is correct."
)


def _format_passage(p: Passage, idx: int) -> str:
    title = p.metadata.get("title", "")
    body = p.text
    if len(body) > _MAX_PASSAGE_CHARS:
        body = body[:_MAX_PASSAGE_CHARS] + " [...]"
    header = f"[{idx}] {title}" if title else f"[{idx}]"
    return f"{header}\n{body}"


def _format_passage_block(passages: list[Passage]) -> str:
    if not passages:
        return ""
    parts = "\n\n".join(_format_passage(p, i + 1) for i, p in enumerate(passages))
    return f"Wikipedia excerpts:\n\n{parts}"


def render(question: Question, passages: list[Passage]) -> list[Message]:
    """Build the message list for a single wiki-RAG turn."""
    block = _format_passage_block(passages)
    user_parts = []
    if block:
        user_parts.append(block)
    user_parts.append(render_question_block(question))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]
