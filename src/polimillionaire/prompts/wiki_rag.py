"""Wikipedia-RAG prompt variants."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polimillionaire._vendor.millionaire_client.models import Question
from polimillionaire.llm import Message
from polimillionaire.prompts._common import PromptVariant, render_question_block

if TYPE_CHECKING:
    # Passage lives in the optional [rag] group; deferred so this module
    # imports cleanly on a base install.
    from polimillionaire.retrieval.retriever import Passage

# ~300 words ≈ 1800 chars for an average English passage. Long wiki
# paragraphs can run 600+ words; truncating keeps the prompt manageable.
_MAX_PASSAGE_CHARS = 1800

_V1_SYSTEM = (
    "You are an expert trivia player. Wikipedia excerpts relevant to the "
    "question are provided below. Use them when they are helpful; rely on "
    "your own knowledge when they are not. First write a brief rationale "
    "(no more than three sentences), then commit to the option you believe "
    "is correct."
)

# Variant without the rationale instruction. Pair with a schema that omits
# the `rationale` field (make_schema(include_rationale=False)) so the model
# returns just confidence + answer_id. Cheaper at decode time -- useful when
# the wall-clock budget is tight and we don't need the chain-of-thought.
_V2_NOREASON_SYSTEM = (
    "You are an expert trivia player. Wikipedia excerpts relevant to the "
    "question are provided below. Use them when they are helpful; rely on "
    "your own knowledge when they are not. Commit to the option you believe "
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


def _render_wiki_rag_v1(question: Question, passages: list[Passage]) -> list[Message]:
    block = _format_passage_block(passages)
    user_parts = []
    if block:
        user_parts.append(block)
    user_parts.append(render_question_block(question))
    return [
        {"role": "system", "content": _V1_SYSTEM},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _render_wiki_rag_v2_noreason(question: Question, passages: list[Passage]) -> list[Message]:
    block = _format_passage_block(passages)
    user_parts = []
    if block:
        user_parts.append(block)
    user_parts.append(render_question_block(question))
    return [
        {"role": "system", "content": _V2_NOREASON_SYSTEM},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


PROMPTS: dict[str, PromptVariant] = {
    "wiki_rag/v1": PromptVariant(version="wiki_rag/v1", render=_render_wiki_rag_v1),
    "wiki_rag/v2-noreason": PromptVariant(
        version="wiki_rag/v2-noreason", render=_render_wiki_rag_v2_noreason
    ),
}

LATEST = "wiki_rag/v1"
NOREASON = "wiki_rag/v2-noreason"

# legacy module-level aliases so existing callers that do `prompt.render(...)` keep working
PROMPT_VERSION = LATEST
render = PROMPTS[LATEST].render
