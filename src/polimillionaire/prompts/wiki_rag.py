"""Live-RAG prompt variants.

Despite the historical "wiki_rag" filename, this module also hosts the
news_rag variant used when the live source is GDELT (cid 5). Co-located
to keep the strategy's `prompt.PROMPTS` lookup single-source.
"""

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


# --- news_rag variant -------------------------------------------------------
# GDELT only gives us titles + metadata, not body text, so headlines render
# as a one-line list rather than the multi-paragraph Wikipedia layout above.

_NEWS_V1_SYSTEM = (
    "You are an expert trivia player. Recent news headlines relevant to the "
    "question are listed below, each with its source and publication date. "
    "Use them when they are helpful; rely on your own knowledge when they "
    "are not. First write a brief rationale (no more than three sentences), "
    "then commit to the option you believe is correct."
)


def _format_news_passage(p: Passage, idx: int) -> str:
    title = p.metadata.get("title") or p.text
    domain = p.metadata.get("domain", "")
    seendate = p.metadata.get("seendate", "")
    tail_bits = [b for b in (domain, seendate) if b]
    tail = f" ({', '.join(tail_bits)})" if tail_bits else ""
    return f"[{idx}] {title}{tail}"


def _format_news_block(passages: list[Passage]) -> str:
    if not passages:
        return ""
    parts = "\n".join(_format_news_passage(p, i + 1) for i, p in enumerate(passages))
    return f"Recent news headlines:\n\n{parts}"


def _render_news_rag_v1(question: Question, passages: list[Passage]) -> list[Message]:
    block = _format_news_block(passages)
    user_parts = []
    if block:
        user_parts.append(block)
    user_parts.append(render_question_block(question))
    return [
        {"role": "system", "content": _NEWS_V1_SYSTEM},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


# --- news_rag/v2-articles ---------------------------------------------------
# Article-snippet variant for sources that return real body text (Tavily).
# Mirrors the wiki_rag layout (header + multi-paragraph blocks) but labels
# the content as news rather than encyclopedia material.

_NEWS_ARTICLES_SYSTEM = (
    "You are an expert trivia player. News article excerpts relevant to the "
    "question are provided below, each with its source and publication date. "
    "Use them when they are helpful; rely on your own knowledge when they "
    "are not. First write a brief rationale (no more than three sentences), "
    "then commit to the option you believe is correct."
)


def _format_news_article_passage(p: Passage, idx: int) -> str:
    title = p.metadata.get("title", "")
    domain = p.metadata.get("domain", "")
    seendate = p.metadata.get("seendate", "")
    tail_bits = [b for b in (domain, seendate) if b]
    tail = f" ({', '.join(tail_bits)})" if tail_bits else ""
    body = p.text
    if len(body) > _MAX_PASSAGE_CHARS:
        body = body[:_MAX_PASSAGE_CHARS] + " [...]"
    header = f"[{idx}] {title}{tail}" if title else f"[{idx}]{tail}"
    return f"{header}\n{body}"


def _format_news_article_block(passages: list[Passage]) -> str:
    if not passages:
        return ""
    parts = "\n\n".join(_format_news_article_passage(p, i + 1) for i, p in enumerate(passages))
    return f"News article excerpts:\n\n{parts}"


def _render_news_rag_v2_articles(question: Question, passages: list[Passage]) -> list[Message]:
    block = _format_news_article_block(passages)
    user_parts = []
    if block:
        user_parts.append(block)
    user_parts.append(render_question_block(question))
    return [
        {"role": "system", "content": _NEWS_ARTICLES_SYSTEM},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


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
    "news_rag/v1": PromptVariant(version="news_rag/v1", render=_render_news_rag_v1),
    "news_rag/v2-articles": PromptVariant(
        version="news_rag/v2-articles", render=_render_news_rag_v2_articles
    ),
}

LATEST = "wiki_rag/v1"
NOREASON = "wiki_rag/v2-noreason"
NEWS_LATEST = "news_rag/v1"
NEWS_ARTICLES = "news_rag/v2-articles"

# legacy module-level aliases so existing callers that do `prompt.render(...)` keep working
PROMPT_VERSION = LATEST
render = PROMPTS[LATEST].render
