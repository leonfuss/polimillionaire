"""Paragraph-packing chunker for Wikipedia article bodies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote


@dataclass(frozen=True)
class WikiChunk:
    id: str  # f"{title}#chunk{i}"
    text: str
    metadata: dict[str, Any]  # {"title", "url"}


def _word_count(text: str) -> int:
    return len(text.split())


def _window_paragraph(
    para: str,
    target: int,
    overlap: int,
) -> list[str]:
    """Split a single long paragraph into overlapping word windows."""
    if overlap >= target:
        raise ValueError(f"overlap_tokens ({overlap}) must be < target_tokens ({target})")
    words = para.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + target
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def chunk_article(
    title: str,
    body: str,
    *,
    target_tokens: int = 256,
    overlap_tokens: int = 32,
) -> list[WikiChunk]:
    """Pack paragraphs into ~target_tokens chunks; oversized paragraphs are word-windowed."""
    if not body or not body.strip():
        return []

    url = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
    metadata_base: dict[str, Any] = {"title": title, "url": url}

    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

    raw_chunks: list[str] = []
    current_parts: list[str] = []
    current_count = 0

    for para in paragraphs:
        para_count = _word_count(para)

        if para_count > target_tokens:
            if current_parts:
                raw_chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_count = 0
            raw_chunks.extend(_window_paragraph(para, target_tokens, overlap_tokens))
        elif current_count + para_count > target_tokens and current_parts:
            raw_chunks.append("\n\n".join(current_parts))
            current_parts = [para]
            current_count = para_count
        else:
            current_parts.append(para)
            current_count += para_count

    if current_parts:
        raw_chunks.append("\n\n".join(current_parts))

    # prepend the title to every chunk so the embedder bakes title context into the vector.
    return [
        WikiChunk(
            id=f"{title}#chunk{i}",
            text=f"# {title}\n\n{text}",
            metadata=dict(metadata_base),
        )
        for i, text in enumerate(raw_chunks)
    ]
