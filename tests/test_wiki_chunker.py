"""Tests for wiki_chunker.chunk_article."""

from __future__ import annotations

import pytest

from polimillionaire.retrieval.wiki_chunker import chunk_article


def test_short_article_produces_one_chunk() -> None:
    body = "This is a short article body."
    chunks = chunk_article("Short Article", body)
    assert len(chunks) == 1
    assert chunks[0].text.startswith("# Short Article\n\n")
    assert "This is a short article body." in chunks[0].text


def test_long_article_splits_into_multiple_chunks() -> None:
    # build a body of ~600 words spread across two paragraphs
    para = " ".join(["word"] * 300)
    body = para + "\n\n" + para
    chunks = chunk_article("Long Article", body, target_tokens=256)
    assert len(chunks) > 1
    for chunk in chunks:
        # each chunk should be at most target + some paragraph overhead
        # (title line adds a few words but we check the body portion)
        word_count = len(chunk.text.split())
        # title prefix is "# Long Article\n\n" ~ 3 words; allow generous headroom
        assert word_count <= 256 + 10, f"chunk too large: {word_count} words"


def test_oversized_single_paragraph_gets_windowed() -> None:
    # one paragraph longer than target_tokens -- must be windowed
    para = " ".join([f"word{i}" for i in range(400)])
    chunks = chunk_article("Huge Para", para, target_tokens=100, overlap_tokens=10)
    assert len(chunks) > 1
    # every chunk body (after stripping the title line) should be <= ~110 words
    for chunk in chunks:
        lines = chunk.text.split("\n\n", 2)
        body_text = lines[-1] if len(lines) > 1 else chunk.text
        assert len(body_text.split()) <= 110


def test_empty_body_returns_empty_list() -> None:
    assert chunk_article("Empty", "") == []
    assert chunk_article("Empty", "   \n\n  ") == []


def test_chunk_ids_are_unique() -> None:
    body = "\n\n".join([" ".join(["x"] * 50) for _ in range(10)])
    chunks = chunk_article("Title", body)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_metadata_fields() -> None:
    chunks = chunk_article("My Article", "Some content here.")
    assert len(chunks) == 1
    meta = chunks[0].metadata
    assert meta["title"] == "My Article"
    assert "url" in meta


def test_url_encoding_for_title_with_spaces() -> None:
    chunks = chunk_article("Solar System", "Planets orbit the Sun.")
    assert len(chunks) == 1
    url = chunks[0].metadata["url"]
    # spaces in the title should be encoded (as _ or %20) in the URL
    assert " " not in url
    assert "Solar" in url


def test_overlap_greater_than_target_raises() -> None:
    para = " ".join(["w"] * 200)
    with pytest.raises(ValueError, match="overlap_tokens"):
        chunk_article("X", para, target_tokens=64, overlap_tokens=64)
