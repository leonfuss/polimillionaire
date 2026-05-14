"""Unit tests for LiveWikiRetriever.

HTTP is fully mocked -- we don't hit Wikipedia from CI. The interesting
behaviours are: search+extract round-trip shape, in-process caching,
graceful failure on error responses, and per-title dedup of empty
extracts (MediaWiki returns `missing: 1` pages for typos).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from polimillionaire.retrieval.live_wiki import LiveWikiRetriever


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        # `_get_with_retry` calls `.raise_for_status()` then returns the
        # response -- no-op since our fakes are always 200.
        self.status_code = 200

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        return None


def _stub_pages(*titles_with_bodies: tuple[str, str]) -> dict[str, Any]:
    """MediaWiki `pages` payload keyed by arbitrary pageids."""
    return {
        str(i): {"title": title, "extract": body}
        for i, (title, body) in enumerate(titles_with_bodies)
    }


def _patch_get(payloads: list[dict[str, Any]]):
    """Return successive payloads in order from a patched `_get_with_retry`."""
    it = iter(payloads)
    return patch(
        "polimillionaire.retrieval.live_wiki._get_with_retry",
        side_effect=lambda *_a, **_kw: _FakeResponse(next(it)),
    )


def test_search_returns_passages_in_search_rank_order() -> None:
    search_payload = {
        "query": {
            "search": [
                {"title": "Inception"},
                {"title": "Christopher Nolan"},
                {"title": "Leonardo DiCaprio"},
            ]
        }
    }
    extracts_payload = {
        "query": {
            "pages": _stub_pages(
                ("Inception", "Inception is a 2010 film by Christopher Nolan."),
                ("Christopher Nolan", "British-American filmmaker."),
                ("Leonardo DiCaprio", "American actor."),
            )
        }
    }
    retr = LiveWikiRetriever(top_k=3)
    with _patch_get([search_payload, extracts_payload]):
        out = retr.search("Inception film director")

    assert [p.metadata["title"] for p in out] == [
        "Inception",
        "Christopher Nolan",
        "Leonardo DiCaprio",
    ]
    # nominal score is monotonically decreasing so reranker-free fallback
    # preserves search-rank order
    assert out[0].score > out[1].score > out[2].score
    assert all(p.metadata["source"] == "live_wiki" for p in out)
    assert all(p.id.startswith("live/") for p in out)
    assert all("en.wikipedia.org/wiki/" in p.metadata["url"] for p in out)


def test_search_drops_pages_with_empty_extracts() -> None:
    """MediaWiki returns empty `extract` for missing/redirect-stub pages.
    Those have no usable text -- the reranker would just rank them as junk."""
    search_payload = {"query": {"search": [{"title": "Real"}, {"title": "MissingTypo"}]}}
    extracts_payload = {
        "query": {
            "pages": _stub_pages(
                ("Real", "A real article body."),
                ("MissingTypo", ""),
            )
        }
    }
    retr = LiveWikiRetriever()
    with _patch_get([search_payload, extracts_payload]):
        out = retr.search("real or typo")
    assert [p.metadata["title"] for p in out] == ["Real"]


def test_search_caches_by_query_string() -> None:
    payload_search = {"query": {"search": [{"title": "T"}]}}
    payload_extracts = {"query": {"pages": _stub_pages(("T", "body"))}}
    retr = LiveWikiRetriever()
    call_count = 0

    def stub(*_a, **_kw):
        nonlocal call_count
        call_count += 1
        return _FakeResponse(payload_search if call_count % 2 == 1 else payload_extracts)

    with patch("polimillionaire.retrieval.live_wiki._get_with_retry", side_effect=stub):
        first = retr.search("same query")
        second = retr.search("same query")
    assert first == second
    # 2 HTTP calls total -- the second `search()` hit the cache.
    assert call_count == 2


def test_search_returns_empty_on_http_failure() -> None:
    """A 429 or DNS hiccup must never raise -- callers fuse [] cleanly."""
    retr = LiveWikiRetriever()
    with patch(
        "polimillionaire.retrieval.live_wiki._get_with_retry",
        side_effect=RuntimeError("boom"),
    ):
        out = retr.search("anything")
    assert out == []


def test_search_returns_empty_when_api_finds_nothing() -> None:
    """No `search` hits -> we shouldn't even try the extracts call."""
    retr = LiveWikiRetriever()
    # Only one payload -- if the code tried to fetch extracts, `next()` on
    # the empty iterator would raise StopIteration and fail the test.
    with _patch_get([{"query": {"search": []}}]):
        out = retr.search("zzzzz no such article")
    assert out == []


def test_char_cap_truncates_long_extracts() -> None:
    long_body = "x" * 5000
    payloads = [
        {"query": {"search": [{"title": "Long"}]}},
        {"query": {"pages": _stub_pages(("Long", long_body))}},
    ]
    retr = LiveWikiRetriever(char_cap=200)
    with _patch_get(payloads):
        out = retr.search("long article")
    assert len(out) == 1
    # truncated body + ellipsis marker
    assert out[0].text.endswith(" [...]")
    assert len(out[0].text) <= 200 + len(" [...]")
