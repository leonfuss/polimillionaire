"""Per-question live Wikipedia retrieval.

The static `wiki_entertainment` / `wiki_science` indices were crawled at
build time and miss anything added or revised since. For entertainment in
particular the static index is *confidently wrong* on long-tail questions
(e.g. old film cast lists, retitled songs) -- the failure mode that
matters is plausible-looking-but-stale, not missing.

This module hits the MediaWiki search API on a per-question basis and
returns its top hits as `Passage` objects with `source="live_wiki"`.
Designed to be fused into the existing rerank pool alongside the static
hits; the reranker scores both kinds together and we let it pick winners.

Hard contract: any HTTP or parsing failure returns `[]` and logs the
reason. A 429 or DNS hiccup must never abort an answer -- the static
index will carry the question.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import requests

from polimillionaire.retrieval.wiki_crawler import _DEFAULT_UA, _get_with_retry

if TYPE_CHECKING:
    from polimillionaire.retrieval.retriever import Passage

_API_URL = "https://en.wikipedia.org/w/api.php"


class LiveWikiRetriever:
    """Per-call Wikipedia lookup. Drop-in alongside `Retriever.search()`."""

    def __init__(
        self,
        *,
        top_k: int = 5,
        char_cap: int = 1200,
        timeout: float = 8.0,
        user_agent: str | None = None,
        verbose: bool = False,
    ) -> None:
        self._top_k = top_k
        self._char_cap = char_cap
        self._timeout = timeout
        self._verbose = verbose
        ua = user_agent or os.environ.get("WIKI_USER_AGENT", _DEFAULT_UA)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": ua})
        # In-process cache keyed on the query string. Live play sees few
        # duplicates, but a kernel-restart retry of the same game will.
        self._cache: dict[str, list[Passage]] = {}

    def search(self, query: str, k: int | None = None) -> list[Passage]:
        """Return up to `k` live Wikipedia passages for `query`.

        On any failure (network, JSON, missing fields), logs and returns
        `[]`. Never propagates exceptions to the caller.
        """
        k = k or self._top_k
        if query in self._cache:
            hits = self._cache[query][:k]
            if self._verbose:
                print(f"   [live_wiki] cache hit: {len(hits)} passage(s) for query")
            return hits

        if self._verbose:
            shown = query if len(query) <= 80 else query[:77] + "..."
            print(f'   [live_wiki] searching: "{shown}" (top {k})')

        try:
            titles = self._search_titles(query, k)
        except Exception as e:  # noqa: BLE001 -- never break play on retrieval
            print(f"   [live_wiki] search failed ({type(e).__name__}: {e}); returning []")
            return []
        if not titles:
            if self._verbose:
                print("   [live_wiki] no titles returned for query")
            return []
        if self._verbose:
            print(f"   [live_wiki] titles: {', '.join(titles)}")

        try:
            extracts = self._fetch_extracts(titles)
        except Exception as e:  # noqa: BLE001
            print(f"   [live_wiki] extract fetch failed ({type(e).__name__}: {e}); returning []")
            return []

        # Local import to keep this module importable on a base install
        # (Passage is defined alongside the heavy [rag] deps).
        from polimillionaire.retrieval.retriever import Passage

        passages: list[Passage] = []
        # `titles` is search-rank order; preserve it via enumerate and a
        # nominal descending score. The reranker reassigns real scores
        # downstream, so the absolute values are only used as a tie-break
        # when reranking is disabled.
        for i, title in enumerate(titles):
            body = extracts.get(title, "").strip()
            if not body:
                continue
            text = body if len(body) <= self._char_cap else body[: self._char_cap] + " [...]"
            passages.append(
                Passage(
                    id=f"live/{title}",
                    text=text,
                    metadata={
                        "source": "live_wiki",
                        "title": title,
                        "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                    },
                    score=1.0 - i * 0.05,
                )
            )

        self._cache[query] = passages
        if self._verbose:
            print(f"   [live_wiki] fetched {len(passages)} passage(s) with non-empty extract")
        return passages

    def _search_titles(self, query: str, k: int) -> list[str]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": k,
            "srprop": "",  # we don't need snippets; we fetch full extracts below
            "format": "json",
        }
        resp = _get_with_retry(self._session, _API_URL, params, timeout=self._timeout)
        data = resp.json()
        return [hit["title"] for hit in data.get("query", {}).get("search", [])]

    def _fetch_extracts(self, titles: list[str]) -> dict[str, str]:
        """One batched call -- MediaWiki accepts up to 50 titles per query."""
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": 1,  # only the lead section -- usually the definition
            "explaintext": 1,  # strip wikitext markup
            "exlimit": "max",  # otherwise extracts caps at 1 title even with many `titles`
            "redirects": 1,  # follow redirects so "WW2" -> "World War II"
            "titles": "|".join(titles),
            "format": "json",
        }
        resp = _get_with_retry(self._session, _API_URL, params, timeout=self._timeout)
        pages = resp.json().get("query", {}).get("pages", {})
        out: dict[str, str] = {}
        for page in pages.values():
            title = page.get("title")
            extract = page.get("extract", "")
            if title is not None:
                out[title] = extract
        return out
