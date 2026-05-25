"""Per-question live news retrieval via GDELT DOC 2.0.

For news questions Wikipedia is the wrong source: it's slow to update,
opinionated about notability, and biased toward historical coverage.
GDELT indexes the global news firehose in near-real-time with no API
key required, which makes it the right primitive for the News
competition (cid 5).

Public surface mirrors `LiveWikiRetriever.search(query, k) -> list[Passage]`
so the strategy is source-agnostic -- only the factory picks which one
to use per competition.

Hard contract (same as live_wiki): any HTTP or parsing failure returns
`[]` and logs the reason. A 429 / 503 / DNS hiccup must never abort an
answer; the LLM will fall back to its parametric knowledge.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

import requests

from polimillionaire.retrieval.wiki_crawler import _DEFAULT_UA, _get_with_retry

if TYPE_CHECKING:
    from polimillionaire.retrieval.retriever import Passage

_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT is keyword/Boolean; question scaffolding hurts recall the same
# way it does on MediaWiki. Reuse a small, news-flavoured stop set --
# narrower than live_wiki's (no trivia-specific scaffolding like
# "primary"/"theme" which can appear meaningfully in news headlines).
_QUERY_STOP_WORDS = frozenset(
    {
        "what",
        "which",
        "who",
        "whom",
        "when",
        "where",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "has",
        "have",
        "had",
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "of",
        "in",
        "on",
        "for",
        "to",
        "with",
        "by",
        "at",
        "from",
        "as",
        "into",
        "and",
        "or",
        "but",
    }
)
_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)


def _clean_query(raw: str) -> str:
    """Trim a question into a keyword-search-friendly form.

    Returns the raw query unchanged if filtering would leave nothing.
    """
    tokens = _TOKEN_RE.findall(raw)
    kept = [t for t in tokens if t.lower() not in _QUERY_STOP_WORDS]
    return " ".join(kept) if kept else raw


def _format_seendate(raw: str) -> str:
    """`20260516T153000Z` -> `2026-05-16`. Returns raw on parse failure."""
    if len(raw) < 8 or not raw[:8].isdigit():
        return raw
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}"


class LiveGDELTRetriever:
    """Per-call GDELT news lookup. Drop-in alongside `LiveWikiRetriever`."""

    def __init__(
        self,
        *,
        top_k: int = 6,
        timeout: float = 12.0,
        timespan: str | None = None,
        source_lang: str | None = "eng",
        user_agent: str | None = None,
        verbose: bool = False,
    ) -> None:
        """
        `timespan` is a GDELT relative window like "1y" / "6m" / "7d".
        None (default) hits the full ~2015-present index ranked by
        relevance, which suits historical news questions; pass "1y" if
        the question set skews recent.

        `source_lang` filters by article language. "eng" by default --
        the quiz is in English so non-English hits are noise.
        """
        self._top_k = top_k
        self._timeout = timeout
        self._timespan = timespan
        self._source_lang = source_lang
        self._verbose = verbose
        ua = user_agent or os.environ.get("WIKI_USER_AGENT", _DEFAULT_UA)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": ua})
        # In-process cache keyed on the query string. Same rationale as
        # live_wiki: live play sees few duplicates within a game, but a
        # kernel-restart retry of the same game will hit this.
        self._cache: dict[str, list[Passage]] = {}

    def search(self, query: str, k: int | None = None) -> list[Passage]:
        """Return up to `k` recent news headlines for `query`.

        On any failure (network, JSON, missing fields), logs and returns
        `[]`. Never propagates exceptions to the caller.
        """
        k = k or self._top_k
        if query in self._cache:
            hits = self._cache[query][:k]
            if self._verbose:
                print(f"   [live_gdelt] cache hit: {len(hits)} article(s) for query")
            return hits

        cleaned = _clean_query(query)
        # GDELT's query DSL ANDs space-separated terms. The language
        # filter is just another space-separated operator; no parens
        # needed. Empirically, wrapping in parens trips a different
        # parse path that returns rate-limit-style errors more often.
        gdelt_query = cleaned
        if self._source_lang:
            gdelt_query = f"{gdelt_query} sourcelang:{self._source_lang}"

        if self._verbose:
            shown = cleaned if len(cleaned) <= 80 else cleaned[:77] + "..."
            print(f'   [live_gdelt] searching: "{shown}" (top {k})')
            if cleaned != query.strip():
                raw_shown = query if len(query) <= 80 else query[:77] + "..."
                print(f'   [live_gdelt] cleaned from: "{raw_shown}"')

        try:
            articles = self._search_articles(gdelt_query, k)
        except Exception as e:  # noqa: BLE001 -- never break play on retrieval
            print(f"   [live_gdelt] search failed ({type(e).__name__}: {e}); returning []")
            return []
        if not articles:
            if self._verbose:
                print("   [live_gdelt] no articles returned for query")
            return []

        from polimillionaire.retrieval.retriever import Passage

        passages: list[Passage] = []
        # GDELT returns articles in the requested sort order (HybridRel
        # here). Preserve that via enumerate + nominal descending score;
        # absolute values only matter as a tiebreak when no reranker
        # runs downstream.
        for i, art in enumerate(articles):
            title = (art.get("title") or "").strip()
            if not title:
                continue
            domain = art.get("domain", "")
            seendate = _format_seendate(art.get("seendate", ""))
            url = art.get("url", "")
            passages.append(
                Passage(
                    id=f"gdelt/{i}",
                    text=title,
                    metadata={
                        "source": "live_gdelt",
                        "title": title,
                        "domain": domain,
                        "seendate": seendate,
                        "url": url,
                    },
                    score=1.0 - i * 0.05,
                )
            )

        self._cache[query] = passages
        if self._verbose:
            print(f"   [live_gdelt] fetched {len(passages)} article(s)")
        return passages

    def _search_articles(self, query: str, k: int) -> list[dict]:
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": str(max(1, min(k, 250))),
            "sort": "HybridRel",
        }
        if self._timespan:
            params["timespan"] = self._timespan
        # max_attempts=1: a single try, no retry. GDELT is slow enough that
        # exponential-backoff retries blow past the 30s game timer. Failing
        # to [] lets the LLM answer from parametric knowledge -- still a
        # better outcome than timing out the answer.
        resp = _get_with_retry(
            self._session, _API_URL, params, timeout=self._timeout, max_attempts=1
        )
        # GDELT enforces ~1 req/5s and signals rate-limiting with a 200
        # body of plain text ("Please limit requests to one every 5
        # seconds..."), not an HTTP status code. Catch it explicitly so
        # we log meaningfully instead of silently returning [].
        body = resp.text
        if "Please limit requests" in body:
            if self._verbose:
                print("   [live_gdelt] rate limited (1 req/5s) -- returning []")
            return []
        try:
            data = resp.json()
        except ValueError:
            return []
        return data.get("articles", []) or []
