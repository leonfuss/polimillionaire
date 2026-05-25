"""Per-question live news retrieval via Tavily.

Tavily is a purpose-built LLM-RAG search API: each result includes a
clean ~200-500 char article snippet (not just a headline), so the
downstream prompt actually has the entity/fact the question is asking
about -- the headline-only limitation that hurt GDELT recall (cid 5
museum spokesperson, drone-attack stats, etc.) goes away.

Public surface matches `LiveWikiRetriever.search(query, k, *, option_texts)`
so the strategy is source-agnostic; the factory picks Tavily over
GDELT when `TAVILY_API_KEY` is set in the environment.

Hard contract (same as live_wiki / live_gdelt): any HTTP or parsing
failure returns `[]` and logs the reason. The circuit breaker arms
after `failure_threshold` consecutive network failures so a flaky
route doesn't burn the wall clock for the rest of the session.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import requests

if TYPE_CHECKING:
    from polimillionaire.retrieval.retriever import Passage

_API_URL = "https://api.tavily.com/search"

# YYYY-MM-DD anywhere in the question body. Used to compute a Tavily
# `days` filter -- Tavily doesn't expose start/end dates directly, only
# a "back N days from now" window.
_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")


def _days_back_from_question(query: str) -> int | None:
    """If the question anchors on a YYYY-MM-DD, return how many days
    back that date is from today plus a small buffer, suitable for
    Tavily's `days` param. Returns None when no date is found."""
    m = _DATE_RE.search(query)
    if not m:
        return None
    try:
        d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=UTC)
    except ValueError:
        return None
    delta = (datetime.now(UTC) - d).days
    if delta < 0:
        # Question references a future date -- nothing to constrain on.
        return None
    # +3 day buffer to catch publication lag without ballooning the window.
    return delta + 3


def _domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.removeprefix("www.")
    except Exception:  # noqa: BLE001
        return ""


class TavilyConfigError(RuntimeError):
    """Raised when Tavily is requested but no API key is available."""


class LiveTavilyRetriever:
    """Per-call Tavily news lookup. Drop-in alongside GDELT/Wikipedia."""

    # Lets the factory pick a source-appropriate prompt variant without
    # isinstance checks.
    source_name = "tavily"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        top_k: int = 5,
        timeout: float = 12.0,
        topic: str = "news",
        search_depth: str = "basic",
        verbose: bool = False,
        failure_threshold: int = 3,
    ) -> None:
        """
        `api_key` defaults to `TAVILY_API_KEY` from the environment.
        Raises `TavilyConfigError` if neither is set -- callers should
        pre-check the env so they can fall back to GDELT silently.

        `topic="news"` biases results toward recent news articles and
        unlocks the `days` recency filter. Use `"general"` for broader
        web results.

        `search_depth="basic"` is cheap and ~1s; "advanced" runs deeper
        crawl + extraction, ~3-5s and 2x the credit cost. Basic is the
        right default on a 30s timer.
        """
        resolved_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not resolved_key:
            raise TavilyConfigError(
                "TAVILY_API_KEY not set; cannot instantiate LiveTavilyRetriever"
            )
        self._api_key = resolved_key
        self._top_k = top_k
        self._timeout = timeout
        self._topic = topic
        self._search_depth = search_depth
        self._verbose = verbose
        self._session = requests.Session()
        self._cache: dict[str, list[Passage]] = {}
        self._failure_threshold = failure_threshold
        self._consecutive_failures = 0
        self._disabled = False

    def search(
        self,
        query: str,
        k: int | None = None,
        *,
        option_texts: list[str] | None = None,
    ) -> list[Passage]:
        """Return up to `k` Tavily search results as Passages.

        `option_texts` is appended to the query as "Possible answers
        include: A, B, C, D" -- Tavily's NL search benefits from the
        extra entity context without breaking on Boolean operators the
        way GDELT does.

        Never raises; failures log and return [].
        """
        k = k or self._top_k

        if self._disabled:
            if self._verbose:
                print("   [live_tavily] circuit breaker open; skipping network call")
            return []

        # Tavily handles natural-language queries well; no aggressive
        # stop-word stripping. Just append the options as anchor hints.
        full_query = query.strip()
        if option_texts:
            opts = [t.strip() for t in option_texts if t and t.strip()]
            if opts:
                full_query = f"{full_query} Possible answers include: {', '.join(opts)}."

        days_back = _days_back_from_question(query) if self._topic == "news" else None

        cache_key = f"{full_query}||{days_back or ''}"
        if cache_key in self._cache:
            hits = self._cache[cache_key][:k]
            if self._verbose:
                print(f"   [live_tavily] cache hit: {len(hits)} result(s)")
            return hits

        if self._verbose:
            shown = full_query if len(full_query) <= 120 else full_query[:117] + "..."
            print(f'   [live_tavily] searching: "{shown}" (top {k})')
            if days_back is not None:
                print(f"   [live_tavily] recency window: last {days_back} days")

        try:
            results = self._post_search(full_query, k, days_back)
            self._note_success()
        except Exception as e:  # noqa: BLE001 -- never break play on retrieval
            self._note_failure(e)
            return []

        if not results:
            if self._verbose:
                print("   [live_tavily] no results returned")
            return []

        from polimillionaire.retrieval.retriever import Passage

        passages: list[Passage] = []
        for i, r in enumerate(results):
            content = (r.get("content") or "").strip()
            if not content:
                continue
            title = (r.get("title") or "").strip()
            url = r.get("url", "")
            domain = _domain_of(url)
            published = (r.get("published_date") or "")[:10]  # YYYY-MM-DD prefix only
            # Tavily returns its own relevance score in [0, 1]; use it
            # directly so the (no-op) downstream rank-pass-through keeps
            # the right order.
            tavily_score = float(r.get("score") or (1.0 - i * 0.05))
            passages.append(
                Passage(
                    id=f"tavily/{i}",
                    text=content,
                    metadata={
                        "source": "live_tavily",
                        "title": title,
                        "domain": domain,
                        "seendate": published,
                        "url": url,
                    },
                    score=tavily_score,
                )
            )

        self._cache[cache_key] = passages
        if self._verbose:
            print(f"   [live_tavily] fetched {len(passages)} result(s)")
        return passages

    def _post_search(self, query: str, k: int, days_back: int | None) -> list[dict]:
        body: dict = {
            "api_key": self._api_key,
            "query": query,
            "topic": self._topic,
            "search_depth": self._search_depth,
            "max_results": max(1, min(k, 20)),
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        if days_back is not None and self._topic == "news":
            body["days"] = min(max(days_back, 1), 365)
        resp = self._session.post(_API_URL, json=body, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", []) or []

    def _note_success(self) -> None:
        # A successful round-trip resets the breaker even if zero
        # results came back -- empty searches are valid signals.
        self._consecutive_failures = 0

    def _note_failure(self, exc: Exception) -> None:
        self._consecutive_failures += 1
        short = str(exc).splitlines()[0]
        if len(short) > 100:
            short = short[:97] + "..."
        print(f"   [live_tavily] search failed ({type(exc).__name__}: {short}); returning []")
        if self._consecutive_failures >= self._failure_threshold and not self._disabled:
            self._disabled = True
            print(
                f"   [live_tavily] circuit breaker open after "
                f"{self._failure_threshold} consecutive failures; skipping for the rest of the session"
            )
