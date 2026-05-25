"""Per-question live news retrieval via GDELT DOC 2.0.

For news questions Wikipedia is the wrong source: it's slow to update,
opinionated about notability, and biased toward historical coverage.
GDELT indexes the global news firehose in near-real-time with no API
key required, which makes it the right primitive for the News
competition (cid 5).

Public surface mirrors `LiveWikiRetriever.search(query, k, *, option_texts)`
so the strategy is source-agnostic -- only the factory picks which one
to use per competition.

Hard contract (same as live_wiki): any HTTP or parsing failure returns
`[]` and logs the reason. A 429 / 503 / DNS hiccup must never abort an
answer; the LLM will fall back to its parametric knowledge.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import requests

from polimillionaire.retrieval.wiki_crawler import _DEFAULT_UA, _get_with_retry

if TYPE_CHECKING:
    from polimillionaire.retrieval.retriever import Passage

_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Stop set has three buckets:
#  - generic interrogatives / copulas / determiners (same as live_wiki)
#  - news-prompt scaffolding ("according to the article published on ...")
#    which is pure filler in every quiz question we saw
#  - calendar words that collapse to noise after date stripping
_QUERY_STOP_WORDS = frozenset(
    {
        # interrogatives
        "what", "which", "who", "whom", "when", "where", "why", "how",
        # copulas / auxiliaries
        "is", "are", "was", "were", "be", "been", "being",
        "do", "does", "did", "has", "have", "had",
        # articles / determiners
        "the", "a", "an", "this", "that", "these", "those",
        # prepositions + conjunctions
        "of", "in", "on", "for", "to", "with", "by", "at", "from", "as", "into",
        "and", "or", "but",
        # news prompt scaffolding -- always filler in the quiz set
        "according", "article", "articles", "news", "published",
        "report", "reports", "reported", "reporting",
        "story", "stories", "stated", "states",
        # calendar words
        "day", "days", "today", "yesterday", "tomorrow",
        "month", "year", "same",
    }
)  # fmt: skip
_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)

# YYYY-MM-DD anywhere in the question body. The quiz consistently anchors
# questions to a publication date in this format; capturing it gives us a
# free narrow time filter for GDELT.
_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
# Year-like tokens (2000-2099) and short numerics (1-2 digit) leak in
# from "2026 05 18" once the tokenizer breaks the hyphenated date apart.
# Strip them so they don't pad the keyword vector.
_DATE_NUM_RE = re.compile(r"^(?:20\d{2}|\d{1,2})$")
# Date bracket around the extracted publication date. ±2 days catches
# regional time-zone slop and articles republished a day later.
_DATE_WINDOW_DAYS = 2


def _extract_date_window(text: str) -> tuple[str, str] | None:
    """Return GDELT (startdatetime, enddatetime) bracketing a YYYY-MM-DD
    found in `text`, or None if no parseable date appears.

    GDELT expects the YYYYMMDDhhmmss format and treats the bracket as
    half-open inclusive on both ends.
    """
    m = _DATE_RE.search(text)
    if not m:
        return None
    try:
        d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None
    start = (d - timedelta(days=_DATE_WINDOW_DAYS)).strftime("%Y%m%d%H%M%S")
    end = (d + timedelta(days=_DATE_WINDOW_DAYS)).strftime("%Y%m%d%H%M%S")
    return start, end


def _clean_query(raw: str) -> str:
    """Strip stopwords + date-fragment tokens; return the keyword vector."""
    tokens = _TOKEN_RE.findall(raw)
    kept: list[str] = []
    for t in tokens:
        lower = t.lower()
        if lower in _QUERY_STOP_WORDS:
            continue
        if _DATE_NUM_RE.match(t):
            continue
        kept.append(t)
    return " ".join(kept)


def _gdelt_or_group(option_texts: list[str]) -> str:
    """Build `(opt1 OR "opt 2" OR opt3)` for GDELT's Boolean DSL.

    Quotes multi-word options so they're treated as exact phrases;
    drops options that collapse to empty after cleaning. Returns ""
    if nothing usable remains.
    """
    cleaned: list[str] = []
    seen: set[str] = set()
    for t in option_texts:
        c = _clean_query(t).strip()
        if not c or c.lower() in seen:
            continue
        seen.add(c.lower())
        cleaned.append(c)
    if not cleaned:
        return ""
    parts = [f'"{c}"' if " " in c else c for c in cleaned]
    return "(" + " OR ".join(parts) + ")"


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
        timeout: float = 20.0,
        timespan: str | None = None,
        source_lang: str | None = "eng",
        user_agent: str | None = None,
        verbose: bool = False,
        failure_threshold: int = 3,
    ) -> None:
        """
        `timespan` is a GDELT relative window like "1y" / "6m" / "7d".
        None (default) hits the full ~2015-present index ranked by
        relevance, which suits historical news questions; pass "1y" if
        the question set skews recent. Per-question date extraction
        (when the question text contains a YYYY-MM-DD) overrides this
        with a tight `startdatetime`/`enddatetime` window.

        `source_lang` filters by article language. "eng" by default --
        the quiz is in English so non-English hits are noise.

        `timeout` covers both TCP connect and HTTP read. GDELT is slow
        and erratic from some egress IPs, so we err on the generous
        side; the circuit breaker below protects the wall clock.

        `failure_threshold` arms a circuit breaker: after N consecutive
        network failures the retriever short-circuits to [] without
        hitting the network, so a flaky GDELT route doesn't burn a
        timeout's worth of game time on every subsequent question.
        """
        self._top_k = top_k
        self._timeout = timeout
        self._timespan = timespan
        self._source_lang = source_lang
        self._verbose = verbose
        ua = user_agent or os.environ.get("WIKI_USER_AGENT", _DEFAULT_UA)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": ua})
        # In-process cache keyed on the full effective query (incl. opts
        # + date window). Same rationale as live_wiki: live play sees few
        # duplicates within a game, but a kernel-restart retry will hit
        # this.
        self._cache: dict[str, list[Passage]] = {}
        # Circuit-breaker state. Trips after `failure_threshold` consecutive
        # network failures and stays open for the rest of the session --
        # callers can flip `_disabled` back if they want a manual reset.
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
        """Return up to `k` recent news headlines for `query`.

        `option_texts` are the answer-option strings, used as OR-grouped
        entity anchors in the GDELT query -- a question like "which
        senator lost" gets paired with `(Cassidy OR Cruz OR McCain OR
        Rubio)`, vastly improving recall on entity-disambiguation
        questions. Pass None to skip the anchor.

        On any failure (network, JSON, missing fields), logs and returns
        `[]`. Never propagates exceptions to the caller.
        """
        k = k or self._top_k

        if self._disabled:
            if self._verbose:
                print("   [live_gdelt] circuit breaker open; skipping network call")
            return []

        date_window = _extract_date_window(query)
        keywords = _clean_query(query)
        or_group = _gdelt_or_group(option_texts) if option_texts else ""

        # Assemble the GDELT query body. We must always have at least one
        # positive term -- a query of only `sourcelang:eng` is rejected.
        body_parts: list[str] = []
        if keywords:
            body_parts.append(keywords)
        if or_group:
            body_parts.append(or_group)
        if not body_parts:
            if self._verbose:
                print("   [live_gdelt] query collapsed to empty after cleaning; returning []")
            return []
        body_parts.append(f"sourcelang:{self._source_lang}") if self._source_lang else None
        gdelt_query = " ".join(p for p in body_parts if p)

        # Cache key includes the date window so two questions about the
        # same topic but different dates don't share a stale answer.
        cache_key = f"{gdelt_query}||{date_window or ''}"
        if cache_key in self._cache:
            hits = self._cache[cache_key][:k]
            if self._verbose:
                print(f"   [live_gdelt] cache hit: {len(hits)} article(s) for query")
            return hits

        if self._verbose:
            shown = gdelt_query if len(gdelt_query) <= 120 else gdelt_query[:117] + "..."
            print(f'   [live_gdelt] searching: "{shown}" (top {k})')
            if date_window:
                print(f"   [live_gdelt] date window: {date_window[0]} .. {date_window[1]}")

        try:
            articles = self._search_articles(gdelt_query, k, date_window)
            self._note_success()
        except Exception as e:  # noqa: BLE001 -- never break play on retrieval
            self._note_failure(e)
            return []
        # Fallback: a date-windowed query that finds nothing is often a
        # publication-lag artifact (article shows up under a different
        # date). Retry once without the window before giving up. Don't
        # let this re-trip the breaker -- a successful empty search is
        # not a network failure.
        if not articles and date_window is not None:
            if self._verbose:
                print("   [live_gdelt] no articles in date window; retrying without window")
            try:
                articles = self._search_articles(gdelt_query, k, None)
            except Exception as e:  # noqa: BLE001
                self._note_failure(e, label="fallback")
                articles = []
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

        self._cache[cache_key] = passages
        if self._verbose:
            print(f"   [live_gdelt] fetched {len(passages)} article(s)")
        return passages

    def _search_articles(
        self,
        query: str,
        k: int,
        date_window: tuple[str, str] | None,
    ) -> list[dict]:
        params: dict[str, str] = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": str(max(1, min(k, 250))),
            "sort": "HybridRel",
        }
        if date_window is not None:
            params["startdatetime"], params["enddatetime"] = date_window
        elif self._timespan:
            params["timespan"] = self._timespan

        # Single timeout covering both connect + read. An earlier split
        # (connect_timeout=5, read=15) failed in user-reported runs --
        # the connect ceiling was too aggressive for high-latency egress
        # to GDELT and tripped every call. max_attempts=1: GDELT is slow
        # enough that retrying blows past the 30s game timer; failing
        # to [] lets the LLM answer from parametric knowledge.
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

    def _note_success(self) -> None:
        # A successful network round-trip resets the breaker, even if
        # the response had zero articles -- that's a successful empty
        # search, not a connectivity failure.
        self._consecutive_failures = 0

    def _note_failure(self, exc: Exception, *, label: str = "") -> None:
        self._consecutive_failures += 1
        prefix = f"{label} " if label else ""
        # Truncate the requests/urllib3 message -- the full one dumps
        # the URL + nested exception chain and dominates the game log.
        short = str(exc).splitlines()[0]
        if len(short) > 100:
            short = short[:97] + "..."
        print(
            f"   [live_gdelt] {prefix}search failed ({type(exc).__name__}: {short})"
            f"; returning []"
        )
        if self._consecutive_failures >= self._failure_threshold and not self._disabled:
            self._disabled = True
            print(
                f"   [live_gdelt] circuit breaker open after "
                f"{self._failure_threshold} consecutive failures; skipping for the rest of the session"
            )
