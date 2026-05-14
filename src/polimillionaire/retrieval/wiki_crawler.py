"""BFS Wikipedia category crawler that returns a set of article titles."""

from __future__ import annotations

import os
import time
from collections import deque

import requests

_DEFAULT_UA = "polimillionaire-rag/0.1 (NLP group project; mailto:leon.fuss@icloud.com)"
_RETRY_STATUSES = (429, 500, 502, 503, 504)


def _get_with_retry(
    session: requests.Session,
    url: str,
    params: dict,
    *,
    timeout: float = 30,
    max_attempts: int = 8,
    max_wait: float = 60.0,
) -> requests.Response:
    """GET with exponential backoff on transient errors. Crawls are long; a single
    429/503 should not kill the run.

    Wikipedia returns a `Retry-After` header on 429s -- we honour it when present
    (capped at `max_wait`) and otherwise fall back to 2**attempt with the same
    cap. From a Kaggle egress IP the limit kicks in fast, so the previous 4-attempt
    8s ceiling was nowhere near enough.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code in _RETRY_STATUSES and attempt < max_attempts - 1:
                wait = _retry_wait(resp, attempt, max_wait)
                print(
                    f"wiki_crawler: {resp.status_code} from API, retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_exc = e
            if attempt < max_attempts - 1:
                wait = min(2**attempt, max_wait)
                print(f"wiki_crawler: {type(e).__name__}, retrying in {wait:.1f}s")
                time.sleep(wait)
                continue
            raise
    # mypy: every path either returns or raises, but the loop boundary needs this
    raise RuntimeError(f"unreachable: get_with_retry exhausted attempts: {last_exc}")


def _retry_wait(resp: requests.Response, attempt: int, max_wait: float) -> float:
    """Prefer the server's `Retry-After` hint; fall back to exponential backoff."""
    hint = resp.headers.get("Retry-After")
    if hint:
        try:
            return min(float(hint), max_wait)
        except ValueError:
            pass  # HTTP-date form -- not worth parsing, fall through
    return min(2**attempt, max_wait)


def enumerate_category_titles(
    root_categories: list[str],
    *,
    max_depth: int = 2,
    max_titles: int | None = None,
    api_url: str = "https://en.wikipedia.org/w/api.php",
    user_agent: str = _DEFAULT_UA,
    request_delay: float = 0.1,
) -> set[str]:
    """BFS-style category walk; returns the unique set of page (mainspace) titles."""
    ua = os.environ.get("WIKI_USER_AGENT", user_agent)
    session = requests.Session()
    session.headers.update({"User-Agent": ua})

    titles: set[str] = set()
    # frontier entries: (category_name, depth)
    frontier: deque[tuple[str, int]] = deque((cat, 0) for cat in root_categories)
    visited_cats: set[str] = set(root_categories)
    processed = 0

    while frontier:
        if max_titles is not None and len(titles) >= max_titles:
            break

        cat_name, depth = frontier.popleft()
        processed += 1

        if processed % 50 == 0:
            print(f"wiki_crawler: depth={depth}, frontier={len(frontier)}, titles={len(titles)}")

        # paginate through all members of this category
        cmcontinue: str | None = None
        while True:
            params: dict[str, str | int] = {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": f"Category:{cat_name}",
                "cmlimit": 500,
                "cmtype": "page|subcat",
                "format": "json",
            }
            if cmcontinue is not None:
                params["cmcontinue"] = cmcontinue

            resp = _get_with_retry(session, api_url, params)
            data = resp.json()

            for member in data.get("query", {}).get("categorymembers", []):
                ns = member.get("ns", -1)
                title = member.get("title", "")
                if ns == 0:
                    # mainspace article
                    titles.add(title)
                    if max_titles is not None and len(titles) >= max_titles:
                        break
                elif ns == 14 and depth < max_depth:
                    # subcategory — strip the "Category:" prefix before queuing
                    subcat = title.removeprefix("Category:")
                    if subcat not in visited_cats:
                        visited_cats.add(subcat)
                        frontier.append((subcat, depth + 1))

            cont = data.get("continue", {})
            cmcontinue = cont.get("cmcontinue")
            # per-request, not per-category -- a large category with many
            # pagination pages waits between each of them.
            time.sleep(request_delay)

            if cmcontinue is None or (max_titles is not None and len(titles) >= max_titles):
                break

    print(f"wiki_crawler: done. processed={processed} categories, titles={len(titles)}")
    return titles
