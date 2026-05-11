"""Load article bodies from the HuggingFace wikimedia/wikipedia dump by title."""

from __future__ import annotations


def load_bodies_by_title(
    titles: set[str],
    *,
    dataset: str = "wikimedia/wikipedia",
    config: str = "20231101.en",
    show_progress: bool = True,
) -> dict[str, str]:
    """Stream the HF wikipedia dump and pull bodies for the requested titles.

    Returns a {title: body} dict. Titles missing from the dump are silently absent
    -- the caller treats those as crawler-only hits that need live-API fallback later.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    ds = load_dataset(dataset, config, split="train", streaming=True)

    if show_progress:
        from tqdm import tqdm  # type: ignore[import-untyped]

        it = tqdm(ds, desc="scanning wikipedia dump")
    else:
        it = iter(ds)

    # track remaining wanted titles separately -- the crawler regularly turns up
    # redirects and disambiguation pages that the dump doesn't index, so without
    # this every build would scan all ~6M rows.
    remaining = set(titles)
    out: dict[str, str] = {}
    for row in it:
        t = row["title"]
        if t in remaining:
            out[t] = row["text"]
            remaining.discard(t)
            if not remaining:
                break

    return out
