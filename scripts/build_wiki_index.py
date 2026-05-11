"""Build dense + BM25 indexes for the three non-math Wikipedia competitions.

One index per competition, saved under <output_dir>/<slug>/:
    manifest.json, embeddings.npy, passages.jsonl, bm25_tokens.jsonl, bm25.json

Run one competition at a time (useful on Colab):

    uv run python scripts/build_wiki_index.py --competition 0
    uv run python scripts/build_wiki_index.py --competition 1
    uv run python scripts/build_wiki_index.py --competition 2
    uv run python scripts/build_wiki_index.py --all

Dry run with a small title cap:

    uv run python scripts/build_wiki_index.py --competition 0 --max-titles 500

Custom output directory (e.g. Colab Drive):

    uv run python scripts/build_wiki_index.py --competition 0 \\
        --output-dir /content/drive/MyDrive/PoliMillionaire/index
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_EMBEDDER = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 64


def _build_one(seed, args: argparse.Namespace) -> None:
    # imports are deferred so `--help` doesn't pay the cost of pulling
    # sentence-transformers / datasets / requests before argparse exits.
    from polimillionaire.retrieval.bm25 import BM25Index
    from polimillionaire.retrieval.embedder import Embedder
    from polimillionaire.retrieval.wiki_chunker import chunk_article
    from polimillionaire.retrieval.wiki_crawler import enumerate_category_titles
    from polimillionaire.retrieval.wiki_dump import load_bodies_by_title

    out_dir = Path(args.output_dir) / seed.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {seed.name} (competition_id={seed.competition_id}) ===")
    print(f"crawling {len(seed.categories)} root categories (max_depth=2)...")
    max_titles = args.max_titles if args.max_titles > 0 else None
    titles = enumerate_category_titles(
        seed.categories,
        max_depth=2,
        max_titles=max_titles,
    )
    print(f"  -> {len(titles)} titles from category walk")

    print("loading bodies from HuggingFace wikipedia dump...")
    bodies = load_bodies_by_title(titles)
    print(f"  -> {len(bodies)} titles matched in dump (out of {len(titles)} crawled)")

    print("chunking articles...")
    passages: list[dict] = []
    for title, body in bodies.items():
        for chunk in chunk_article(title, body):
            passages.append(
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
            )
    print(f"  -> {len(passages)} chunks total")

    if not passages:
        print("no passages produced; skipping index write.", file=sys.stderr)
        return

    print(f"loading embedder {args.embedder}...")
    emb = Embedder(args.embedder)
    print(f"  -> device={emb.device}, dim={emb.dim}")

    print(f"embedding {len(passages)} chunks (batch_size={BATCH_SIZE})...")
    embeddings = emb.encode(
        [p["text"] for p in passages],
        batch_size=BATCH_SIZE,
        show_progress=True,
    )
    print(f"  -> embeddings shape {embeddings.shape}")

    np.save(out_dir / "embeddings.npy", embeddings)
    with (out_dir / "passages.jsonl").open("w") as f:
        for p in passages:
            f.write(json.dumps(p) + "\n")

    manifest = {
        "model_name": args.embedder,
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "competition_id": seed.competition_id,
        "competition_name": seed.name,
        "crawled_titles": len(titles),
        "matched_titles": len(bodies),
        "dataset": "wikimedia/wikipedia:20231101.en",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("building BM25 index...")
    bm25 = BM25Index.build(passages)
    bm25.save(out_dir)

    print(f"saved index to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Wikipedia RAG indexes.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--competition",
        type=int,
        choices=[0, 1, 2],
        help="competition ID to build (0=entertainment, 1=history, 2=science)",
    )
    group.add_argument("--all", action="store_true", help="build all three competitions")
    parser.add_argument(
        "--output-dir",
        default=str(_PROJECT_ROOT / "data" / "index"),
        help="root output directory (default: <repo>/data/index)",
    )
    parser.add_argument(
        "--embedder",
        default=DEFAULT_EMBEDDER,
        help=f"sentence-transformers model name (default: {DEFAULT_EMBEDDER})",
    )
    parser.add_argument(
        "--max-titles",
        type=int,
        default=0,
        help="cap title count for a quick dry run (0 = no cap)",
    )
    args = parser.parse_args()

    from polimillionaire.retrieval.wiki_seeds import SEEDS

    ids_to_build = [0, 1, 2] if args.all else [args.competition]

    for cid in ids_to_build:
        seed = SEEDS[cid]
        _build_one(seed, args)

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
