"""Build dense + BM25 indexes for the three non-math Wikipedia competitions.

Two-phase pipeline so the heavy CPU work (Wikipedia crawl, HF dump scan,
chunking, BM25) can run on a free CPU runtime and only the embedding step
needs a GPU. Each step's output is cached on disk and skipped on re-run,
so an interrupted build resumes where it stopped.

    # Phase 1 -- CPU runtime, no GPU quota used.
    # Outputs _titles.json, _bodies.jsonl, passages.jsonl, bm25_*.
    uv run python scripts/build_wiki_index.py --all --phase stage \\
        --output-dir /content/drive/MyDrive/PoliMillionaire/index

    # Phase 2 -- GPU runtime. Reads passages.jsonl, writes embeddings.npy.
    uv run python scripts/build_wiki_index.py --all --phase embed \\
        --output-dir /content/drive/MyDrive/PoliMillionaire/index

    # Default (--phase all) runs both sequentially -- fine for local builds.
    uv run python scripts/build_wiki_index.py --competition 0

Other useful flags:
    --max-titles 500    cap the crawl for a quick dry run
    --refresh           ignore cached intermediates and rebuild from scratch
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

# Intermediate file names. Leading underscore = staging artifact (consumed by
# this script), no underscore = index artifact (consumed by the retriever).
TITLES_FILE = "_titles.json"
BODIES_FILE = "_bodies.jsonl"
PASSAGES_FILE = "passages.jsonl"
EMBEDDINGS_FILE = "embeddings.npy"
MANIFEST_FILE = "manifest.json"
BM25_TOKENS_FILE = "bm25_tokens.jsonl"
BM25_PARAMS_FILE = "bm25.json"


def _atomic_write(path: Path, content: str) -> None:
    """Write text via a sibling .tmp file then rename, so a killed run can't
    leave a half-written file that the next run mistakes for complete."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    _atomic_write(path, "\n".join(json.dumps(r) for r in rows) + "\n")


def _stage_one(seed, args: argparse.Namespace) -> None:
    """CPU phase: crawl titles, pull bodies from the HF dump, chunk, build BM25.

    Each step writes its output to `<out_dir>/<intermediate>` and is skipped
    on re-run if the file already exists (unless `--refresh` is set).
    """
    from polimillionaire.retrieval.bm25 import BM25Index
    from polimillionaire.retrieval.wiki_chunker import chunk_article
    from polimillionaire.retrieval.wiki_crawler import enumerate_category_titles
    from polimillionaire.retrieval.wiki_dump import load_bodies_by_title

    out_dir = Path(args.output_dir) / seed.name
    out_dir.mkdir(parents=True, exist_ok=True)

    titles_path = out_dir / TITLES_FILE
    bodies_path = out_dir / BODIES_FILE
    passages_path = out_dir / PASSAGES_FILE
    bm25_tokens_path = out_dir / BM25_TOKENS_FILE

    print(f"\n=== stage {seed.name} (competition_id={seed.competition_id}) ===")

    # --- 1. crawl titles ---
    if titles_path.exists() and not args.refresh:
        titles = set(json.loads(titles_path.read_text()))
        print(f"[1/4] titles: reusing {len(titles)} from {titles_path}")
    else:
        max_titles = args.max_titles if args.max_titles > 0 else None
        print(f"[1/4] titles: crawling {len(seed.categories)} root categories...")
        titles = enumerate_category_titles(
            seed.categories,
            max_depth=2,
            max_titles=max_titles,
        )
        _atomic_write(titles_path, json.dumps(sorted(titles)))
        print(f"      saved {len(titles)} -> {titles_path}")

    # --- 2. scan HF wikipedia dump for bodies ---
    if bodies_path.exists() and not args.refresh:
        rows = _read_jsonl(bodies_path)
        bodies = {r["title"]: r["body"] for r in rows}
        print(f"[2/4] bodies: reusing {len(bodies)} from {bodies_path}")
    else:
        print("[2/4] bodies: scanning HF wikipedia dump (one-shot, ~6M rows)...")
        bodies = load_bodies_by_title(titles)
        _write_jsonl(
            bodies_path,
            [{"title": t, "body": b} for t, b in bodies.items()],
        )
        print(f"      saved {len(bodies)} (of {len(titles)} crawled) -> {bodies_path}")

    # --- 3. chunk articles ---
    passages: list[dict]
    if passages_path.exists() and not args.refresh:
        passages = _read_jsonl(passages_path)
        print(f"[3/4] chunks: reusing {len(passages)} from {passages_path}")
    else:
        print(f"[3/4] chunks: paragraph-packing {len(bodies)} articles...")
        passages = []
        for title, body in bodies.items():
            for chunk in chunk_article(title, body):
                passages.append(
                    {
                        "id": chunk.id,
                        "text": chunk.text,
                        "metadata": chunk.metadata,
                    }
                )
        if not passages:
            print("      no passages produced; skipping index write.", file=sys.stderr)
            return
        _write_jsonl(passages_path, passages)
        print(f"      saved {len(passages)} -> {passages_path}")

    # --- 4. BM25 sparse index (CPU, ~minutes) ---
    if bm25_tokens_path.exists() and not args.refresh:
        print(f"[4/4] bm25: reusing index at {out_dir}")
    else:
        print(f"[4/4] bm25: tokenising + building over {len(passages)} passages...")
        bm25 = BM25Index.build(passages)
        bm25.save(out_dir)
        print(f"      saved -> {bm25_tokens_path.name}, {BM25_PARAMS_FILE}")

    print(f"stage done for {seed.name}.")


def _embed_one(seed, args: argparse.Namespace) -> None:
    """GPU phase: load cached passages, embed, write manifest.

    Skips entirely if both `embeddings.npy` and `manifest.json` are present
    and `--refresh` is not set.
    """
    from polimillionaire.retrieval.embedder import Embedder

    out_dir = Path(args.output_dir) / seed.name
    passages_path = out_dir / PASSAGES_FILE
    embeddings_path = out_dir / EMBEDDINGS_FILE
    manifest_path = out_dir / MANIFEST_FILE

    print(f"\n=== embed {seed.name} (competition_id={seed.competition_id}) ===")

    if not passages_path.exists():
        raise FileNotFoundError(
            f"{passages_path} missing — run `--phase stage` for this competition first."
        )

    if embeddings_path.exists() and manifest_path.exists() and not args.refresh:
        print(f"embeddings + manifest already present at {out_dir}; skipping.")
        return

    passages = _read_jsonl(passages_path)
    print(f"loading embedder {args.embedder}...")
    emb = Embedder(args.embedder)
    print(f"  device={emb.device}, dim={emb.dim}")

    print(f"embedding {len(passages)} chunks (batch_size={BATCH_SIZE})...")
    embeddings = emb.encode(
        [p["text"] for p in passages],
        batch_size=BATCH_SIZE,
        show_progress=True,
    )
    print(f"  embeddings shape {embeddings.shape}")

    np.save(embeddings_path, embeddings)

    # populate the descriptive fields from cached intermediates if present
    titles_path = out_dir / TITLES_FILE
    bodies_path = out_dir / BODIES_FILE
    n_titles: int | None = None
    n_bodies: int | None = None
    if titles_path.exists():
        n_titles = len(json.loads(titles_path.read_text()))
    if bodies_path.exists():
        n_bodies = sum(1 for line in bodies_path.read_text().splitlines() if line.strip())

    manifest = {
        "model_name": args.embedder,
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "competition_id": seed.competition_id,
        "competition_name": seed.name,
        "crawled_titles": n_titles,
        "matched_titles": n_bodies,
        "dataset": "wikimedia/wikipedia:20231101.en",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"embed done for {seed.name}. index at {out_dir}.")


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
        "--phase",
        choices=["stage", "embed", "all"],
        default="all",
        help=(
            "stage = CPU work (crawl, scan, chunk, BM25). "
            "embed = GPU work (encode + manifest). "
            "all = both, default."
        ),
    )
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
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="ignore cached intermediates and rebuild every step from scratch",
    )
    args = parser.parse_args()

    from polimillionaire.retrieval.wiki_seeds import SEEDS

    ids_to_build = [0, 1, 2] if args.all else [args.competition]

    for cid in ids_to_build:
        seed = SEEDS[cid]
        if args.phase in ("stage", "all"):
            _stage_one(seed, args)
        if args.phase in ("embed", "all"):
            _embed_one(seed, args)

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
