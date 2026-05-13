"""One-off: compress on-disk FAISS indexes from flat fp32 to mmap'd IVF,PQ.

For each subdirectory under `data/index/` that has an `embeddings.npy`,
reads the vectors, trains an `IVF{nlist},PQ32` FAISS index, and writes
`faiss.index` next to the existing files. `Retriever` mmap-loads
`faiss.index` when present, dropping resident dense-index memory from
~2.3 GB (IndexFlatIP on 794k x 768 fp32) to ~25 MB.

Run once after building the raw indexes, or whenever embeddings.npy is
regenerated:

    uv run python scripts/compress_indexes.py            # all index dirs
    uv run python scripts/compress_indexes.py --force    # rebuild existing
    uv run python scripts/compress_indexes.py --root data/index/wiki_science

Quality: nprobe=32 (set as the saved default) gives ~95% recall@k on
800k-vector indexes vs IndexFlatIP. The reranker stage in WikiRagStrategy
compensates for the small recall loss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Tiny indexes (e.g. the 12.5k MATH corpus) don't benefit from IVF -- the
# minimum recommended training set size is ~39 * nlist, so under this
# threshold we skip and let the retriever fall back to IndexFlatIP.
_IVF_MIN_VECTORS = 40_000

# Default search-time nprobe baked into the saved index.
_DEFAULT_NPROBE = 32


def _index_dirs(root: Path) -> list[Path]:
    if (root / "embeddings.npy").exists():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "embeddings.npy").exists())


def _choose_factory(n: int, dim: int) -> tuple[str, int]:
    """Pick an IVF,PQ string and nlist appropriate for `n` vectors of `dim`.

    nlist scales ~4*sqrt(N), capped at 4096 so training stays under a minute
    on CPU. PQ m must divide dim; bge-base (768) uses m=32, math (384) m=16.
    """
    nlist = min(4096, max(256, int(4 * np.sqrt(n))))
    for m in (32, 16, 8, 4):
        if dim % m == 0:
            return f"IVF{nlist},PQ{m}", nlist
    raise ValueError(f"no PQ subquantizer count divides dim={dim}")


def compress(index_dir: Path, *, force: bool = False) -> None:
    import faiss

    embeddings_path = index_dir / "embeddings.npy"
    out_path = index_dir / "faiss.index"

    if out_path.exists() and not force:
        print(f"[skip] {index_dir.name}: faiss.index exists (use --force to rebuild)")
        return

    embeddings = np.load(embeddings_path, mmap_mode="r")
    if embeddings.ndim != 2:
        print(f"[skip] {index_dir.name}: embeddings.npy is not 2D")
        return
    n, dim = int(embeddings.shape[0]), int(embeddings.shape[1])

    if n < _IVF_MIN_VECTORS:
        print(
            f"[skip] {index_dir.name}: only {n} vectors -- below IVF threshold "
            f"({_IVF_MIN_VECTORS}); flat path is already cheap"
        )
        return

    factory, nlist = _choose_factory(n, dim)
    print(f"[{index_dir.name}] {n} x {dim} -> {factory}")

    index = faiss.index_factory(dim, factory, faiss.METRIC_INNER_PRODUCT)

    # Train on a uniform sample so the codebooks see the full distribution.
    # 39 * nlist is FAISS's recommended minimum; we go a bit higher for safety.
    train_size = min(n, max(50_000, 50 * nlist))
    rng = np.random.default_rng(0)
    train_idx = np.sort(rng.choice(n, size=train_size, replace=False))
    sample = np.ascontiguousarray(embeddings[train_idx], dtype=np.float32)
    print(f"[{index_dir.name}] training on {train_size} vectors ...")
    index.train(sample)
    del sample

    # Add in chunks so we never hold the full fp32 copy in memory.
    chunk = 50_000
    for start in range(0, n, chunk):
        block = np.ascontiguousarray(embeddings[start : start + chunk], dtype=np.float32)
        index.add(block)
        del block
        print(f"[{index_dir.name}] added {min(start + chunk, n)}/{n}")

    index.nprobe = _DEFAULT_NPROBE
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    faiss.write_index(index, str(tmp_path))
    tmp_path.replace(out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"[{index_dir.name}] wrote {out_path.name} ({size_mb:.1f} MB, nprobe={_DEFAULT_NPROBE})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--root",
        type=Path,
        default=_PROJECT_ROOT / "data" / "index",
        help="Index root (a single index dir or a parent containing several).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Rebuild faiss.index even when one already exists.",
    )
    args = ap.parse_args()

    if not args.root.exists():
        print(f"no index root at {args.root}", file=sys.stderr)
        sys.exit(1)

    dirs = _index_dirs(args.root)
    if not dirs:
        print(f"no index dirs with embeddings.npy under {args.root}", file=sys.stderr)
        sys.exit(1)

    for index_dir in dirs:
        compress(index_dir, force=args.force)


if __name__ == "__main__":
    main()
