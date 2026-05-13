"""Build a dense retrieval index over the Hendrycks MATH dataset.

Downloads the MATH problems from HuggingFace, embeds the problem text
(not the solution -- we retrieve by problem similarity, then the
solution is shown to the model as the reference), and saves the index
under data/index/math/.

By default also augments the corpus with Wikipedia math articles
(abstract algebra, statistics, etc. -- see `MATH_WIKI_CATEGORIES`) to
plug topic gaps the MATH dataset doesn't cover (group theory, Sylow
theorems, real statistics). The two sub-corpora share one passages.jsonl
and one FAISS index; each row carries `metadata.source` so the prompt
formatter can render problem-solution pairs and wiki chunks differently.

Run:

    uv sync --group rag
    uv run python scripts/build_math_index.py             # MATH + wiki
    uv run python scripts/build_math_index.py --no-wiki   # MATH only
    uv run python scripts/build_math_index.py --wiki-only # extend existing

The dataset name can be swapped at the top -- a few mirrors exist
(`hendrycks/competition_math`, `lighteval/MATH`, `qwedsacf/competition_math`)
with the same {problem, level, type, solution} schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from polimillionaire.retrieval.embedder import DEFAULT_MODEL, Embedder

# scripts/ -> repo root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Tried in order; first one that loads wins. Lighteval's mirror tends to
# be the most reliably available; the original Hendrycks repo has had
# intermittent access issues over the project's lifetime.
DATASET_CANDIDATES: list[str] = [
    "lighteval/MATH",
    "hendrycks/competition_math",
    "qwedsacf/competition_math",
]
MODEL_NAME = DEFAULT_MODEL  # bge-small-en-v1.5 (~50 MB, 384 dim)
OUT_DIR = _PROJECT_ROOT / "data" / "index" / "math"
BATCH_SIZE = 64

# Staging files for the wiki side (cached on disk so an interrupted run
# can resume without re-crawling). Match the naming used by
# build_wiki_index.py so a future operator can tell the artifacts apart.
WIKI_TITLES_FILE = "_wiki_titles.json"
WIKI_BODIES_FILE = "_wiki_bodies.jsonl"


def _load_math_dataset():
    from datasets import load_dataset

    last_err: Exception | None = None
    for name in DATASET_CANDIDATES:
        try:
            print(f"loading dataset {name}...")
            return name, load_dataset(name)
        except Exception as e:  # noqa: BLE001 -- fall through to next mirror
            last_err = e
            print(f"  -> {type(e).__name__}: {e}")
    raise RuntimeError(
        "no MATH dataset mirror available. Tried: "
        + ", ".join(DATASET_CANDIDATES)
        + f". Last error: {last_err}"
    )


def _load_math_problem_rows() -> tuple[str | None, list[dict]]:
    dataset_name, splits = _load_math_dataset()
    rows: list[dict] = []
    for split_name, split in splits.items():
        for i, ex in enumerate(split):
            problem = ex.get("problem")
            solution = ex.get("solution", "")
            if not problem:
                continue
            rows.append(
                {
                    "id": f"{split_name}/{i}",
                    "text": problem,
                    "metadata": {
                        "source": "math_problems",
                        "split": split_name,
                        "subject": ex.get("type", "unknown"),
                        "level": ex.get("level", "unknown"),
                        "solution": solution,
                    },
                }
            )
    return dataset_name, rows


def _load_math_wiki_rows(
    out_dir: Path,
    *,
    max_titles: int,
    refresh: bool,
) -> list[dict]:
    """Crawl math Wikipedia categories, pull bodies, chunk, return rows.

    Caches the title list and body dump under `out_dir` so a partial run
    resumes without re-hitting the wiki API.
    """
    from polimillionaire.retrieval.wiki_chunker import chunk_article
    from polimillionaire.retrieval.wiki_crawler import enumerate_category_titles
    from polimillionaire.retrieval.wiki_dump import load_bodies_by_title
    from polimillionaire.retrieval.wiki_seeds import MATH_WIKI_CATEGORIES

    titles_path = out_dir / WIKI_TITLES_FILE
    bodies_path = out_dir / WIKI_BODIES_FILE

    if titles_path.exists() and not refresh:
        titles = set(json.loads(titles_path.read_text()))
        print(f"[wiki 1/3] titles: reusing {len(titles)} from {titles_path.name}")
    else:
        cap = max_titles if max_titles > 0 else None
        print(f"[wiki 1/3] titles: crawling {len(MATH_WIKI_CATEGORIES)} categories...")
        titles = enumerate_category_titles(
            MATH_WIKI_CATEGORIES,
            max_depth=2,
            max_titles=cap,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        titles_path.write_text(json.dumps(sorted(titles)))
        print(f"           saved {len(titles)} -> {titles_path.name}")

    if bodies_path.exists() and not refresh:
        bodies = {
            r["title"]: r["body"]
            for r in (
                json.loads(line) for line in bodies_path.read_text().splitlines() if line.strip()
            )
        }
        print(f"[wiki 2/3] bodies: reusing {len(bodies)} from {bodies_path.name}")
    else:
        print("[wiki 2/3] bodies: scanning HF wikipedia dump (one-shot, ~6M rows)...")
        bodies = load_bodies_by_title(titles)
        out_dir.mkdir(parents=True, exist_ok=True)
        with bodies_path.open("w") as f:
            for t, b in bodies.items():
                f.write(json.dumps({"title": t, "body": b}) + "\n")
        print(f"           saved {len(bodies)} (of {len(titles)} crawled) -> {bodies_path.name}")

    print(f"[wiki 3/3] chunks: paragraph-packing {len(bodies)} articles...")
    rows: list[dict] = []
    for title, body in bodies.items():
        for chunk in chunk_article(title, body):
            rows.append(
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": {
                        "source": "math_wiki",
                        "title": chunk.metadata.get("title", title),
                        "url": chunk.metadata.get("url"),
                    },
                }
            )
    print(f"           produced {len(rows)} chunks")
    return rows


def _embed_and_save(
    rows: list[dict],
    *,
    out_dir: Path,
    dataset_name: str | None,
    batch_size: int,
) -> None:
    print(f"loading embedder {MODEL_NAME}...")
    emb = Embedder(MODEL_NAME)
    print(f"  -> device={emb.device}, dim={emb.dim}")

    print(f"embedding {len(rows)} passages (batch_size={batch_size})...")
    embeddings = emb.encode(
        [r["text"] for r in rows],
        batch_size=batch_size,
        show_progress=True,
    )
    print(f"  -> embeddings shape {embeddings.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", embeddings)
    with (out_dir / "passages.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_problems = sum(1 for r in rows if r["metadata"].get("source") == "math_problems")
    n_wiki = sum(1 for r in rows if r["metadata"].get("source") == "math_wiki")
    manifest = {
        "model_name": MODEL_NAME,
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "dataset": dataset_name,
        "sources": {"math_problems": n_problems, "math_wiki": n_wiki},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved index to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(OUT_DIR),
        help=f"output directory (default: {OUT_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"embedding batch size (default: {BATCH_SIZE})",
    )
    wiki_group = parser.add_mutually_exclusive_group()
    wiki_group.add_argument(
        "--no-wiki",
        action="store_true",
        help="skip the math-wiki augmentation; build only MATH problems",
    )
    wiki_group.add_argument(
        "--wiki-only",
        action="store_true",
        help="skip MATH problems; build only the wiki augmentation "
        "(useful for extending an existing problem-only index)",
    )
    parser.add_argument(
        "--max-wiki-titles",
        type=int,
        default=0,
        help="cap math-wiki title crawl for a quick dry run (0 = no cap)",
    )
    parser.add_argument(
        "--refresh-wiki",
        action="store_true",
        help="ignore cached wiki staging artifacts and re-crawl from scratch",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    dataset_name: str | None = None
    problem_rows: list[dict] = []
    if not args.wiki_only:
        dataset_name, problem_rows = _load_math_problem_rows()
        print(f"  -> {len(problem_rows)} MATH problems")

    wiki_rows: list[dict] = []
    if not args.no_wiki:
        wiki_rows = _load_math_wiki_rows(
            out_dir,
            max_titles=args.max_wiki_titles,
            refresh=args.refresh_wiki,
        )

    rows = problem_rows + wiki_rows
    if not rows:
        print("no rows to embed; aborting.", file=sys.stderr)
        return 1

    _embed_and_save(
        rows,
        out_dir=out_dir,
        dataset_name=dataset_name,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
