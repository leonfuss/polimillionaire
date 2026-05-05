"""Build a dense retrieval index over the Hendrycks MATH dataset.

Downloads the MATH problems from HuggingFace, embeds the problem text
(not the solution -- we retrieve by problem similarity, then the
solution is shown to the model as the reference), and saves the index
under data/index/math/.

This is a one-off per machine. The resulting directory is small
(~25 MB) and rsync-able to Colab so a slow-network laptop can build
once and copy.

Run:

    uv sync --group rag
    uv run python scripts/build_math_index.py

The dataset name can be swapped at the top -- a few mirrors exist
(`hendrycks/competition_math`, `lighteval/MATH`, `qwedsacf/competition_math`)
with the same {problem, level, type, solution} schema.
"""

from __future__ import annotations

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


def main() -> int:
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
                        "split": split_name,
                        "subject": ex.get("type", "unknown"),
                        "level": ex.get("level", "unknown"),
                        "solution": solution,
                    },
                }
            )
    print(f"  -> {len(rows)} problems across splits {list(splits.keys())}")
    if not rows:
        print("dataset returned no usable rows; aborting.", file=sys.stderr)
        return 1

    print(f"loading embedder {MODEL_NAME}...")
    emb = Embedder(MODEL_NAME)
    print(f"  -> device={emb.device}, dim={emb.dim}")

    print(f"embedding {len(rows)} problems (batch_size={BATCH_SIZE})...")
    embeddings = emb.encode(
        [r["text"] for r in rows],
        batch_size=BATCH_SIZE,
        show_progress=True,
    )
    print(f"  -> embeddings shape {embeddings.shape}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "embeddings.npy", embeddings)
    with (OUT_DIR / "passages.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    manifest = {
        "model_name": MODEL_NAME,
        "dim": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "dataset": dataset_name,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved index to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
