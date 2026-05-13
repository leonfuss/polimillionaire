"""One-time: seed a private Kaggle Dataset from the local question DB.

After this runs, the dataset is the canonical store. From then on, the
notebook pulls it at session start and pushes a new version at session
end via `polimillionaire.kaggle_db.{pull_db, push_db}`.

Prerequisites:
    pip install kaggle
    # then place an API token at ~/.kaggle/kaggle.json
    # (Kaggle → Account → Create New Token)

Run:
    uv run python scripts/kaggle_db_init.py
    # creates `<kaggle_username>/polimillionaire-question-log`

To use a custom slug:
    POLIMILLIONAIRE_KAGGLE_DATASET=youraccount/some-slug \
        uv run python scripts/kaggle_db_init.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DB = _PROJECT_ROOT / "data" / "questions.sqlite"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=_DEFAULT_DB,
        help=f"path to the SQLite log (default: {_DEFAULT_DB})",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="full dataset slug `<owner>/<slug>` (default: "
        "POLIMILLIONAIRE_KAGGLE_DATASET env or `<kaggle_username>/polimillionaire-question-log`)",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="make the dataset public (default: private)",
    )
    args = parser.parse_args()

    if not args.db.exists():
        print(f"no DB at {args.db}; nothing to seed.", file=sys.stderr)
        return 1

    from polimillionaire.kaggle_db import create_db_dataset

    dataset_id = create_db_dataset(
        args.db,
        args.dataset_id,
        public=args.public,
    )
    print(f"\ncreated Kaggle Dataset: {dataset_id}")
    print(
        "to use in a Kaggle kernel: open the kernel → Add data → search "
        f"`{dataset_id}` → attach. It'll mount read-only at "
        f"/kaggle/input/{dataset_id.split('/')[-1]}/."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
