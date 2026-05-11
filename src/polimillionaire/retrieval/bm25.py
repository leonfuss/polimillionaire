"""BM25 sparse retrieval over tokenised passage text.

Complement to the dense FAISS retriever -- BM25 handles entity-heavy queries
(proper nouns, exact tokens) much better than embedding-based similarity.
Pairs with `fusion.reciprocal_rank_fusion` for hybrid retrieval.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from polimillionaire.retrieval.retriever import Passage

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi

_TOKENS_FILE = "bm25_tokens.jsonl"
_PARAMS_FILE = "bm25.json"


def _tokenize(text: str) -> list[str]:
    # plain word-char split keeps exact-token matching on entities like
    # "Kallinikos" or "Thanos" -- stemming would smear those.
    return re.findall(r"\w+", text.lower())


class BM25Index:
    """BM25Okapi index over a list of passages."""

    def __init__(
        self,
        bm25: BM25Okapi,
        passages: list[dict[str, Any]],
        tokenized: list[list[str]],
    ) -> None:
        self._bm25 = bm25
        self._passages = passages
        self._tokenized = tokenized

    @classmethod
    def build(cls, passages: list[dict[str, Any]]) -> BM25Index:
        """Build an in-memory index from a list of passage dicts."""
        from rank_bm25 import BM25Okapi

        tokenized = [_tokenize(p["text"]) for p in passages]
        bm25 = BM25Okapi(tokenized)
        return cls(bm25, list(passages), tokenized)

    def save(self, index_dir: Path) -> None:
        """Write tokens and BM25 params to `index_dir`."""
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        tokens_lines = [json.dumps(toks) for toks in self._tokenized]
        (index_dir / _TOKENS_FILE).write_text("\n".join(tokens_lines))

        params = {
            "k1": self._bm25.k1,
            "b": self._bm25.b,
            "epsilon": self._bm25.epsilon,
            "count": len(self._passages),
        }
        (index_dir / _PARAMS_FILE).write_text(json.dumps(params))

    @classmethod
    def load(cls, index_dir: Path, passages: list[dict[str, Any]] | None = None) -> BM25Index:
        """Rebuild a BM25Index from disk.

        Reads `passages.jsonl` from `index_dir` by default -- shared with the
        dense retriever's on-disk format. Tests can pass an explicit `passages`
        list to avoid writing the file.
        """
        from rank_bm25 import BM25Okapi

        index_dir = Path(index_dir)
        tokens_path = index_dir / _TOKENS_FILE
        params_path = index_dir / _PARAMS_FILE
        for p in (tokens_path, params_path):
            if not p.exists():
                raise FileNotFoundError(f"bm25 index at {index_dir} is missing {p.name}")

        if passages is None:
            passages_path = index_dir / "passages.jsonl"
            if not passages_path.exists():
                raise FileNotFoundError(
                    f"bm25 index at {index_dir} expects passages.jsonl alongside it"
                )
            passages = [
                json.loads(line) for line in passages_path.read_text().splitlines() if line.strip()
            ]

        tokenized: list[list[str]] = [
            json.loads(line) for line in tokens_path.read_text().splitlines() if line.strip()
        ]
        params = json.loads(params_path.read_text())

        if len(tokenized) != params["count"]:
            raise ValueError(f"token count {len(tokenized)} != expected {params['count']}")
        if len(passages) != params["count"]:
            raise ValueError(
                f"passages list length {len(passages)} != index count {params['count']}"
            )

        # `epsilon` was added later -- fall back to BM25Okapi's default for old indexes.
        kwargs: dict[str, Any] = {"k1": params["k1"], "b": params["b"]}
        if "epsilon" in params:
            kwargs["epsilon"] = params["epsilon"]
        bm25 = BM25Okapi(tokenized, **kwargs)
        return cls(bm25, list(passages), tokenized)

    def search(self, query: str, k: int) -> list[Passage]:
        """Return top-k passages by BM25 score. Score is raw (not normalised)."""
        if k <= 0:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        out: list[Passage] = []
        for idx in top_indices:
            row = self._passages[idx]
            out.append(
                Passage(
                    id=row["id"],
                    text=row["text"],
                    metadata=row.get("metadata", {}),
                    score=float(scores[idx]),
                )
            )
        return out
