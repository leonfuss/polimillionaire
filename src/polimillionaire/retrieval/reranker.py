"""Cross-encoder reranker: re-scores retrieved passages against a query.

Loading is lazy: instantiating `Reranker` does not import sentence-transformers
or pull weights. The first `rerank()` call triggers the load, so notebooks and
strategy imports stay fast when the model isn't needed yet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from polimillionaire.retrieval.embedder import select_device
from polimillionaire.retrieval.retriever import Passage

DEFAULT_RERANKER = "BAAI/bge-reranker-v2-m3"


class Reranker:
    """Lazy wrapper around a sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = DEFAULT_RERANKER, *, device: str | None = None) -> None:
        self.name = model_name
        self.device = device or select_device()
        self._model: CrossEncoder | None = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            # On CUDA, run the cross-encoder in fp16. bge-reranker-v2-m3 is
            # XLM-RoBERTa-large (568M params); fp32 weights eat 2.27 GB of
            # VRAM, fp16 cuts that to 1.14 GB with no measurable quality
            # drop on ranking. MPS / CPU stay at default precision -- same
            # rationale as Embedder.
            kwargs: dict = {}
            if self.device == "cuda":
                import torch

                kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
            self._model = CrossEncoder(self.name, device=self.device, **kwargs)

    def rerank(
        self, query: str, passages: list[Passage], *, top_k: int | None = None
    ) -> list[Passage]:
        """Re-score `passages` with the cross-encoder against `query`.

        Returns the same passages sorted by reranker score (highest first), with the
        `score` field replaced by the reranker logit. `top_k=None` means return everything."""
        if not passages:
            return []

        self._ensure_loaded()
        pairs = [[query, p.text] for p in passages]
        logits = self._model.predict(pairs)  # type: ignore[union-attr]

        scored = sorted(
            zip(passages, logits, strict=True),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        if top_k is not None:
            scored = scored[:top_k]

        return [
            Passage(id=p.id, text=p.text, metadata=p.metadata, score=float(logit))
            for p, logit in scored
        ]
