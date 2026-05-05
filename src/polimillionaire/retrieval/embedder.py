"""Sentence-transformers wrapper with auto device selection.

The same `Embedder(model_name)` line works on Mac M-series (MPS), Colab
(CUDA), or plain CPU. Vectors are L2-normalised at encode time so the
retriever can use a single FAISS `IndexFlatIP` -- inner product on
normalised vectors equals cosine similarity, no renormalisation step.

Loading is lazy: instantiating `Embedder` does *not* import
sentence-transformers or pull weights. The first `encode()` triggers
the load. This keeps notebook startup fast and lets `polimillionaire`
import cleanly when the optional `[rag]` deps aren't installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def select_device() -> str:
    """Pick the fastest locally-available device. Order: cuda > mps > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """Lazy wrapper around a sentence-transformers model."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, device: str | None = None) -> None:
        self.name = model_name
        self.device = device or select_device()
        self._model: SentenceTransformer | None = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name, device=self.device)

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._model is not None
        # `get_embedding_dimension` is the post-5.x name; the older one
        # still exists but emits a FutureWarning. Try the new one first.
        getter = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        return int(getter())

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts as L2-normalised float32 vectors of shape (N, dim)."""
        self._ensure_loaded()
        assert self._model is not None
        out = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # sentence-transformers returns float32 already on most paths, but
        # FAISS IndexFlatIP demands float32 contiguous, so normalise dtype.
        return np.ascontiguousarray(out, dtype=np.float32)
