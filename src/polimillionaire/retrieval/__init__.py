"""Dense retrieval primitives for RAG strategies.

Wrappers around `sentence-transformers` + FAISS that hide the device-
selection and on-disk-format details from strategy code. Imported lazily
inside functions where possible so the optional `[rag]` deps don't break
import of the rest of the package.
"""

from polimillionaire.retrieval.embedder import DEFAULT_MODEL, Embedder, select_device
from polimillionaire.retrieval.retriever import Passage, Retriever

__all__ = ["DEFAULT_MODEL", "Embedder", "Passage", "Retriever", "select_device"]
