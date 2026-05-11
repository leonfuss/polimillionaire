"""Retrieval primitives for RAG strategies: dense (FAISS) + sparse (BM25) + fusion.

Wrappers around `sentence-transformers` + FAISS that hide the device-
selection and on-disk-format details from strategy code. Imported lazily
inside functions where possible so the optional `[rag]` deps don't break
import of the rest of the package.
"""

from polimillionaire.retrieval.bm25 import BM25Index
from polimillionaire.retrieval.embedder import DEFAULT_MODEL, Embedder, select_device
from polimillionaire.retrieval.fusion import reciprocal_rank_fusion
from polimillionaire.retrieval.reranker import DEFAULT_RERANKER, Reranker
from polimillionaire.retrieval.retriever import Passage, Retriever
from polimillionaire.retrieval.wiki_chunker import WikiChunk, chunk_article
from polimillionaire.retrieval.wiki_crawler import enumerate_category_titles
from polimillionaire.retrieval.wiki_dump import load_bodies_by_title
from polimillionaire.retrieval.wiki_seeds import SEEDS, CompetitionSeed

__all__ = [
    "BM25Index",
    "CompetitionSeed",
    "DEFAULT_MODEL",
    "DEFAULT_RERANKER",
    "Embedder",
    "Passage",
    "Reranker",
    "Retriever",
    "SEEDS",
    "WikiChunk",
    "chunk_article",
    "enumerate_category_titles",
    "load_bodies_by_title",
    "reciprocal_rank_fusion",
    "select_device",
]
