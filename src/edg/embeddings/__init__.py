"""Batched, L2-normalized semantic embeddings with persistent caching and model comparison."""

from edg.embeddings.cache import EmbeddingCache, compute_text_hash
from edg.embeddings.embedder import TextEmbedder, compare_embedding_models, embed_interactions

__all__ = [
    "EmbeddingCache",
    "TextEmbedder",
    "compare_embedding_models",
    "compute_text_hash",
    "embed_interactions",
]
