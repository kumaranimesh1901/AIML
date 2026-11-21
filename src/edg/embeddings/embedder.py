"""Semantic text embedder with batching, normalization, caching, and model comparison."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.config import AppConfig
from edg.db.models import Embedding, InteractionClean, InteractionRaw
from edg.embeddings.cache import EmbeddingCache, compute_text_hash
from edg.logging import get_logger
from edg.sut.failure_injector import CONTROLLED_SOURCE_TAG
from edg.world.policies import hash_embed_texts

logger = get_logger(__name__)


class TextEmbedder:
    """Batched, L2-normalized embedder supporting sentence-transformers with hash fallback."""

    def __init__(
        self,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        batch_size: int = 64,
        use_sentence_transformers: bool = True,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self.model_name = model_name
        self.dimension = dimension
        self.batch_size = batch_size
        self.use_sentence_transformers = use_sentence_transformers
        self.cache = cache or EmbeddingCache()
        self._st_model: Any = None
        self._st_attempted = False

    def _load_st_model(self) -> Any:
        if not self.use_sentence_transformers:
            return None
        if not self._st_attempted:
            self._st_attempted = True
            try:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(self.model_name)
            except Exception as exc:
                logger.warning(
                    "embedder_fallback_to_hash",
                    extra={"model_name": self.model_name, "reason": str(exc)},
                )
                self._st_model = None
        return self._st_model

    def _encode_uncached(self, texts: list[str]) -> np.ndarray:
        st_model = self._load_st_model()
        if st_model is not None:
            vecs = st_model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vecs, dtype=np.float32)

        # Deterministic hash-projection fallback (salted by model_name for distinct model comparisons)
        salted = [f"[{self.model_name}] {t}" for t in texts]
        return hash_embed_texts(salted, dim=self.dimension)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized float32 embeddings `(len(texts), dim)` using the cache."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        hashes = [compute_text_hash(t) for t in texts]
        results: list[np.ndarray | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        seen_missing: dict[str, int] = {}

        for idx, (text, t_hash) in enumerate(zip(texts, hashes, strict=True)):
            cached = self.cache.get(t_hash, self.model_name)
            if cached is not None:
                results[idx] = cached
            elif t_hash in seen_missing:
                missing_indices.append(idx)
            else:
                seen_missing[t_hash] = len(missing_texts)
                missing_texts.append(text)
                missing_indices.append(idx)

        if missing_texts:
            computed = self._encode_uncached(missing_texts)
            for t_hash, pos in seen_missing.items():
                vec = computed[pos]
                norm = float(np.linalg.norm(vec))
                if norm > 0:
                    vec = vec / norm
                self.cache.set(t_hash, self.model_name, vec)
            for idx in missing_indices:
                results[idx] = self.cache._memory[(hashes[idx], self.model_name)]

        return np.vstack([r for r in results if r is not None]).astype(np.float32)


def embed_interactions(
    session: Session,
    config: AppConfig,
    *,
    run_id: str,
    model_name: str | None = None,
    use_sentence_transformers: bool = True,
) -> dict[str, Any]:
    """Embed clean queries and clean responses separately with DB-backed caching."""
    start = time.perf_counter()
    resolved_model = model_name or config.embeddings.model_name

    cache = EmbeddingCache(session=session)
    cache.preload_for_model(resolved_model)

    embedder = TextEmbedder(
        model_name=resolved_model,
        dimension=config.embeddings.dimension,
        batch_size=config.embeddings.batch_size,
        use_sentence_transformers=use_sentence_transformers,
        cache=cache,
    )

    clean_rows = list(
        session.scalars(
            select(InteractionClean)
            .where(InteractionClean.valid.is_(True))
            .order_by(InteractionClean.interaction_id)
        ).all()
    )
    if not clean_rows:
        return {
            "model_name": resolved_model,
            "queries_processed": 0,
            "responses_processed": 0,
            "rows_inserted": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_hit_rate": 0.0,
            "seconds_per_1k_texts": 0.0,
        }

    existing_keys = set(
        session.execute(
            select(Embedding.interaction_id, Embedding.kind).where(
                Embedding.model_name == resolved_model
            )
        ).all()
    )

    q_texts = [r.clean_query for r in clean_rows]
    r_texts = [r.clean_response for r in clean_rows]
    q_vecs = embedder.encode(q_texts)
    r_vecs = embedder.encode(r_texts)

    rows_inserted = 0
    for idx, row in enumerate(clean_rows):
        if (row.interaction_id, "query") not in existing_keys:
            session.add(
                Embedding(
                    interaction_id=row.interaction_id,
                    kind="query",
                    model_name=resolved_model,
                    text_hash=compute_text_hash(row.clean_query),
                    vector=[float(x) for x in q_vecs[idx]],
                    run_id=run_id,
                )
            )
            rows_inserted += 1

        if (row.interaction_id, "response") not in existing_keys:
            session.add(
                Embedding(
                    interaction_id=row.interaction_id,
                    kind="response",
                    model_name=resolved_model,
                    text_hash=compute_text_hash(row.clean_response),
                    vector=[float(x) for x in r_vecs[idx]],
                    run_id=run_id,
                )
            )
            rows_inserted += 1

    session.flush()
    elapsed = max(time.perf_counter() - start, 1e-6)
    total_texts = len(q_texts) + len(r_texts)
    secs_per_1k = round(elapsed * (1000.0 / max(1, total_texts)), 3)

    counts = {
        "model_name": resolved_model,
        "queries_processed": len(q_texts),
        "responses_processed": len(r_texts),
        "rows_inserted": rows_inserted,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "cache_hit_rate": cache.hit_rate,
        "seconds_per_1k_texts": secs_per_1k,
    }
    logger.info("embeddings_completed", extra=counts)
    return counts


def _recall_at_k_same_intent(vectors: np.ndarray, intents: list[str], k: int = 5) -> float:
    """Compute recall@k: fraction of queries whose top-k cosine neighbors contain >=1 same-intent match."""
    n = len(intents)
    if n <= 1:
        return 0.0
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, -np.inf)
    k_eff = min(k, n - 1)
    top_k = np.argpartition(-sim, kth=k_eff - 1, axis=1)[:, :k_eff]
    hits = 0
    for i in range(n):
        if any(intents[j] == intents[i] for j in top_k[i]):
            hits += 1
    return round(hits / n, 4)


def compare_embedding_models(
    session: Session,
    config: AppConfig,
    *,
    model_a: str | None = None,
    model_b: str | None = None,
    use_sentence_transformers: bool = True,
) -> dict[str, Any]:
    """Compare two candidate embedding models on within-intent paraphrase recall@5 and speed."""
    m_a = model_a or config.embeddings.model_name
    m_b = model_b or config.embeddings.alternative_model_name

    rows = session.execute(
        select(InteractionClean.clean_query, InteractionRaw.intent_label)
        .join(InteractionRaw, InteractionRaw.interaction_id == InteractionClean.interaction_id)
        .where(
            InteractionClean.valid.is_(True),
            InteractionClean.is_representative.is_(True),
            InteractionRaw.intent_label.is_not(None),
            # Controlled-tier rewrites copy natural queries verbatim and would inflate recall@k
            InteractionRaw.source_tag != CONTROLLED_SOURCE_TAG,
        )
        .order_by(InteractionClean.interaction_id)
    ).all()

    texts = [r[0] for r in rows]
    intents = [str(r[1]) for r in rows]

    report: dict[str, Any] = {"n_queries": len(texts), "models": {}}
    for m_name in (m_a, m_b):
        emb = TextEmbedder(
            model_name=m_name,
            dimension=config.embeddings.dimension,
            batch_size=config.embeddings.batch_size,
            use_sentence_transformers=use_sentence_transformers,
        )
        t0 = time.perf_counter()
        vecs = emb.encode(texts)
        elapsed = max(time.perf_counter() - t0, 1e-6)
        recall5 = _recall_at_k_same_intent(vecs, intents, k=5)
        secs_per_1k = round(elapsed * (1000.0 / max(1, len(texts))), 3)
        report["models"][m_name] = {
            "recall_at_5": recall5,
            "seconds_per_1k_texts": secs_per_1k,
        }

    return report
