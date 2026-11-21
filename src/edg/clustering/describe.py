"""Cluster labeling and description from the 15 most central cluster members."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
from pydantic import BaseModel, Field

from edg.llm.client import BaseLLMClient, LLMError
from edg.logging import get_logger

logger = get_logger(__name__)

_STOPWORDS = frozenset(
    {
        "the", "and", "for", "that", "this", "with", "you", "your", "have", "can", "are",
        "was", "were", "not", "but", "from", "they", "what", "how", "when", "where", "why",
        "please", "help", "order", "ord", "about", "would", "could", "there", "their",
        "hello", "thanks", "hi", "hey", "need", "want", "check", "let", "know", "just",
    }
)
_WORD_RE = re.compile(r"[a-z]{3,}")


class ClusterDescription(BaseModel):
    label: str = Field(min_length=2, max_length=120)
    description: str = Field(min_length=5, max_length=500)


def select_central_examples(
    queries: list[str],
    vectors: np.ndarray,
    *,
    max_examples: int = 15,
) -> list[str]:
    """Select up to max_examples queries closest to the cluster centroid."""
    if len(queries) <= max_examples:
        return list(queries)
    centroid = np.mean(vectors, axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 0:
        centroid = centroid / norm
    sims = vectors @ centroid
    top_idx = np.argsort(-sims)[:max_examples]
    return [queries[int(i)] for i in top_idx]


def _keyword_fallback_description(examples: list[str]) -> ClusterDescription:
    counts: Counter[str] = Counter()
    for q in examples:
        for word in _WORD_RE.findall(q.lower()):
            if word not in _STOPWORDS:
                counts[word] += 1
    top_terms = [w for w, _c in counts.most_common(4)]
    if not top_terms:
        return ClusterDescription(
            label="General Support Inquiries",
            description=f"Cluster of {len(examples)} general customer-support queries.",
        )
    label = " / ".join(t.title() for t in top_terms[:3])
    desc = (
        f"Customer queries centered on {', '.join(top_terms)} "
        f"(summarized from {len(examples)} central examples)."
    )
    return ClusterDescription(label=label, description=desc)


def describe_cluster(
    central_examples: list[str],
    *,
    llm_client: BaseLLMClient | None = None,
    model: str | None = None,
) -> ClusterDescription:
    """Ask an LLM for a JSON label + description of a cluster from its central examples."""
    if not central_examples:
        return ClusterDescription(label="Empty Cluster", description="No examples available.")

    use_live_llm = llm_client is not None and llm_client.provider != "fake"
    if use_live_llm and llm_client is not None:
        bullets = "\n".join(f"- {ex}" for ex in central_examples[:15])
        prompt = (
            "Given these representative e-commerce customer-support queries from one cluster, "
            "return JSON with a short `label` (2-5 words) and a 1-2 sentence `description`:\n\n"
            f"{bullets}\n\n"
            'Return ONLY valid JSON: {"label": "...", "description": "..."}'
        )
        try:
            parsed, _resp = llm_client.generate_json(
                prompt=prompt,
                schema_model=ClusterDescription,
                model=model,
                temperature=0.0,
            )
            return parsed
        except LLMError as exc:
            logger.warning("cluster_describe_llm_fallback", extra={"error": str(exc)})

    return _keyword_fallback_description(central_examples)
