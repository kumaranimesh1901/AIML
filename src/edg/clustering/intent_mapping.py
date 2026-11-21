"""Map discovered clusters to majority Bitext intents and flag candidate new intents."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class ClusterIntentMapping:
    mapped_intent: str | None
    purity: float
    hard_query_share: float
    is_new_intent: bool


def map_cluster_intent(
    intents: list[str | None],
    *,
    is_hard_query_flags: list[bool],
    purity_threshold: float = 0.50,
) -> ClusterIntentMapping:
    """Compute majority intent, purity, and candidate new-intent flag for a cluster."""
    total = len(intents)
    if total == 0:
        return ClusterIntentMapping(
            mapped_intent=None, purity=0.0, hard_query_share=0.0, is_new_intent=False
        )

    labeled = [i for i in intents if i]
    if labeled:
        counts = Counter(labeled)
        majority_intent, majority_count = counts.most_common(1)[0]
        purity = round(majority_count / len(labeled), 4)
    else:
        majority_intent = None
        purity = 0.0

    hard_share = round(sum(1 for f in is_hard_query_flags if f) / total, 4)
    is_new = (purity < purity_threshold) or (hard_share > 0.50)

    return ClusterIntentMapping(
        mapped_intent=majority_intent,
        purity=purity,
        hard_query_share=hard_share,
        is_new_intent=is_new,
    )
