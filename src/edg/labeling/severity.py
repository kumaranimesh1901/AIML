"""Severity matrix resolution from taxonomy.yaml (highest severity wins)."""

from __future__ import annotations

from edg.world.hard_states import (
    BOUNDARY_RETURN_WINDOW,
    DELIVERED_DISPUTED,
    DUPLICATE_CHARGE,
    PAST_RETURN_WINDOW,
    REFUND_APPROVED_NOT_ISSUED,
)
from edg.world.taxonomy import ImpactArea, SeverityLevel, TaxonomyConfig

_MONEY_HARD_STATES = {
    DUPLICATE_CHARGE,
    PAST_RETURN_WINDOW,
    BOUNDARY_RETURN_WINDOW,
    REFUND_APPROVED_NOT_ISSUED,
}


def resolve_impact_area(
    *,
    intent: str | None,
    hard_state_tags: list[str] | None,
    taxonomy: TaxonomyConfig,
) -> ImpactArea:
    """Determine the impact area (money, delivery, account, information) for an interaction."""
    tags = set(hard_state_tags or [])
    if tags & _MONEY_HARD_STATES:
        return "money"
    if DELIVERED_DISPUTED in tags:
        return "delivery"
    return taxonomy.get_impact_area(intent)


def resolve_severity(
    *,
    failure_types: list[str],
    impact_area: ImpactArea,
    taxonomy: TaxonomyConfig,
) -> SeverityLevel:
    """Look up each taxonomy failure type in the severity matrix; highest severity wins."""
    known = [ft for ft in failure_types if ft in taxonomy.severity_matrix]
    if not known:
        return "low"
    return taxonomy.max_severity(known, impact_area)


def resolve_severity_for_interaction(
    *,
    failure_types: list[str],
    intent: str | None,
    hard_state_tags: list[str] | None,
    taxonomy: TaxonomyConfig,
) -> tuple[ImpactArea, SeverityLevel]:
    area = resolve_impact_area(intent=intent, hard_state_tags=hard_state_tags, taxonomy=taxonomy)
    sev = resolve_severity(failure_types=failure_types, impact_area=area, taxonomy=taxonomy)
    return area, sev
