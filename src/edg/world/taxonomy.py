"""Taxonomy schema and loader for intents, failure types, and severity matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from edg.config import _find_project_root

ImpactArea = Literal["money", "delivery", "account", "information"]
SeverityLevel = Literal["critical", "high", "medium", "low"]

SEVERITY_ORDER: dict[str, int] = {"low": 1, "medium": 2, "high": 3, "critical": 4}


class IntentDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    impact_area: ImpactArea


class FailureTypeDef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    decision_rule: str


class TaxonomyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intents: dict[str, IntentDef]
    failure_types: dict[str, FailureTypeDef]
    impact_areas: list[ImpactArea]
    severity_matrix: dict[str, dict[ImpactArea, SeverityLevel]]

    def get_impact_area(self, intent: str | None) -> ImpactArea:
        if intent and intent in self.intents:
            return self.intents[intent].impact_area
        return "information"

    def get_severity(self, failure_type: str, impact_area: ImpactArea) -> SeverityLevel:
        if failure_type not in self.severity_matrix:
            raise KeyError(f"Unknown failure_type: {failure_type}")
        return self.severity_matrix[failure_type][impact_area]

    def max_severity(self, failure_types: list[str], impact_area: ImpactArea) -> SeverityLevel:
        if not failure_types:
            return "low"
        severities = [self.get_severity(ft, impact_area) for ft in failure_types]
        return max(severities, key=lambda s: SEVERITY_ORDER[s])


def load_taxonomy(path: str | Path | None = None) -> TaxonomyConfig:
    """Load and validate configs/taxonomy.yaml."""
    resolved = Path(path) if path is not None else _find_project_root() / "configs" / "taxonomy.yaml"
    if not resolved.exists():
        raise FileNotFoundError(f"Taxonomy file not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return TaxonomyConfig.model_validate(raw)
