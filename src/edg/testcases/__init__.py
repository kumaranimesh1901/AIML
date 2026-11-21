"""Test case generation, snapshot-validated checklists, within-cluster dedupe, and variants."""

from edg.testcases.checklist import (
    ExpectedChecklist,
    generate_and_validate_checklist,
    validate_checklist_against_snapshot,
)
from edg.testcases.dedupe import dedupe_test_cases_within_cluster
from edg.testcases.generator import run_testcase_generation
from edg.testcases.variants import generate_critical_variants

__all__ = [
    "ExpectedChecklist",
    "dedupe_test_cases_within_cluster",
    "generate_and_validate_checklist",
    "generate_critical_variants",
    "run_testcase_generation",
    "validate_checklist_against_snapshot",
]
