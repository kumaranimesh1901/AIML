"""Enforce insert-only tables with database triggers.

Revision ID: 0002_immutability_triggers
Revises: 0001_initial
Create Date: 2026-11-21 00:00:00.000000
"""

from __future__ import annotations

from alembic import op

from edg.db.immutability import drop_immutability_triggers, install_immutability_triggers

revision = "0002_immutability_triggers"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    install_immutability_triggers(op.get_bind())


def downgrade() -> None:
    drop_immutability_triggers(op.get_bind())
