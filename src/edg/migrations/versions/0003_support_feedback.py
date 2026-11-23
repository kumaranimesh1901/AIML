"""Customer feedback and support escalation tables (chat_turns, support_tickets).

Revision ID: 0003_support_feedback
Revises: 0002_immutability_triggers
Create Date: 2026-11-23 00:00:00.000000
"""

from __future__ import annotations

from alembic import op

from edg.db.models import Base

revision = "0003_support_feedback"
down_revision = "0002_immutability_triggers"
branch_labels = None
depends_on = None

_TABLES = ("chat_turns", "support_tickets")


def upgrade() -> None:
    # 0001 creates every model table on a fresh database, so only add what an older DB lacks
    tables = [Base.metadata.tables[name] for name in _TABLES]
    Base.metadata.create_all(bind=op.get_bind(), tables=tables, checkfirst=True)


def downgrade() -> None:
    tables = [Base.metadata.tables[name] for name in reversed(_TABLES)]
    Base.metadata.drop_all(bind=op.get_bind(), tables=tables, checkfirst=True)
