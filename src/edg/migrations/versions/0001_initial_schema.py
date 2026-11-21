"""Initial EDG schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-21 00:00:00.000000
"""

from __future__ import annotations

from alembic import op

from edg.db.models import Base

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind, checkfirst=True)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
