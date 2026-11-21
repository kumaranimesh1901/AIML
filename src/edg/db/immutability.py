"""Database-level immutability triggers for insert-only tables.

ORM `before_update`/`before_delete` hooks only fire for ORM unit-of-work flushes. Raw SQL and
ORM bulk `update()`/`delete()` statements bypass them, so the insert-only guarantees for
`interactions_raw` and frozen benchmark releases are also enforced by database triggers.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

IMMUTABLE_TABLES: tuple[str, ...] = ("interactions_raw", "benchmark_releases", "release_items")


def _trigger_names(table: str) -> tuple[str, str]:
    return f"trg_{table}_no_update", f"trg_{table}_no_delete"


def install_immutability_triggers(conn: Connection) -> None:
    """Create UPDATE/DELETE-blocking triggers on every immutable table (idempotent)."""
    dialect = conn.dialect.name
    if dialect == "sqlite":
        for table in IMMUTABLE_TABLES:
            upd, dele = _trigger_names(table)
            for name, op in ((upd, "UPDATE"), (dele, "DELETE")):
                conn.execute(
                    text(
                        f"CREATE TRIGGER IF NOT EXISTS {name} BEFORE {op} ON {table} "
                        f"BEGIN SELECT RAISE(ABORT, '{table} is immutable; {op} is forbidden'); END"
                    )
                )
    elif dialect == "postgresql":
        conn.execute(
            text(
                "CREATE OR REPLACE FUNCTION edg_block_immutable_change() RETURNS trigger AS $$ "
                "BEGIN RAISE EXCEPTION '% is immutable; % is forbidden', TG_TABLE_NAME, TG_OP; END; "
                "$$ LANGUAGE plpgsql"
            )
        )
        for table in IMMUTABLE_TABLES:
            upd, dele = _trigger_names(table)
            for name, op in ((upd, "UPDATE"), (dele, "DELETE")):
                conn.execute(text(f"DROP TRIGGER IF EXISTS {name} ON {table}"))
                conn.execute(
                    text(
                        f"CREATE TRIGGER {name} BEFORE {op} ON {table} "
                        "FOR EACH ROW EXECUTE FUNCTION edg_block_immutable_change()"
                    )
                )


def drop_immutability_triggers(conn: Connection) -> None:
    """Remove the immutability triggers (used by migration downgrade)."""
    dialect = conn.dialect.name
    for table in IMMUTABLE_TABLES:
        for name in _trigger_names(table):
            if dialect == "sqlite":
                conn.execute(text(f"DROP TRIGGER IF EXISTS {name}"))
            elif dialect == "postgresql":
                conn.execute(text(f"DROP TRIGGER IF EXISTS {name} ON {table}"))
    if dialect == "postgresql":
        conn.execute(text("DROP FUNCTION IF EXISTS edg_block_immutable_change()"))
