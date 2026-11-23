"""A bundled copy of real pipeline results, so a fresh clone can open the dashboard without
re-running the pipeline.

`edg db snapshot` (run by the project owner after a real run) copies the working database to
`data/demo/edg_demo.db`, which is committed. On a fresh clone, `edg db init` (and the dashboard)
copy that snapshot to `data/edg.db` when it is missing or holds no pipeline data. Nothing here
generates results: the snapshot is the owner's own run, byte for byte, minus the LLM response
cache and the owner's own chat and support-desk test sessions."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("data/edg.db")
DEMO_DIR = Path("data/demo")
DEMO_DB_PATH = DEMO_DIR / "edg_demo.db"
EXPORTS_DIR = Path("data/exports")
# Exports the dashboard reads next to the database (the real two-annotator check of the judge)
DEMO_EXPORTS = ("human_agreement.json", "human_agreement.md")
# Rows that are not results: the response cache and the owner's own live chat / ticket tests
_CLEARED_TABLES = ("support_tickets", "chat_turns", "llm_cache")


def sqlite_path(database_url: str) -> Path | None:
    """The file behind a `sqlite:///` URL, or None for any other database."""
    if not database_url.startswith("sqlite:///"):
        return None
    raw = database_url.removeprefix("sqlite:///")
    return None if raw in ("", ":memory:") else Path(raw)


def _has_pipeline_data(path: Path) -> bool:
    """True when the file already holds a store (the first pipeline stage), so any real run,
    even a partial one, counts."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return bool(con.execute("SELECT 1 FROM orders LIMIT 1").fetchone())
        finally:
            con.close()
    except sqlite3.Error:
        return False


def is_default_database(database_url: str) -> bool:
    path = sqlite_path(database_url)
    return path is not None and path.resolve() == DEFAULT_DB_PATH.resolve()


def seed_from_demo(database_url: str) -> bool:
    """Copy the bundled results into the default database if it is missing or empty.

    Only the default `data/edg.db` is ever seeded, and never one that already holds data, so a
    real or partial run, a custom `--db-url` and the test suite are never touched."""
    if not DEMO_DB_PATH.exists() or not is_default_database(database_url):
        return False
    target = sqlite_path(database_url)
    assert target is not None
    if _has_pipeline_data(target):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{target}{suffix}").unlink(missing_ok=True)
    shutil.copyfile(DEMO_DB_PATH, target)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    for name in DEMO_EXPORTS:
        src, dst = DEMO_DIR / name, EXPORTS_DIR / name
        if src.exists() and not dst.exists():
            shutil.copyfile(src, dst)
    return True


def make_snapshot(database_url: str, out_path: Path = DEMO_DB_PATH) -> Path:
    """Write a compact copy of a real run's database for sharing the results."""
    source = sqlite_path(database_url)
    if source is None or not source.exists():
        raise FileNotFoundError(f"No SQLite database at {database_url}")
    if not _has_pipeline_data(source):
        raise ValueError(f"{source} holds no pipeline results yet; run the pipeline first")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    src = sqlite3.connect(source)
    try:
        dst = sqlite3.connect(tmp)
        try:
            src.backup(dst)  # consistent copy, even if the dashboard has the file open
            existing = {r[0] for r in dst.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in _CLEARED_TABLES:
                if table in existing:
                    dst.execute(f"DELETE FROM {table}")  # noqa: S608 - fixed table names
            dst.commit()
            dst.execute("PRAGMA journal_mode=DELETE")
            dst.execute("VACUUM")
        finally:
            dst.close()
    finally:
        src.close()
    tmp.replace(out_path)
    for name in DEMO_EXPORTS:
        if (EXPORTS_DIR / name).exists():
            shutil.copyfile(EXPORTS_DIR / name, out_path.parent / name)
    return out_path
