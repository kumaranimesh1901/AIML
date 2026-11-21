"""Unit and acceptance tests for Phase 3: Ingestion validation, cleaning, PII, dedupe, and incremental runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Engine, event, select
from typer.testing import CliRunner

from edg.cleaning.cleaner import clean_text, normalize_timestamp_utc
from edg.cleaning.dedupe import assign_duplicate_groups
from edg.cleaning.pii import PIIPseudonymizer
from edg.cleaning.pipeline import run_cleaning_pipeline
from edg.cli import app
from edg.config import AppConfig
from edg.db.models import (
    Customer,
    InteractionClean,
    InteractionRaw,
    Order,
    PIIMapping,
    StateSnapshot,
)
from edg.db.repositories import track_run
from edg.db.session import get_session
from edg.ingestion.ingest import ingest_raw_records

runner = CliRunner()


def test_clean_text_preserves_case_and_stopwords_strips_html() -> None:
    raw = "  <p>Hello   <b>NovaMart</b> &amp; Support! Where is the Order ORD-000001?</p>\n\t  "
    cleaned = clean_text(raw)
    assert cleaned == "Hello NovaMart & Support! Where is the Order ORD-000001?"
    assert "NovaMart" in cleaned  # case preserved
    assert "is the" in cleaned  # stopwords preserved
    assert "<p>" not in cleaned

    iso = normalize_timestamp_utc("2026-11-15T10:00:00Z")
    assert iso == "2026-11-15T10:00:00+00:00"


def test_pii_same_email_maps_to_same_token_and_preserves_order_ids() -> None:
    pseudo = PIIPseudonymizer(use_spacy=False)
    t1, r1 = pseudo.pseudonymize(
        "My name is Jane Doe, email alice@example.com, phone 555-234-5678, order ORD-000042."
    )
    t2, r2 = pseudo.pseudonymize(
        "Please reply to ALICE@EXAMPLE.COM about order ORD-000042 and card 4111-2222-3333-4444."
    )

    assert "alice@example.com" not in t1.lower()
    assert "555-234-5678" not in t1
    assert "Jane Doe" not in t1
    assert "4111-2222-3333-4444" not in t2
    # Order ID stays intact and joinable
    assert "ORD-000042" in t1
    assert "ORD-000042" in t2

    # Same email (case-insensitive) maps to the exact same token
    email_tok_1 = next(rec["token"] for rec in r1 if rec["entity_type"] == "EMAIL")
    email_tok_2 = next(rec["token"] for rec in r2 if rec["entity_type"] == "EMAIL")
    assert email_tok_1 == email_tok_2


def test_exact_and_near_dedupe_within_intent() -> None:
    items = [
        ("int_1", "track_order", "Where is my package for order ORD-000001?"),
        ("int_2", "track_order", "Where is my package for order ORD-000001?"),  # exact dup
        ("int_3", "get_refund", "Where is my package for order ORD-000001?"),  # different intent
        ("int_4", "track_order", "Can you check delivery options for express shipping?"),
    ]
    res = assign_duplicate_groups(items, cosine_threshold=0.95)
    gid1, cnt1, rep1 = res["int_1"]
    gid2, cnt2, rep2 = res["int_2"]
    gid3, cnt3, rep3 = res["int_3"]

    assert gid1 == gid2
    assert rep1 is True and cnt1 == 2
    assert rep2 is False and cnt2 == 1
    # Different intent must NOT merge with track_order
    assert gid3 != gid1
    assert rep3 is True and cnt3 == 1


def test_ingest_raw_records_rejects_invalid_without_silent_repair(
    db_engine: Engine, small_config: AppConfig
) -> None:
    now = datetime.now(UTC)
    with get_session(db_engine) as session:
        with track_run(session, step="seed_snap", config=small_config) as (run, _counts):
            session.add(
                Customer(
                    customer_id="CUST-1", name_token="N1", email_token="E1", run_id=run.run_id
                )
            )
            session.flush()
            session.add(
                Order(
                    order_id="ORD-1",
                    customer_id="CUST-1",
                    placed_at=now,
                    status="shipped",
                    total_amount=25.0,
                    currency="USD",
                    run_id=run.run_id,
                )
            )
            session.flush()
            session.add(
                StateSnapshot(
                    snapshot_id="SNAP-1",
                    order_id="ORD-1",
                    policy_version="v1",
                    snapshot_json={"order": {"order_id": "ORD-1"}},
                    hard_state_tags=[],
                    run_id=run.run_id,
                )
            )
            session.flush()

            stats = ingest_raw_records(
                session,
                records=[
                    {
                        "interaction_id": "RAW-OK",
                        "conversation_id": "c1",
                        "turn_index": 0,
                        "user_query": "Where is ORD-1?",
                        "ai_response": "It is shipped.",
                        "model_version": "v1",
                        "timestamp": "2026-11-15T12:00:00Z",
                        "source_tag": "synthetic_natural",
                        "snapshot_id": "SNAP-1",
                    },
                    {
                        "interaction_id": "RAW-BAD",
                        "conversation_id": "c2",
                        "turn_index": 0,
                        "user_query": "   ",  # invalid blank query
                        "ai_response": "Reply",
                        "model_version": "v1",
                        "timestamp": "2026-11-15T12:00:00Z",
                        "source_tag": "synthetic_natural",
                        "snapshot_id": "SNAP-1",
                    },
                ],
                run_id=run.run_id,
            )

    assert stats["inserted"] == 1
    assert stats["rejected_count"] == 1
    assert len(stats["reject_reasons"]) == 1


def test_cleaning_pipeline_never_updates_raw_is_incremental_and_records_reject_reasons(
    db_engine: Engine, small_config: AppConfig
) -> None:
    now = datetime.now(UTC)
    with get_session(db_engine) as session:
        with track_run(session, step="seed_raw_clean", config=small_config) as (run, _counts):
            session.add(
                Customer(
                    customer_id="CUST-1", name_token="N1", email_token="E1", run_id=run.run_id
                )
            )
            session.flush()
            session.add(
                Order(
                    order_id="ORD-1",
                    customer_id="CUST-1",
                    placed_at=now,
                    status="shipped",
                    total_amount=50.0,
                    currency="USD",
                    run_id=run.run_id,
                )
            )
            session.flush()
            session.add(
                StateSnapshot(
                    snapshot_id="SNAP-1",
                    order_id="ORD-1",
                    policy_version="v1",
                    snapshot_json={"order": {"order_id": "ORD-1"}},
                    hard_state_tags=[],
                    run_id=run.run_id,
                )
            )
            session.flush()
            session.add_all(
                [
                    InteractionRaw(
                        interaction_id="INT-1",
                        conversation_id="c1",
                        turn_index=0,
                        user_query="<p>Contact bob@example.com about ORD-1</p>",
                        ai_response="Your order ORD-1 is shipped.",
                        model_version="v1",
                        timestamp=now,
                        source_tag="synthetic_natural",
                        intent_label="track_order",
                        snapshot_id="SNAP-1",
                        raw_json={},
                        run_id=run.run_id,
                    ),
                    InteractionRaw(
                        interaction_id="INT-2",
                        conversation_id="c2",
                        turn_index=0,
                        user_query="Contact bob@example.com about ORD-1",
                        ai_response="Your order ORD-1 is shipped.",
                        model_version="v1",
                        timestamp=now,
                        source_tag="synthetic_natural",
                        intent_label="track_order",
                        snapshot_id="SNAP-1",
                        raw_json={},
                        run_id=run.run_id,
                    ),
                    InteractionRaw(
                        interaction_id="INT-BAD",
                        conversation_id="c3",
                        turn_index=0,
                        user_query=" ",  # invalid blank query
                        ai_response="Empty query",
                        model_version="v1",
                        timestamp=now,
                        source_tag="invalid_tag",
                        intent_label="track_order",
                        snapshot_id="SNAP-1",
                        raw_json={},
                        run_id=run.run_id,
                    ),
                ]
            )

    # Track SQL statements to assert ZERO UPDATEs on interactions_raw
    raw_updates: list[str] = []

    def _check_sql(_conn, _cursor, statement, _params, _ctx, _executemany) -> None:
        upper = statement.upper()
        if "UPDATE" in upper and "INTERACTIONS_RAW" in upper:
            raw_updates.append(statement)

    event.listen(db_engine, "before_cursor_execute", _check_sql)
    try:
        with get_session(db_engine) as session:
            with track_run(session, step="clean_1", config=small_config) as (run, counts):
                res1 = run_cleaning_pipeline(
                    session, small_config, run_id=run.run_id, use_spacy=False
                )
                counts.update(res1)

        assert raw_updates == []
        assert res1["records_in"] == 3
        assert res1["valid_count"] == 2
        assert res1["rejected_count"] == 1
        assert res1["pii_replacements"] == 2
        assert res1["dup_groups"] == 1

        with get_session(db_engine) as session:
            c1 = session.get(InteractionClean, "INT-1")
            c2 = session.get(InteractionClean, "INT-2")
            c_bad = session.get(InteractionClean, "INT-BAD")
            assert c1 is not None and c2 is not None and c_bad is not None
            assert c1.clean_query == c2.clean_query
            assert "bob@example.com" not in c1.clean_query
            assert "ORD-1" in c1.clean_query
            assert c1.dup_group_id == c2.dup_group_id
            assert c1.is_representative is True and c1.dup_count == 2
            assert c2.is_representative is False
            assert c_bad.valid is False
            assert c_bad.reject_reason is not None

            pii_rows = list(session.scalars(select(PIIMapping)).all())
            assert len(pii_rows) == 2
            assert pii_rows[0].token == pii_rows[1].token

        # Second run must process 0 new rows (incremental)
        with get_session(db_engine) as session:
            with track_run(session, step="clean_2", config=small_config) as (run, counts):
                res2 = run_cleaning_pipeline(
                    session, small_config, run_id=run.run_id, use_spacy=False
                )
                counts.update(res2)

        assert res2["records_in"] == 0
        assert res2["records_out"] == 0
        assert raw_updates == []
    finally:
        event.remove(db_engine, "before_cursor_execute", _check_sql)


def test_cli_clean_run_small(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'cli_clean.db'}"
    assert (
        runner.invoke(
            app,
            ["world", "build", "--small", "--db-url", db_url, "--synthetic-catalogue", "--no-st"],
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            app,
            ["sut", "run", "--small", "--db-url", db_url, "--provider", "fake", "--no-hf"],
        ).exit_code
        == 0
    )

    c1 = runner.invoke(app, ["clean", "run", "--small", "--db-url", db_url, "--no-spacy"])
    assert c1.exit_code == 0, c1.output
    data1 = json.loads(c1.stdout)
    assert data1["records_in"] == 550
    assert data1["valid_count"] == 550
    assert data1["rejected_count"] == 0

    c2 = runner.invoke(app, ["clean", "run", "--small", "--db-url", db_url, "--no-spacy"])
    assert c2.exit_code == 0, c2.output
    data2 = json.loads(c2.stdout)
    assert data2["records_in"] == 0
