"""Unit and acceptance tests for Phase 1: World, policies, catalogue, hard states, and snapshots."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import Engine, select
from typer.testing import CliRunner

from edg.cli import app
from edg.config import AppConfig, load_config
from edg.db.models import PipelineRun, PolicyDocument, StateSnapshot
from edg.db.repositories import track_run
from edg.db.session import create_db_engine, get_session, init_db
from edg.world.catalogue import load_catalogue
from edg.world.hard_states import (
    BOUNDARY_RETURN_WINDOW,
    CANCEL_AFTER_SHIPPING,
    DELIVERED_DISPUTED,
    DUPLICATE_CHARGE,
    PARTIAL_SHIPMENT,
    PAST_RETURN_WINDOW,
    REFUND_APPROVED_NOT_ISSUED,
    WRONG_ITEM_DELIVERED,
)
from edg.world.policies import load_and_store_policies
from edg.world.store_generator import generate_store_world
from edg.world.taxonomy import load_taxonomy

runner = CliRunner()


def _parse_iso(val: str) -> datetime:
    return datetime.fromisoformat(val)


def test_taxonomy_loads_and_resolves_severity() -> None:
    tax = load_taxonomy()
    assert len(tax.intents) == 27
    assert len(tax.failure_types) == 9
    assert set(tax.failure_types.keys()) == {
        "factual_error",
        "hallucination",
        "incomplete_response",
        "irrelevant_response",
        "misunderstood_intent",
        "inadequate_resolution",
        "policy_violation",
        "unsafe_response",
        "contradictory_information",
    }
    assert tax.get_severity("factual_error", "money") == "critical"
    assert tax.get_severity("factual_error", "delivery") == "high"
    assert tax.get_severity("unsafe_response", "information") == "critical"
    assert tax.max_severity(["incomplete_response", "policy_violation"], "money") == "critical"
    assert tax.get_impact_area("get_refund") == "money"
    assert tax.get_impact_area("track_order") == "delivery"


def test_policies_chunk_and_embed_into_db(db_engine: Engine, small_config: AppConfig) -> None:
    with get_session(db_engine) as session:
        with track_run(session, step="policies_test", config=small_config) as (run, _counts):
            n_chunks = load_and_store_policies(
                session,
                run_id=run.run_id,
                policy_version="v1",
                embedding_dim=384,
                use_sentence_transformers=False,
            )
    assert n_chunks >= 12

    with get_session(db_engine) as session:
        docs = list(session.scalars(select(PolicyDocument)).all())
        assert len(docs) == n_chunks
        for doc in docs:
            assert doc.version == "v1"
            assert doc.embedding is not None
            assert len(doc.embedding) == 384
            norm = float(np.linalg.norm(doc.embedding))
            assert abs(norm - 1.0) < 1e-4


def test_catalogue_fallback_on_hf_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_args, **_kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr("edg.world.catalogue._try_load_amazon_reviews_2023", _boom)
    products = load_catalogue(size=25, seed=42, prefer_hf=True)
    assert len(products) == 25
    assert all(p.source == "synthetic" for p in products)
    assert all(p.price > 0 for p in products)


def test_world_dates_amounts_and_hard_states(db_engine: Engine, small_config: AppConfig) -> None:
    with get_session(db_engine) as session:
        with track_run(session, step="world_build", config=small_config) as (run, counts):
            res = generate_store_world(
                session,
                small_config,
                run_id=run.run_id,
                prefer_hf_catalogue=False,
                use_sentence_transformers=False,
            )
            counts.update(res)

    with get_session(db_engine) as session:
        snapshots = list(session.scalars(select(StateSnapshot)).all())
        assert len(snapshots) == small_config.world.n_orders

        tag_counts: dict[str, int] = {k: 0 for k in small_config.world.hard_state_rates}
        for snap in snapshots:
            data = snap.snapshot_json
            assert isinstance(data, dict)
            # Round-trip valid JSON check
            assert json.loads(json.dumps(data)) == data

            tags = set(snap.hard_state_tags)
            for t in tags:
                tag_counts[t] = tag_counts.get(t, 0) + 1

            order = data["order"]
            placed_at = _parse_iso(order["placed_at"])

            # Monetary consistency
            computed_total = round(sum(it["qty"] * it["unit_price"] for it in data["items"]), 2)
            assert abs(order["total_amount"] - computed_total) < 1e-6

            # Payment date consistency
            payments = data["payments"]
            assert len(payments) >= 1
            for pay in payments:
                assert _parse_iso(pay["charged_at"]) >= placed_at

            if DUPLICATE_CHARGE in tags:
                assert len(payments) == 2
                assert any(p["is_duplicate"] for p in payments)
                assert all(p["status"] == "captured" for p in payments)

            # Shipment date consistency
            shipments = data["shipments"]
            delivered_dates: list[datetime] = []
            for shp in shipments:
                scan_at = _parse_iso(shp["carrier_scan_at"])
                exp_del = _parse_iso(shp["expected_delivery"])
                assert scan_at >= placed_at
                assert exp_del >= placed_at
                if shp["actual_delivery"] is not None:
                    act_del = _parse_iso(shp["actual_delivery"])
                    assert act_del >= placed_at
                    assert act_del >= scan_at
                    delivered_dates.append(act_del)

            if DELIVERED_DISPUTED in tags:
                assert any(s["customer_disputed"] and s["actual_delivery"] is not None for s in shipments)

            if PARTIAL_SHIPMENT in tags:
                assert len(shipments) == 2
                assert len(data["items"]) >= 2
                assert any(s["actual_delivery"] is not None for s in shipments)
                assert any(s["actual_delivery"] is None for s in shipments)

            if CANCEL_AFTER_SHIPPING in tags:
                assert order["status"] in {"shipped", "out_for_delivery"}
                assert len(shipments) >= 1
                assert all(s["actual_delivery"] is None for s in shipments)

            # Return & refund date consistency
            returns = data["refunds_returns"]
            first_charge = min(_parse_iso(p["charged_at"]) for p in payments)
            for ret in returns:
                req_at = _parse_iso(ret["requested_at"])
                assert req_at >= first_charge
                if delivered_dates:
                    assert req_at >= min(delivered_dates)
                if ret["refund_issued_at"] is not None:
                    iss_at = _parse_iso(ret["refund_issued_at"])
                    assert iss_at >= req_at
                    assert iss_at >= first_charge

            if PAST_RETURN_WINDOW in tags:
                assert len(returns) == 1
                delta_days = (_parse_iso(returns[0]["requested_at"]) - min(delivered_dates)).days
                assert delta_days > 30

            if BOUNDARY_RETURN_WINDOW in tags:
                assert len(returns) == 1
                delta_days = (_parse_iso(returns[0]["requested_at"]) - min(delivered_dates)).days
                assert delta_days in (29, 30)

            if REFUND_APPROVED_NOT_ISSUED in tags:
                assert len(returns) == 1
                assert returns[0]["status"] == "approved"
                assert returns[0]["refund_issued_at"] is None

            if WRONG_ITEM_DELIVERED in tags:
                assert len(returns) == 1
                assert returns[0]["reason"] == "wrong_item_delivered"

        # Every hard-state tag appears within ±2 percentage points of configured rate
        n_orders = small_config.world.n_orders
        for tag, target_rate in small_config.world.hard_state_rates.items():
            actual_rate = tag_counts[tag] / n_orders
            assert abs(actual_rate - target_rate) <= 0.02, (
                f"Tag {tag}: actual_rate={actual_rate:.4f} vs target_rate={target_rate:.4f}"
            )


def test_snapshots_are_reproducible_with_same_seed(tmp_path: Path) -> None:
    def _run_world(db_name: str) -> dict[str, str]:
        db_url = f"sqlite:///{tmp_path / db_name}"
        cfg = load_config(
            small=True,
            overrides={"database": {"url": db_url}, "world": {"n_orders": 60, "n_customers": 30}},
        )
        engine = init_db(db_url, use_alembic=True)
        try:
            with get_session(engine) as session:
                with track_run(session, step="world_repro", config=cfg) as (run, _counts):
                    generate_store_world(
                        session,
                        cfg,
                        run_id=run.run_id,
                        prefer_hf_catalogue=False,
                        use_sentence_transformers=False,
                    )
            with get_session(engine) as session:
                snaps = session.scalars(select(StateSnapshot)).all()
                return {s.snapshot_id: json.dumps(s.snapshot_json, sort_keys=True) for s in snaps}
        finally:
            engine.dispose()

    run_a = _run_world("repro_a.db")
    run_b = _run_world("repro_b.db")
    assert len(run_a) == 60
    assert run_a == run_b


def test_cli_world_build_small(tmp_path: Path) -> None:
    db_url = f"sqlite:///{tmp_path / 'cli_world.db'}"
    res = runner.invoke(
        app,
        ["world", "build", "--small", "--db-url", db_url, "--synthetic-catalogue", "--no-st"],
    )
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)
    assert data["orders"] == 300
    assert data["snapshots"] == 300
    assert data["products"] == 200
    assert data["policy_chunks"] >= 12

    engine = create_db_engine(db_url)
    try:
        with get_session(engine) as session:
            runs = list(session.scalars(select(PipelineRun)).all())
            assert len(runs) == 1
            assert runs[0].step == "world_build"
            assert runs[0].status == "completed"
    finally:
        engine.dispose()
