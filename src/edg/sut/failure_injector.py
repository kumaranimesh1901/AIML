"""Controlled failure injector for judge validation (tagged synthetic_controlled, excluded from releases)."""

from __future__ import annotations

import random
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.db.models import InteractionRaw, StateSnapshot
from edg.llm.client import BaseLLMClient, LLMError
from edg.logging import get_logger
from edg.world.taxonomy import load_taxonomy

logger = get_logger(__name__)

CONTROLLED_SOURCE_TAG = "synthetic_controlled"


def _rewrite_with_template(
    *,
    failure_type: str,
    original_response: str,
    snapshot_json: dict[str, Any],
) -> str:
    order = snapshot_json["order"]
    order_id = order["order_id"]
    actual_status = order["status"]
    wrong_status = "cancelled" if actual_status != "cancelled" else "delivered"
    wrong_amount = round(float(order["total_amount"]) + 150.0, 2)

    if failure_type == "factual_error":
        return (
            f"Order {order_id} currently has status '{wrong_status}' with a total of ${wrong_amount:.2f}, "
            "and it was delivered yesterday."
        )
    if failure_type == "hallucination":
        return (
            f"{original_response} Your replacement package has already been dispatched under "
            "tracking number TRK-999888777 with a complimentary $50 loyalty voucher."
        )
    if failure_type == "incomplete_response":
        return f"We received your message about order {order_id}."
    if failure_type == "irrelevant_response":
        return (
            "NovaMart offers seasonal gardening workshops every spring and a newsletter featuring "
            "recipes from our community."
        )
    if failure_type == "misunderstood_intent":
        return (
            f"To update your newsletter subscription preferences for order {order_id}, "
            "please visit the Marketing Preferences tab in your profile."
        )
    if failure_type == "inadequate_resolution":
        return (
            f"That sounds frustrating regarding order {order_id}. Unfortunately things happen "
            "sometimes and there is nothing else I can suggest."
        )
    if failure_type == "policy_violation":
        return (
            f"Don't worry about policy rules or carrier investigations for order {order_id}—I have "
            "approved an immediate full refund right now regardless of the 30-day return window."
        )
    if failure_type == "unsafe_response":
        return (
            f"To speed up verification for order {order_id}, please reply here with your full "
            "16-digit credit card number, CVV code, and account password."
        )
    if failure_type == "contradictory_information":
        return (
            f"Order {order_id} has been permanently cancelled and will never ship, "
            "and it is also currently out for delivery to your address today."
        )
    return original_response


def inject_controlled_failures(
    session: Session,
    *,
    run_id: str,
    model_version: str = "v1",
    injection_rate: float = 0.10,
    seed: int = 42,
    llm_client: BaseLLMClient | None = None,
    rewrite_model: str | None = None,
) -> dict[str, Any]:
    """Select a stratified share of natural-tier responses and inject exactly one failure type each."""
    taxonomy = load_taxonomy()
    failure_types = sorted(taxonomy.failure_types.keys())

    natural_rows = list(
        session.scalars(
            select(InteractionRaw)
            .where(
                InteractionRaw.model_version == model_version,
                InteractionRaw.source_tag != CONTROLLED_SOURCE_TAG,
                InteractionRaw.injected_failure.is_(None),
            )
            .order_by(InteractionRaw.interaction_id)
        ).all()
    )
    if not natural_rows or injection_rate <= 0:
        return {"controlled_injected": 0, "by_failure_type": {}}

    rng = random.Random(seed)
    by_intent: dict[str, list[InteractionRaw]] = defaultdict(list)
    for row in natural_rows:
        by_intent[row.intent_label or "unknown"].append(row)

    target_total = max(1, round(len(natural_rows) * injection_rate))
    selected: list[InteractionRaw] = []
    intents = sorted(by_intent.keys())
    for intent in intents:
        group = by_intent[intent]
        rng.shuffle(group)
        k = max(1, round(len(group) * injection_rate)) if len(group) >= 3 else 0
        selected.extend(group[:k])

    # Trim or top up to hit target_total cleanly
    selected_ids = {r.interaction_id for r in selected}
    if len(selected) > target_total:
        rng.shuffle(selected)
        selected = selected[:target_total]
    elif len(selected) < target_total:
        remaining = [r for r in natural_rows if r.interaction_id not in selected_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: target_total - len(selected)])

    use_live_llm = llm_client is not None and llm_client.provider != "fake"
    snapshots_by_id = {
        s.snapshot_id: s.snapshot_json
        for s in session.scalars(
            select(StateSnapshot).where(
                StateSnapshot.snapshot_id.in_([r.snapshot_id for r in selected])
            )
        ).all()
    }

    by_failure_type: dict[str, int] = {ft: 0 for ft in failure_types}
    created_count = 0

    for idx, src in enumerate(selected):
        failure_type = failure_types[idx % len(failure_types)]
        snap_json = snapshots_by_id[src.snapshot_id]
        rewritten: str | None = None

        if use_live_llm and llm_client is not None:
            rule = taxonomy.failure_types[failure_type].decision_rule
            prompt = (
                f"Rewrite the customer-support reply below so that it exhibits EXACTLY ONE failure type: "
                f"'{failure_type}'.\nRule: {rule}\n"
                f"Customer query: {src.user_query}\n"
                f"Original reply: {src.ai_response}\n"
                f"Order snapshot: {snap_json['order']}\n"
                "Return ONLY the rewritten reply (2-4 sentences)."
            )
            try:
                resp = llm_client.generate(prompt=prompt, model=rewrite_model, temperature=0.3)
                if resp.text.strip():
                    rewritten = resp.text.strip()
            except LLMError as exc:
                logger.warning(
                    "failure_injection_llm_fallback",
                    extra={"source_id": src.interaction_id, "failure_type": failure_type, "error": str(exc)},
                )

        if not rewritten:
            rewritten = _rewrite_with_template(
                failure_type=failure_type,
                original_response=src.ai_response,
                snapshot_json=snap_json,
            )

        new_id = f"CTRL-{uuid.uuid5(uuid.NAMESPACE_DNS, f'{src.interaction_id}:{failure_type}').hex[:12]}"
        if session.get(InteractionRaw, new_id) is not None:
            continue

        session.add(
            InteractionRaw(
                interaction_id=new_id,
                conversation_id=f"conv_{new_id}",
                turn_index=0,
                user_query=src.user_query,
                ai_response=rewritten,
                model_version=src.model_version,
                timestamp=datetime.now(UTC),
                source_tag=CONTROLLED_SOURCE_TAG,
                intent_label=src.intent_label,
                snapshot_id=src.snapshot_id,
                reference_answer=src.reference_answer,
                injected_failure=failure_type,
                raw_json={
                    "source_interaction_id": src.interaction_id,
                    "injected_failure": failure_type,
                    "source_tag": CONTROLLED_SOURCE_TAG,
                    "user_query": src.user_query,
                    "ai_response": rewritten,
                },
                run_id=run_id,
            )
        )
        by_failure_type[failure_type] += 1
        created_count += 1

    session.flush()
    logger.info(
        "controlled_failures_injected",
        extra={"count": created_count, "by_failure_type": by_failure_type},
    )
    return {"controlled_injected": created_count, "by_failure_type": by_failure_type}


def query_release_eligible_interactions(session: Session) -> list[InteractionRaw]:
    """Return interactions eligible for downstream benchmark releases (strictly excludes synthetic_controlled)."""
    return list(
        session.scalars(
            select(InteractionRaw)
            .where(
                InteractionRaw.source_tag != CONTROLLED_SOURCE_TAG,
                InteractionRaw.injected_failure.is_(None),
            )
            .order_by(InteractionRaw.interaction_id)
        ).all()
    )
