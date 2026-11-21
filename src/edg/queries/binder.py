"""Bind customer queries to matching store order state snapshots and fill placeholders."""

from __future__ import annotations

import random
import re
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.db.models import StateSnapshot
from edg.queries.bitext_loader import RawQueryItem, SourceTag
from edg.world.hard_states import CANCEL_AFTER_SHIPPING, DUPLICATE_CHARGE

_PLACEHOLDER_RE = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")


class BoundQuery(BaseModel):
    query_id: str
    user_query: str
    intent: str | None = None
    category: str | None = None
    order_id: str
    snapshot_id: str
    source_tag: SourceTag
    reference_answer: str | None = None
    hard_state_tags: list[str] = Field(default_factory=list)
    generator_model: str | None = None


def _fill_placeholders(text: str, snapshot_json: dict[str, Any]) -> str:
    order = snapshot_json["order"]
    customer = snapshot_json["customer"]
    shipments = snapshot_json.get("shipments") or []
    returns = snapshot_json.get("refunds_returns") or []
    items = snapshot_json.get("items") or []

    order_id = str(order["order_id"])
    total_str = f"{order['total_amount']:.2f}"
    refund_str = (
        f"{returns[0]['refund_amount']:.2f}"
        if returns and returns[0].get("refund_amount") is not None
        else None
    )
    delivery_date = (
        shipments[0]["expected_delivery"][:10]
        if shipments and shipments[0].get("expected_delivery")
        else order["placed_at"][:10]
    )
    item_title = items[0]["title"] if items else "item"

    def _replace(match: re.Match[str]) -> str:
        key = match.group(1).strip().lower()
        # Store-contact and website placeholders are not order facts: give them neutral NovaMart
        # wording rather than an order id or customer token the judge would read as a fact.
        if "phone" in key:
            return "the NovaMart support line"
        if "hours" in key:
            return "our support hours"
        if "live chat" in key or "chat" in key:
            return "NovaMart live chat"
        if any(k in key for k in ("url", "website", "portal", "interaction", "section", "settings", "profile")):
            return "your NovaMart account page"
        if any(k in key for k in ("city", "country", "address", "location")):
            return "your delivery address"
        if "order number" in key or "order id" in key or key == "order":
            return order_id
        if "invoice" in key:
            return f"the invoice for order {order_id}"
        if "tracking" in key or "shipment" in key:
            return str(shipments[0]["shipment_id"]) if shipments else order_id
        if "refund" in key:
            return f"${refund_str}" if refund_str is not None else "the refund amount"
        if "amount" in key or "price" in key or "money" in key:
            return f"${total_str}"
        if "currency" in key:
            return "$"
        if "date" in key or "delivery" in key:
            return delivery_date
        if "email" in key:
            return str(customer["email_token"])
        if "salutation" in key:
            return ""
        if "name" in key or "client" in key or "customer" in key:
            return str(customer["name_token"])
        if "product" in key or "item" in key:
            return str(item_title)
        if "account" in key:
            return "Standard"
        if "order" in key:
            return order_id
        return "NovaMart"

    filled = _PLACEHOLDER_RE.sub(_replace, text)
    if order_id not in filled:
        filled = f"{filled} (Order {order_id})"
    return filled


def _snapshot_fits_intent(intent: str, snap: StateSnapshot) -> bool:
    data = snap.snapshot_json
    status = data["order"]["status"]
    shipments = data.get("shipments") or []
    returns = data.get("refunds_returns") or []
    payments = data.get("payments") or []
    tags = set(snap.hard_state_tags or [])

    if intent in {"track_order", "delivery_period", "delivery_options"}:
        return len(shipments) > 0
    if intent in {"get_refund", "track_refund", "check_refund_policy"}:
        return len(returns) > 0 or status in {"delivered", "returned"}
    if intent in {"cancel_order", "check_cancellation_fee"}:
        return status in {"placed", "shipped", "out_for_delivery"} or CANCEL_AFTER_SHIPPING in tags
    if intent in {"change_shipping_address", "change_order"}:
        return status in {"placed", "shipped"}
    if intent in {"payment_issue", "check_invoice", "get_invoice"}:
        return len(payments) > 0 and (
            DUPLICATE_CHARGE in tags or any(p["status"] in {"captured", "failed"} for p in payments)
        )
    return True


def bind_queries_to_snapshots(
    session: Session,
    queries: list[RawQueryItem],
    *,
    seed: int = 42,
    id_prefix: str = "QRY",
) -> list[BoundQuery]:
    """Bind each query to an appropriate StateSnapshot and fill placeholders."""
    snapshots = list(session.scalars(select(StateSnapshot).order_by(StateSnapshot.snapshot_id)).all())
    if not snapshots:
        raise ValueError("No state_snapshots found in database; run world build first.")

    rng = random.Random(seed)
    pools: dict[str, list[StateSnapshot]] = {}

    bound: list[BoundQuery] = []
    for idx, item in enumerate(queries, start=1):
        if item.intent not in pools:
            matching = [s for s in snapshots if _snapshot_fits_intent(item.intent, s)]
            pools[item.intent] = matching if matching else snapshots

        snap = rng.choice(pools[item.intent])
        filled_query = _fill_placeholders(item.instruction, snap.snapshot_json)
        filled_ref = (
            _fill_placeholders(item.reference_answer, snap.snapshot_json)
            if item.reference_answer
            else None
        )
        bound.append(
            BoundQuery(
                query_id=f"{id_prefix}-{idx:06d}",
                user_query=filled_query,
                intent=item.intent,
                category=item.category,
                order_id=snap.order_id,
                snapshot_id=snap.snapshot_id,
                source_tag=item.source_tag,
                reference_answer=filled_ref,
                hard_state_tags=list(snap.hard_state_tags or []),
            )
        )
    return bound
