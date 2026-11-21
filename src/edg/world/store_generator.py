"""Seeded synthetic e-commerce store generator and immutable state snapshot builder."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from edg.config import AppConfig
from edg.db.models import (
    Customer,
    Order,
    OrderItem,
    Payment,
    RefundReturn,
    Shipment,
    StateSnapshot,
)
from edg.logging import get_logger
from edg.world.catalogue import ProductRecord, load_catalogue, store_catalogue
from edg.world.hard_states import (
    BOUNDARY_RETURN_WINDOW,
    CANCEL_AFTER_SHIPPING,
    DELIVERED_DISPUTED,
    DUPLICATE_CHARGE,
    PARTIAL_SHIPMENT,
    PAST_RETURN_WINDOW,
    REFUND_APPROVED_NOT_ISSUED,
    WRONG_ITEM_DELIVERED,
    assign_hard_states,
)
from edg.world.policies import load_and_store_policies

logger = get_logger(__name__)

# Fixed reference anchor date for deterministic, reproducible timestamps
ANCHOR_TIME = datetime(2026, 11, 15, 12, 0, 0, tzinfo=UTC)

_CARRIERS = ["NovaExpress", "SwiftParcel", "AtlasFreight", "MetroCourier"]
_RETURN_REASONS = [
    "damaged_in_transit",
    "item_not_as_described",
    "changed_mind",
    "defective_item",
    "missing_parts",
]


def _iso(dt: datetime | None) -> str | None:
    return dt.astimezone(UTC).isoformat() if dt is not None else None


def _generate_customers(
    session: Session,
    *,
    n_customers: int,
    run_id: str,
) -> list[Customer]:
    customers: list[Customer] = []
    for idx in range(1, n_customers + 1):
        cust = Customer(
            customer_id=f"CUST-{idx:06d}",
            name_token=f"NAME_{idx:06d}",
            email_token=f"EMAIL_{idx:06d}",
            run_id=run_id,
        )
        customers.append(cust)
        session.add(cust)
    session.flush()
    return customers


def _choose_order_status(tags: set[str], rng: random.Random) -> str:
    if CANCEL_AFTER_SHIPPING in tags:
        return rng.choice(["shipped", "out_for_delivery"])
    if PARTIAL_SHIPMENT in tags:
        return "shipped"
    if tags & {
        DELIVERED_DISPUTED,
        PAST_RETURN_WINDOW,
        BOUNDARY_RETURN_WINDOW,
        WRONG_ITEM_DELIVERED,
        REFUND_APPROVED_NOT_ISSUED,
    }:
        return "delivered"
    # Natural distribution for non-hard-state orders
    return rng.choices(
        ["placed", "shipped", "out_for_delivery", "delivered", "cancelled", "returned"],
        weights=[0.12, 0.20, 0.08, 0.42, 0.08, 0.10],
        k=1,
    )[0]


def generate_store_world(
    session: Session,
    config: AppConfig,
    *,
    run_id: str,
    prefer_hf_catalogue: bool = True,
    use_sentence_transformers: bool = True,
) -> dict[str, Any]:
    """Generate policies, catalogue, customers, orders, related records, and state snapshots."""
    rng = random.Random(config.seed)

    policy_chunks = load_and_store_policies(
        session,
        run_id=run_id,
        policy_version=config.world.policy_version,
        model_name=config.embeddings.model_name,
        embedding_dim=config.embeddings.dimension,
        use_sentence_transformers=use_sentence_transformers,
    )

    catalogue: list[ProductRecord] = load_catalogue(
        size=config.world.catalogue_size,
        seed=config.seed,
        prefer_hf=prefer_hf_catalogue,
    )
    store_catalogue(session, run_id=run_id, products=catalogue)

    customers = _generate_customers(
        session,
        n_customers=config.world.n_customers,
        run_id=run_id,
    )

    n_orders = config.world.n_orders
    hard_state_lists = assign_hard_states(n_orders, config.world.hard_state_rates, rng)

    payment_counter = 0
    shipment_counter = 0
    return_counter = 0
    hard_state_counts: dict[str, int] = {k: 0 for k in config.world.hard_state_rates}

    for order_idx in range(1, n_orders + 1):
        tags_list = hard_state_lists[order_idx - 1]
        tags = set(tags_list)
        for tag in tags_list:
            hard_state_counts[tag] = hard_state_counts.get(tag, 0) + 1

        order_id = f"ORD-{order_idx:06d}"
        customer = customers[rng.randrange(len(customers))]
        status = _choose_order_status(tags, rng)

        # Pick order items (partial_shipment always gets at least 2 items)
        min_items = 2 if PARTIAL_SHIPMENT in tags else 1
        n_items = rng.randint(min_items, 3)
        chosen_products = rng.sample(catalogue, k=min(n_items, len(catalogue)))

        items_data: list[dict[str, Any]] = []
        total_amount = 0.0
        for prod in chosen_products:
            qty = rng.randint(1, 2)
            line_total = round(qty * prod.price, 2)
            total_amount = round(total_amount + line_total, 2)
            items_data.append(
                {
                    "product_id": prod.product_id,
                    "title": prod.title,
                    "category": prod.category,
                    "qty": qty,
                    "unit_price": prod.price,
                }
            )

        # Choose placed_at far enough in the past to support all downstream dates <= ANCHOR_TIME
        if PAST_RETURN_WINDOW in tags:
            days_ago = rng.randint(48, 75)
        elif BOUNDARY_RETURN_WINDOW in tags:
            days_ago = rng.randint(36, 45)
        elif status in ("delivered", "returned"):
            days_ago = rng.randint(10, 35)
        elif status in ("shipped", "out_for_delivery"):
            days_ago = rng.randint(2, 8)
        else:
            days_ago = rng.randint(0, 4)

        placed_at = ANCHOR_TIME - timedelta(days=days_ago, hours=rng.randint(0, 12))

        order = Order(
            order_id=order_id,
            customer_id=customer.customer_id,
            placed_at=placed_at,
            status=status,
            total_amount=total_amount,
            currency="USD",
            run_id=run_id,
        )
        session.add(order)
        session.flush()

        for item in items_data:
            session.add(
                OrderItem(
                    order_id=order_id,
                    product_id=item["product_id"],
                    qty=item["qty"],
                    unit_price=item["unit_price"],
                    run_id=run_id,
                )
            )

        # Payments (charged_at >= placed_at)
        charged_at = placed_at + timedelta(minutes=rng.randint(1, 30))
        if status == "placed":
            pay_status = rng.choices(["authorised", "captured", "failed"], weights=[0.4, 0.5, 0.1], k=1)[0]
        elif status == "cancelled":
            pay_status = rng.choice(["authorised", "refunded"])
        elif status == "returned":
            pay_status = "refunded"
        else:
            pay_status = "captured"

        if DUPLICATE_CHARGE in tags:
            pay_status = "captured"

        payment_counter += 1
        primary_payment = Payment(
            payment_id=f"PAY-{payment_counter:06d}",
            order_id=order_id,
            amount=total_amount,
            status=pay_status,
            charged_at=charged_at,
            is_duplicate=False,
            run_id=run_id,
        )
        session.add(primary_payment)
        payments_list = [primary_payment]

        if DUPLICATE_CHARGE in tags:
            payment_counter += 1
            dup_payment = Payment(
                payment_id=f"PAY-{payment_counter:06d}",
                order_id=order_id,
                amount=total_amount,
                status="captured",
                charged_at=charged_at + timedelta(minutes=rng.randint(2, 15)),
                is_duplicate=True,
                run_id=run_id,
            )
            session.add(dup_payment)
            payments_list.append(dup_payment)

        # Shipments
        shipments_list: list[Shipment] = []
        actual_delivery_dt: datetime | None = None

        if status not in ("placed", "cancelled"):
            carrier = rng.choice(_CARRIERS)
            ship_delay_hours = rng.randint(12, 36)
            shipped_at = placed_at + timedelta(hours=ship_delay_hours)
            expected_delivery = shipped_at + timedelta(days=rng.randint(3, 5))

            if PARTIAL_SHIPMENT in tags:
                # Shipment 1: already delivered
                deliv_1 = shipped_at + timedelta(days=rng.randint(2, 4))
                shipment_counter += 1
                s1 = Shipment(
                    shipment_id=f"SHP-{shipment_counter:06d}",
                    order_id=order_id,
                    carrier=carrier,
                    expected_delivery=expected_delivery,
                    carrier_scan_status="delivered",
                    carrier_scan_at=deliv_1,
                    actual_delivery=deliv_1,
                    customer_disputed=False,
                    run_id=run_id,
                )
                session.add(s1)
                shipments_list.append(s1)

                # Shipment 2: still in transit
                shipment_counter += 1
                s2_scan = min(shipped_at + timedelta(days=1), ANCHOR_TIME)
                s2 = Shipment(
                    shipment_id=f"SHP-{shipment_counter:06d}",
                    order_id=order_id,
                    carrier=rng.choice(_CARRIERS),
                    expected_delivery=ANCHOR_TIME + timedelta(days=rng.randint(2, 5)),
                    carrier_scan_status="in_transit",
                    carrier_scan_at=s2_scan,
                    actual_delivery=None,
                    customer_disputed=False,
                    run_id=run_id,
                )
                session.add(s2)
                shipments_list.append(s2)
                actual_delivery_dt = deliv_1
            else:
                if status in ("delivered", "returned"):
                    transit_days = rng.randint(2, 5)
                    actual_delivery_dt = shipped_at + timedelta(days=transit_days)
                    carrier_scan_status = "delivered"
                    carrier_scan_at = actual_delivery_dt - timedelta(hours=rng.randint(1, 6))
                    disputed = DELIVERED_DISPUTED in tags
                elif status == "out_for_delivery":
                    carrier_scan_status = "out_for_delivery"
                    carrier_scan_at = min(shipped_at + timedelta(days=2), ANCHOR_TIME)
                    disputed = False
                else:  # shipped
                    carrier_scan_status = "in_transit"
                    carrier_scan_at = min(shipped_at + timedelta(hours=rng.randint(4, 24)), ANCHOR_TIME)
                    disputed = False

                shipment_counter += 1
                shp = Shipment(
                    shipment_id=f"SHP-{shipment_counter:06d}",
                    order_id=order_id,
                    carrier=carrier,
                    expected_delivery=expected_delivery,
                    carrier_scan_status=carrier_scan_status,
                    carrier_scan_at=carrier_scan_at,
                    actual_delivery=actual_delivery_dt,
                    customer_disputed=disputed,
                    run_id=run_id,
                )
                session.add(shp)
                shipments_list.append(shp)

        # Refunds & Returns
        returns_list: list[RefundReturn] = []
        needs_return = (
            status == "returned"
            or PAST_RETURN_WINDOW in tags
            or BOUNDARY_RETURN_WINDOW in tags
            or REFUND_APPROVED_NOT_ISSUED in tags
            or WRONG_ITEM_DELIVERED in tags
        )

        if needs_return and actual_delivery_dt is not None:
            if PAST_RETURN_WINDOW in tags:
                requested_at = actual_delivery_dt + timedelta(days=rng.randint(35, 45))
                ret_status = "rejected_past_window"
                refund_issued_at = None
                reason = rng.choice(_RETURN_REASONS)
            elif BOUNDARY_RETURN_WINDOW in tags:
                requested_at = actual_delivery_dt + timedelta(days=rng.choice([29, 30]))
                ret_status = "approved" if REFUND_APPROVED_NOT_ISSUED in tags else "requested"
                refund_issued_at = None
                reason = WRONG_ITEM_DELIVERED if WRONG_ITEM_DELIVERED in tags else rng.choice(_RETURN_REASONS)
            elif REFUND_APPROVED_NOT_ISSUED in tags:
                requested_at = actual_delivery_dt + timedelta(days=rng.randint(2, 10))
                ret_status = "approved"
                refund_issued_at = None
                reason = WRONG_ITEM_DELIVERED if WRONG_ITEM_DELIVERED in tags else rng.choice(_RETURN_REASONS)
            elif WRONG_ITEM_DELIVERED in tags:
                requested_at = actual_delivery_dt + timedelta(days=rng.randint(1, 7))
                ret_status = "requested"
                refund_issued_at = None
                reason = "wrong_item_delivered"
            else:
                requested_at = actual_delivery_dt + timedelta(days=rng.randint(2, 14))
                ret_status = "refunded"
                refund_issued_at = requested_at + timedelta(days=rng.randint(2, 5))
                reason = rng.choice(_RETURN_REASONS)

            return_counter += 1
            ret = RefundReturn(
                return_id=f"RET-{return_counter:06d}",
                order_id=order_id,
                requested_at=requested_at,
                reason=reason,
                status=ret_status,
                refund_amount=total_amount,
                refund_issued_at=refund_issued_at,
                run_id=run_id,
            )
            session.add(ret)
            returns_list.append(ret)

        # Frozen state snapshot
        snapshot_payload: dict[str, Any] = {
            "snapshot_id": f"SNAP-{order_idx:06d}",
            "policy_version": config.world.policy_version,
            "reference_date": _iso(ANCHOR_TIME),
            "hard_state_tags": tags_list,
            "customer": {
                "customer_id": customer.customer_id,
                "name_token": customer.name_token,
                "email_token": customer.email_token,
            },
            "order": {
                "order_id": order.order_id,
                "customer_id": order.customer_id,
                "placed_at": _iso(order.placed_at),
                "status": order.status,
                "total_amount": order.total_amount,
                "currency": order.currency,
            },
            "items": items_data,
            "payments": [
                {
                    "payment_id": p.payment_id,
                    "amount": p.amount,
                    "status": p.status,
                    "charged_at": _iso(p.charged_at),
                    "is_duplicate": p.is_duplicate,
                }
                for p in payments_list
            ],
            "shipments": [
                {
                    "shipment_id": s.shipment_id,
                    "carrier": s.carrier,
                    "expected_delivery": _iso(s.expected_delivery),
                    "carrier_scan_status": s.carrier_scan_status,
                    "carrier_scan_at": _iso(s.carrier_scan_at),
                    "actual_delivery": _iso(s.actual_delivery),
                    "customer_disputed": s.customer_disputed,
                }
                for s in shipments_list
            ],
            "refunds_returns": [
                {
                    "return_id": r.return_id,
                    "requested_at": _iso(r.requested_at),
                    "reason": r.reason,
                    "status": r.status,
                    "refund_amount": r.refund_amount,
                    "refund_issued_at": _iso(r.refund_issued_at),
                }
                for r in returns_list
            ],
        }

        session.add(
            StateSnapshot(
                snapshot_id=f"SNAP-{order_idx:06d}",
                order_id=order_id,
                policy_version=config.world.policy_version,
                snapshot_json=snapshot_payload,
                hard_state_tags=tags_list,
                run_id=run_id,
            )
        )

    session.flush()

    counts: dict[str, Any] = {
        "policy_chunks": policy_chunks,
        "products": len(catalogue),
        "catalogue_source": catalogue[0].source if catalogue else "none",
        "customers": len(customers),
        "orders": n_orders,
        "payments": payment_counter,
        "shipments": shipment_counter,
        "refunds_returns": return_counter,
        "snapshots": n_orders,
        "hard_state_counts": hard_state_counts,
    }
    logger.info("world_generated", extra=counts)
    return counts
