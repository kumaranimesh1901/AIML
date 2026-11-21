"""Hard-state definitions and deterministic rate-accurate assignment."""

from __future__ import annotations

import random

DELIVERED_DISPUTED = "delivered_disputed"
DUPLICATE_CHARGE = "duplicate_charge"
PAST_RETURN_WINDOW = "past_return_window"
BOUNDARY_RETURN_WINDOW = "boundary_return_window"
PARTIAL_SHIPMENT = "partial_shipment"
CANCEL_AFTER_SHIPPING = "cancel_after_shipping"
REFUND_APPROVED_NOT_ISSUED = "refund_approved_not_issued"
WRONG_ITEM_DELIVERED = "wrong_item_delivered"

ALL_HARD_STATES: tuple[str, ...] = (
    DELIVERED_DISPUTED,
    DUPLICATE_CHARGE,
    PAST_RETURN_WINDOW,
    BOUNDARY_RETURN_WINDOW,
    PARTIAL_SHIPMENT,
    CANCEL_AFTER_SHIPPING,
    REFUND_APPROVED_NOT_ISSUED,
    WRONG_ITEM_DELIVERED,
)

# Tags that require an order to be delivered/returned (mutually incompatible with cancel_after_shipping)
_DELIVERY_REQUIRED_TAGS = {
    DELIVERED_DISPUTED,
    PAST_RETURN_WINDOW,
    BOUNDARY_RETURN_WINDOW,
    REFUND_APPROVED_NOT_ISSUED,
    WRONG_ITEM_DELIVERED,
}

# Tags that create/configure a return record with specific window timing
_RETURN_WINDOW_EXCLUSIVE = {PAST_RETURN_WINDOW, BOUNDARY_RETURN_WINDOW}


def assign_hard_states(
    n_orders: int,
    rates: dict[str, float],
    rng: random.Random,
) -> list[list[str]]:
    """Deterministically assign hard-state tags across n_orders matching configured rates.

    Ensures mutually incompatible states (e.g. `cancel_after_shipping` vs delivered-only states,
    or `past_return_window` vs `boundary_return_window`) are never assigned to the same order.
    """
    assignments: list[set[str]] = [set() for _ in range(n_orders)]
    all_indices = list(range(n_orders))

    # Process cancel_after_shipping first so delivered-only tags avoid those orders
    ordered_tags = [t for t in [CANCEL_AFTER_SHIPPING] if t in rates] + [
        t for t in sorted(rates.keys()) if t != CANCEL_AFTER_SHIPPING
    ]

    for tag in ordered_tags:
        rate = rates[tag]
        target_count = round(n_orders * rate)
        if target_count <= 0:
            continue

        eligible: list[int] = []
        for idx in all_indices:
            existing = assignments[idx]
            if tag == CANCEL_AFTER_SHIPPING and (existing & _DELIVERY_REQUIRED_TAGS):
                continue
            if tag in _DELIVERY_REQUIRED_TAGS and CANCEL_AFTER_SHIPPING in existing:
                continue
            if tag in _RETURN_WINDOW_EXCLUSIVE and (existing & _RETURN_WINDOW_EXCLUSIVE):
                continue
            if tag == PARTIAL_SHIPMENT and (existing & _DELIVERY_REQUIRED_TAGS):
                continue
            if tag in _DELIVERY_REQUIRED_TAGS and PARTIAL_SHIPMENT in existing:
                continue
            eligible.append(idx)

        if target_count > len(eligible):
            raise ValueError(
                f"Cannot assign {target_count} orders to hard state '{tag}' "
                f"(only {len(eligible)} eligible out of {n_orders})"
            )

        chosen = rng.sample(eligible, target_count)
        for idx in chosen:
            assignments[idx].add(tag)

    return [sorted(tags) for tags in assignments]
