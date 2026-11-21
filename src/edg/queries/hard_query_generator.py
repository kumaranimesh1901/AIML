"""Hard-state synthetic query generator (multi-intent, contradictory, emotional, vague, conditional, mixed-language)."""

from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from edg.db.models import StateSnapshot
from edg.llm.client import BaseLLMClient, LLMError
from edg.logging import get_logger
from edg.queries.binder import BoundQuery
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

logger = get_logger(__name__)

HardQueryStyle = Literal[
    "multi_intent",
    "contradictory",
    "emotional",
    "vague",
    "conditional",
    "non_english_mixed",
]

HARD_QUERY_STYLES: tuple[HardQueryStyle, ...] = (
    "multi_intent",
    "contradictory",
    "emotional",
    "vague",
    "conditional",
    "non_english_mixed",
)

_HARD_STATE_INTENT: dict[str, str] = {
    DELIVERED_DISPUTED: "track_order",
    DUPLICATE_CHARGE: "payment_issue",
    PAST_RETURN_WINDOW: "get_refund",
    BOUNDARY_RETURN_WINDOW: "get_refund",
    PARTIAL_SHIPMENT: "track_order",
    CANCEL_AFTER_SHIPPING: "cancel_order",
    REFUND_APPROVED_NOT_ISSUED: "track_refund",
    WRONG_ITEM_DELIVERED: "get_refund",
}

_STYLE_TEMPLATES: dict[str, dict[HardQueryStyle, str]] = {
    DELIVERED_DISPUTED: {
        "multi_intent": "My tracking for {order_id} says delivered but nothing arrived, can you refund me now and also update my shipping address?",
        "contradictory": "Order {order_id} was never delivered to my door even though the carrier marked it delivered—please cancel it immediately.",
        "emotional": "This is ridiculous! Order {order_id} says delivered and I waited all day for nothing. Give me my ${amount} back right now!",
        "vague": "Something is wrong with {order_id}, it says done on your site but I don't have it.",
        "conditional": "If {order_id} shows delivered and I didn't get it, do I get an instant refund or do we have to wait for an investigation?",
        "non_english_mixed": "Hola, mi pedido {order_id} says delivered pero I never received the package, necesito un refund por favor.",
    },
    DUPLICATE_CHARGE: {
        "multi_intent": "I was charged twice for {order_id} (${amount} each time)—refund the extra charge and send me the corrected invoice.",
        "contradictory": "My payment for {order_id} failed, wait no it charged me twice! Why are there two charges of ${amount}?",
        "emotional": "You people billed my card twice for {order_id}! That's ${amount} taken twice, fix this immediately!",
        "vague": "Look at the billing on {order_id}, the bank statement looks doubled.",
        "conditional": "If the second ${amount} charge on {order_id} is already captured and not pending, how many days until I get it back?",
        "non_english_mixed": "Bonjour, pour order {order_id} I see deux charges of ${amount}, can you refund the duplicate?",
    },
    PAST_RETURN_WINDOW: {
        "multi_intent": "I want to return order {order_id} for a full refund and also ask if there is any exception past 30 days.",
        "contradictory": "I just got {order_id} a few days ago—actually it was over a month ago—can I still get a full refund?",
        "emotional": "I barely even opened {order_id}! It's unfair if you won't let me return it just because 30 days passed.",
        "vague": "Can I send {order_id} back? I've had it for a while.",
        "conditional": "If my return request for {order_id} is past the 30-day window, is there any way to still get store credit or a refund?",
        "non_english_mixed": "Quiero devolver order {order_id} even though pasaron mas de 30 days, is it still eligible?",
    },
    BOUNDARY_RETURN_WINDOW: {
        "multi_intent": "Today is right around day 29 or 30 since {order_id} arrived—am I still eligible to return it and how fast is the refund?",
        "contradictory": "I think {order_id} is past the return window, or maybe it's day 30 today—can you still accept my return?",
        "emotional": "Please don't reject my return for {order_id}, I requested it right on the deadline!",
        "vague": "Is it too late to return {order_id}?",
        "conditional": "If I requested the return for {order_id} on day 29 or 30 after delivery, do you still accept it?",
        "non_english_mixed": "Hoy es el dia 30 for order {order_id}, can I still return it sin problema?",
    },
    PARTIAL_SHIPMENT: {
        "multi_intent": "Only part of order {order_id} showed up today—where is the rest and can I cancel the missing items?",
        "contradictory": "Order {order_id} is marked delivered but half my items are missing, so it wasn't delivered!",
        "emotional": "Seriously, half of my order {order_id} is missing from the box! Did you forget to pack it?",
        "vague": "I opened the box for {order_id} and not everything is in here.",
        "conditional": "If order {order_id} was split into two packages, when does the second shipment arrive?",
        "non_english_mixed": "Solo llego una parte of order {order_id}, where is the second package?",
    },
    CANCEL_AFTER_SHIPPING: {
        "multi_intent": "Please cancel order {order_id} immediately and refund my ${amount} back to my card.",
        "contradictory": "I know {order_id} already shipped, but I need you to cancel it right now before it moves any further.",
        "emotional": "Stop order {order_id} right now! I changed my mind and I refuse to wait for a return!",
        "vague": "I don't want {order_id} anymore, can you undo it?",
        "conditional": "If order {order_id} has already shipped, can I refuse delivery or do I have to return it later?",
        "non_english_mixed": "Por favor cancela order {order_id} even though ya salio con the carrier.",
    },
    REFUND_APPROVED_NOT_ISSUED: {
        "multi_intent": "My refund for {order_id} says approved—when will the ${amount} actually hit my card and can I get a confirmation email?",
        "contradictory": "You told me my refund for {order_id} was already sent to my bank, so why isn't it issued yet?",
        "emotional": "My refund for {order_id} has been stuck on approved and I still don't have my ${amount}!",
        "vague": "Still waiting on my money for {order_id}.",
        "conditional": "If my refund for {order_id} is approved but not issued yet, how many business days until it posts to my card?",
        "non_english_mixed": "Mi reembolso for order {order_id} dice approved pero no money yet, cuando llega?",
    },
    WRONG_ITEM_DELIVERED: {
        "multi_intent": "You sent me the wrong item in order {order_id}—I need a free return label and a replacement shipped ASAP.",
        "contradictory": "Order {order_id} arrived on time and intact, except it's completely the wrong product!",
        "emotional": "How do you mess up order {order_id} and send the wrong product?! I'm not paying return shipping for your mistake!",
        "vague": "What came in the box for {order_id} is not what I ordered.",
        "conditional": "If I received the wrong item in order {order_id}, do you cover the return shipping fee and refund original shipping?",
        "non_english_mixed": "Me enviaron the wrong item en order {order_id}, necesito un cambio o refund gratis.",
    },
}


class HardQueryOutput(BaseModel):
    query: str = Field(min_length=8)


def _fallback_hard_query(tag: str, style: HardQueryStyle, snap: StateSnapshot) -> str:
    order = snap.snapshot_json["order"]
    template = _STYLE_TEMPLATES.get(tag, _STYLE_TEMPLATES[DELIVERED_DISPUTED])[style]
    return template.format(order_id=order["order_id"], amount=f"{order['total_amount']:.2f}")


def generate_hard_queries(
    session: Session,
    *,
    count: int,
    seed: int = 42,
    llm_client: BaseLLMClient | None = None,
    generator_model: str = "qwen2.5:3b",
) -> list[BoundQuery]:
    """Generate synthetic hard-state queries bound to real hard-state snapshots."""
    if count <= 0:
        return []

    all_snapshots = list(session.scalars(select(StateSnapshot).order_by(StateSnapshot.snapshot_id)).all())
    hard_snapshots = [s for s in all_snapshots if s.hard_state_tags]
    if not hard_snapshots:
        raise ValueError("No hard-state snapshots found in database; run world build first.")

    rng = random.Random(seed)
    use_live_llm = llm_client is not None and llm_client.provider != "fake"

    generated: list[BoundQuery] = []
    for idx in range(1, count + 1):
        snap = hard_snapshots[(idx - 1) % len(hard_snapshots)]
        tag = rng.choice(list(snap.hard_state_tags))
        style = HARD_QUERY_STYLES[(idx - 1) % len(HARD_QUERY_STYLES)]
        order_id = snap.order_id

        query_text: str | None = None
        if use_live_llm and llm_client is not None:
            prompt = (
                f"Write one realistic customer-support message (1-2 sentences) for NovaMart order {order_id}.\n"
                f"Situation: {tag.replace('_', ' ')}.\n"
                f"Style: {style.replace('_', ' ')}.\n"
                f"Include order ID {order_id}. Return JSON: {{\"query\": \"...\"}}."
            )
            try:
                parsed, _resp = llm_client.generate_json(
                    prompt=prompt,
                    schema_model=HardQueryOutput,
                    model=generator_model,
                    temperature=0.7,
                )
                candidate = parsed.query.strip()
                if order_id not in candidate:
                    candidate = f"{candidate} (Order {order_id})"
                query_text = candidate
            except LLMError as exc:
                logger.warning(
                    "hard_query_llm_fallback",
                    extra={"order_id": order_id, "tag": tag, "error": str(exc)},
                )

        if not query_text:
            query_text = _fallback_hard_query(tag, style, snap)

        generated.append(
            BoundQuery(
                query_id=f"HQRY-{idx:06d}",
                user_query=query_text,
                intent=_HARD_STATE_INTENT.get(tag, "track_order"),
                category=f"HARD_STATE:{style}",
                order_id=order_id,
                snapshot_id=snap.snapshot_id,
                source_tag="synthetic_natural",
                reference_answer=None,
                hard_state_tags=list(snap.hard_state_tags),
                generator_model=generator_model if use_live_llm else f"template:{generator_model}",
            )
        )

    logger.info(
        "hard_queries_generated",
        extra={"count": len(generated), "generator_model": generator_model, "live_llm": use_live_llm},
    )
    return generated
