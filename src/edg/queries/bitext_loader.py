"""Bitext customer-support query loader with stratified sampling and offline fallback."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel

from edg.logging import get_logger
from edg.world.taxonomy import load_taxonomy

logger = get_logger(__name__)

SourceTag = Literal["public_bitext", "synthetic_natural", "synthetic_controlled", "synthetic_variant"]

# Offline fallback templates covering all 27 Bitext intents (tagged synthetic_natural, never public_bitext)
_FALLBACK_TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "cancel_order": [
        (
            "I want to cancel my order {{Order Number}}, can you help?",
            "I can help check whether order {{Order Number}} is still eligible for cancellation before shipping.",
        ),
        (
            "Please cancel order {{Order Number}} right away.",
            "Let me check the current shipping status of order {{Order Number}} to see if it can be cancelled.",
        ),
    ],
    "change_order": [
        (
            "Can I modify the items in my order {{Order Number}}?",
            "I will check the status of order {{Order Number}} to see if changes are still possible.",
        ),
    ],
    "change_shipping_address": [
        (
            "I entered the wrong address on order {{Order Number}}, can I change it?",
            "Shipping addresses can be updated while an order is still in placed status before shipment.",
        ),
    ],
    "check_cancellation_fee": [
        (
            "Is there a fee if I cancel order {{Order Number}}?",
            "Orders cancelled before shipping have no cancellation fee.",
        ),
    ],
    "check_invoice": [
        (
            "Can you check the invoice amount for order {{Order Number}}?",
            "I can review the payment and invoice details for order {{Order Number}}.",
        ),
    ],
    "check_payment_methods": [
        (
            "What payment methods do you accept for order {{Order Number}}?",
            "NovaMart accepts major credit and debit cards, NovaMart Store Credit, and verified bank transfers.",
        ),
    ],
    "check_refund_policy": [
        (
            "What is your return and refund policy for order {{Order Number}}?",
            "Eligible items can be returned within 30 calendar days of the actual delivery date.",
        ),
    ],
    "complaint": [
        (
            "I am really unhappy with how order {{Order Number}} was handled.",
            "I am sorry to hear about your experience with order {{Order Number}} and will review the details.",
        ),
    ],
    "contact_customer_service": [
        (
            "How can I contact customer service about order {{Order Number}}?",
            "Customer service is available daily from 8:00 AM to 10:00 PM UTC via live chat, phone, and email.",
        ),
    ],
    "contact_human_agent": [
        (
            "Can I speak to a human agent about order {{Order Number}}?",
            "I can connect you with a human support specialist available daily from 8:00 AM to 10:00 PM UTC.",
        ),
    ],
    "create_account": [
        (
            "How do I create a new account after placing order {{Order Number}}?",
            "You can create an account in the NovaMart account portal using your email address.",
        ),
    ],
    "delete_account": [
        (
            "I want to delete my account once order {{Order Number}} is finished.",
            "You can request account deletion in the portal; transaction records are retained for 7 years.",
        ),
    ],
    "delivery_options": [
        (
            "What shipping options are available for orders like {{Order Number}}?",
            "We offer Standard (3-5 business days), Expedited (2 business days), and Next-Day shipping.",
        ),
    ],
    "delivery_period": [
        (
            "How long will delivery take for order {{Order Number}}?",
            "Let me check the expected delivery date for order {{Order Number}}.",
        ),
    ],
    "edit_account": [
        (
            "How do I update my account details linked to order {{Order Number}}?",
            "You can update your profile details anytime in the NovaMart account portal.",
        ),
    ],
    "get_invoice": [
        (
            "Where can I download the invoice for order {{Order Number}}?",
            "Downloadable itemized invoices are available in your account once payment is captured.",
        ),
    ],
    "get_refund": [
        (
            "I would like a refund for order {{Order Number}}.",
            "Let me review the delivery date, return status, and refund eligibility for order {{Order Number}}.",
        ),
        (
            "Can I get my money back for order {{Order Number}}?",
            "I will check the return window and payment records for order {{Order Number}}.",
        ),
    ],
    "newsletter_subscription": [
        (
            "How do I unsubscribe from promotional emails?",
            "You can manage newsletter preferences directly in your account portal.",
        ),
    ],
    "payment_issue": [
        (
            "There is a billing problem with my payment on order {{Order Number}}.",
            "Let me check the payment records for order {{Order Number}} to verify the charge status.",
        ),
    ],
    "place_order": [
        (
            "Can you confirm that my order {{Order Number}} went through?",
            "I will check the order status and payment confirmation for order {{Order Number}}.",
        ),
    ],
    "recover_password": [
        (
            "I forgot my password and cannot view order {{Order Number}}.",
            "You can reset your forgotten password using the recovery link on the account login page.",
        ),
    ],
    "registration_problems": [
        (
            "I am having trouble registering my account to track order {{Order Number}}.",
            "Let me help troubleshoot your account registration and check order {{Order Number}}.",
        ),
    ],
    "review": [
        (
            "Where can I leave a product review for order {{Order Number}}?",
            "You can submit a review from your order history page after delivery.",
        ),
    ],
    "set_up_shipping_address": [
        (
            "How do I add a new shipping address for future orders like {{Order Number}}?",
            "You can add or edit saved shipping addresses in your account settings.",
        ),
    ],
    "switch_account": [
        (
            "Can I switch which account order {{Order Number}} is associated with?",
            "Orders remain tied to the purchasing account, though you can sign into another account anytime.",
        ),
    ],
    "track_order": [
        (
            "Where is my package for order {{Order Number}}?",
            "Let me check the latest carrier scan and expected delivery date for order {{Order Number}}.",
        ),
        (
            "Can you track shipment status for order {{Order Number}}?",
            "I will look up the carrier tracking status for order {{Order Number}} right now.",
        ),
    ],
    "track_refund": [
        (
            "What is the status of my refund for order {{Order Number}}?",
            "Let me check whether your refund for order {{Order Number}} has been approved or issued.",
        ),
    ],
}


class RawQueryItem(BaseModel):
    instruction: str
    intent: str
    category: str
    reference_answer: str | None = None
    source_tag: SourceTag = "public_bitext"


def _stratified_sample(
    items: list[RawQueryItem],
    sample_size: int,
    seed: int,
) -> list[RawQueryItem]:
    rng = random.Random(seed)
    by_intent: dict[str, list[RawQueryItem]] = defaultdict(list)
    for item in items:
        by_intent[item.intent].append(item)

    intents = sorted(by_intent.keys())
    if not intents:
        return []

    for intent in intents:
        rng.shuffle(by_intent[intent])

    total_available = len(items)
    target = min(sample_size, total_available)

    # Proportional stratified allocation with at least 1 per intent when target >= len(intents)
    selected: list[RawQueryItem] = []
    remaining_slots = target
    quotas: dict[str, int] = {}
    for intent in intents:
        pool_len = len(by_intent[intent])
        raw_q = round(target * (pool_len / total_available))
        q = max(1, min(pool_len, raw_q)) if target >= len(intents) else min(pool_len, raw_q)
        quotas[intent] = q

    # Adjust sum of quotas to match target exactly
    while sum(quotas.values()) > target:
        largest = max(intents, key=lambda i: quotas[i])
        if quotas[largest] > 1:
            quotas[largest] -= 1
        else:
            break
    while sum(quotas.values()) < target:
        candidates = [i for i in intents if quotas[i] < len(by_intent[i])]
        if not candidates:
            break
        pick = rng.choice(candidates)
        quotas[pick] += 1

    for intent in intents:
        selected.extend(by_intent[intent][: quotas[intent]])
        remaining_slots -= quotas[intent]

    rng.shuffle(selected)
    return selected[:target]


def _generate_fallback_queries(sample_size: int, seed: int) -> list[RawQueryItem]:
    rng = random.Random(seed)
    taxonomy = load_taxonomy()
    intents = sorted(taxonomy.intents.keys())
    prefixes = [
        "",
        "Hi, ",
        "Hello support, ",
        "Quick question: ",
        "Please help — ",
        "Hey there, ",
    ]
    suffixes = [
        "",
        " Thanks!",
        " Appreciate your help.",
        " Please check as soon as possible.",
        " Let me know.",
    ]

    items: list[RawQueryItem] = []
    for idx in range(sample_size):
        intent = intents[idx % len(intents)]
        cat = taxonomy.intents[intent].category
        templates = _FALLBACK_TEMPLATES.get(
            intent,
            [("I need help with order {{Order Number}}.", "I will review order {{Order Number}} for you.")],
        )
        tmpl_q, tmpl_a = rng.choice(templates)
        query = f"{rng.choice(prefixes)}{tmpl_q}{rng.choice(suffixes)}".strip()
        items.append(
            RawQueryItem(
                instruction=query,
                intent=intent,
                category=cat,
                reference_answer=tmpl_a,
                source_tag="synthetic_natural",
            )
        )
    rng.shuffle(items)
    return items


def _try_load_bitext_hf(sample_size: int, seed: int) -> list[RawQueryItem]:
    from datasets import load_dataset

    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )
    items: list[RawQueryItem] = []
    for row in ds:
        instruction = str(row.get("instruction") or "").strip()
        intent = str(row.get("intent") or "").strip()
        category = str(row.get("category") or "GENERAL").strip()
        response = str(row.get("response") or "").strip() or None
        if not instruction or not intent:
            continue
        items.append(
            RawQueryItem(
                instruction=instruction,
                intent=intent,
                category=category,
                reference_answer=response,
                source_tag="public_bitext",
            )
        )
    return _stratified_sample(items, sample_size=sample_size, seed=seed)


def load_bitext_queries(
    *,
    sample_size: int,
    seed: int = 42,
    prefer_hf: bool = True,
) -> list[RawQueryItem]:
    """Load stratified Bitext queries from HF, falling back to clearly-tagged synthetic queries."""
    if prefer_hf:
        try:
            items = _try_load_bitext_hf(sample_size=sample_size, seed=seed)
            logger.info(
                "bitext_queries_loaded_hf",
                extra={"count": len(items), "source_tag": "public_bitext"},
            )
            return items
        except Exception as exc:
            logger.warning(
                "bitext_hf_fallback_to_synthetic",
                extra={"requested_size": sample_size, "reason": str(exc)},
            )

    items = _generate_fallback_queries(sample_size=sample_size, seed=seed)
    logger.info(
        "bitext_queries_generated_synthetic",
        extra={"count": len(items), "source_tag": "synthetic_natural"},
    )
    return items
