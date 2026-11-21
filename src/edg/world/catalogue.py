"""Product catalogue loader (Amazon Reviews 2023 metadata sample with synthetic fallback)."""

from __future__ import annotations

import random
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from edg.db.models import Product
from edg.logging import get_logger

logger = get_logger(__name__)

_SYNTHETIC_CATEGORIES: dict[str, list[tuple[str, float, float]]] = {
    "Electronics": [
        ("Wireless Noise-Cancelling Headphones", 49.99, 249.99),
        ("USB-C 65W Fast Charger", 19.99, 59.99),
        ("Bluetooth Portable Speaker", 29.99, 149.99),
        ("4K HDMI Braided Cable 6ft", 9.99, 24.99),
        ("Mechanical Gaming Keyboard", 59.99, 179.99),
        ("Smart Fitness Watch", 79.99, 299.99),
    ],
    "Home & Kitchen": [
        ("Stainless Steel Pour-Over Kettle", 24.99, 69.99),
        ("Ceramic Non-Stick Frying Pan 12in", 29.99, 89.99),
        ("HEPA Air Purifier Compact", 69.99, 199.99),
        ("Insulated Travel Tumbler 24oz", 14.99, 39.99),
        ("Digital Kitchen Food Scale", 12.99, 34.99),
    ],
    "Apparel": [
        ("Waterproof Running Jacket", 44.99, 139.99),
        ("Merino Wool Crew Socks 3-Pack", 16.99, 34.99),
        ("Cotton Fleece Pullover Hoodie", 29.99, 79.99),
        ("Lightweight Trail Running Shoes", 64.99, 159.99),
    ],
    "Books & Media": [
        ("Practical Machine Learning Systems", 34.99, 64.99),
        ("Modern Data Engineering Handbook", 29.99, 54.99),
        ("Everyday Mediterranean Cooking", 18.99, 35.99),
    ],
    "Sports & Outdoors": [
        ("Non-Slip Cork Yoga Mat", 29.99, 79.99),
        ("Adjustable Dumbbell Pair", 89.99, 249.99),
        ("Insulated Hiking Daypack 25L", 49.99, 129.99),
        ("LED Rechargeable Camping Lantern", 19.99, 49.99),
    ],
}

_BRANDS = ["NovaTech", "Aether", "Solis", "Verve", "Kinetix", "LumaHome", "PeakLine", "Cinder"]
_COLORS = ["Black", "Silver", "Navy", "Forest Green", "Sand", "Slate Gray", "Crimson"]


class ProductRecord(BaseModel):
    product_id: str
    title: str
    category: str
    price: float = Field(gt=0)
    attributes_json: dict[str, Any] = Field(default_factory=dict)
    source: Literal["amazon_catalogue_sample", "synthetic"]


def generate_synthetic_catalogue(size: int, seed: int = 42) -> list[ProductRecord]:
    """Deterministically generate synthetic store products."""
    rng = random.Random(seed)
    categories = list(_SYNTHETIC_CATEGORIES.keys())
    products: list[ProductRecord] = []

    for idx in range(1, size + 1):
        cat = categories[(idx - 1) % len(categories)]
        base_title, low_p, high_p = rng.choice(_SYNTHETIC_CATEGORIES[cat])
        brand = rng.choice(_BRANDS)
        color = rng.choice(_COLORS)
        edition = rng.choice(["Standard", "Pro", "Lite", "Plus", "Series 2"])
        price = round(rng.uniform(low_p, high_p), 2)
        products.append(
            ProductRecord(
                product_id=f"PROD-{idx:06d}",
                title=f"{brand} {base_title} ({edition}, {color})",
                category=cat,
                price=price,
                attributes_json={"brand": brand, "color": color, "edition": edition},
                source="synthetic",
            )
        )
    return products


_AMAZON_META_ELECTRONICS_JSONL = (
    "hf://datasets/McAuley-Lab/Amazon-Reviews-2023/raw/meta_categories/meta_Electronics.jsonl"
)


def _try_load_amazon_reviews_2023(size: int, seed: int) -> list[ProductRecord]:
    """Attempt streaming load from Hugging Face McAuley-Lab/Amazon-Reviews-2023 metadata."""
    from datasets import load_dataset

    rng = random.Random(seed)
    # The dataset is script-based; `datasets>=4` no longer executes dataset scripts, so stream the
    # raw metadata JSONL directly with the generic json builder (same rows, same fields).
    stream = load_dataset(
        "json",
        data_files=_AMAZON_META_ELECTRONICS_JSONL,
        split="train",
        streaming=True,
    )
    records: list[ProductRecord] = []
    for raw in stream:
        title = (raw.get("title") or "").strip()
        raw_price = raw.get("price")
        if not title:
            continue
        try:
            price = float(str(raw_price).replace("$", "").replace(",", "").strip())
        except (TypeError, ValueError):
            price = round(rng.uniform(12.99, 199.99), 2)
        if price <= 0:
            price = round(rng.uniform(12.99, 199.99), 2)

        category = str(raw.get("main_category") or "Electronics")[:128]
        details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
        attrs = {str(k): str(v) for k, v in list(details.items())[:8]}
        idx = len(records) + 1
        records.append(
            ProductRecord(
                product_id=f"PROD-{idx:06d}",
                title=title[:512],
                category=category,
                price=round(price, 2),
                attributes_json=attrs,
                source="amazon_catalogue_sample",
            )
        )
        if len(records) >= size:
            break

    if len(records) < size:
        raise RuntimeError(f"Only fetched {len(records)} products from HF, expected {size}")
    return records


def load_catalogue(
    *,
    size: int,
    seed: int = 42,
    prefer_hf: bool = True,
) -> list[ProductRecord]:
    """Load product catalogue from HF Amazon-Reviews-2023, falling back to synthetic on failure."""
    if prefer_hf:
        try:
            records = _try_load_amazon_reviews_2023(size=size, seed=seed)
            logger.info(
                "catalogue_loaded_hf",
                extra={"count": len(records), "source": "amazon_catalogue_sample"},
            )
            return records
        except Exception as exc:
            logger.warning(
                "catalogue_hf_fallback_to_synthetic",
                extra={"requested_size": size, "reason": str(exc)},
            )

    records = generate_synthetic_catalogue(size=size, seed=seed)
    logger.info("catalogue_generated_synthetic", extra={"count": len(records), "source": "synthetic"})
    return records


def store_catalogue(session: Session, *, run_id: str, products: list[ProductRecord]) -> int:
    """Persist product catalogue records into the products table."""
    for prod in products:
        existing = session.get(Product, prod.product_id)
        if existing is None:
            session.add(
                Product(
                    product_id=prod.product_id,
                    title=prod.title,
                    category=prod.category,
                    price=prod.price,
                    attributes_json=prod.attributes_json,
                    source=prod.source,
                    run_id=run_id,
                )
            )
    session.flush()
    return len(products)
