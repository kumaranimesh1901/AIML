"""SQLAlchemy 2.0 ORM models for all EDG tables.

Supports PostgreSQL 16 + pgvector in production and SQLite as a fallback for tests.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    event,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


JSONType = JSON().with_variant(JSONB(), "postgresql")


class EmbeddingVector(TypeDecorator[list[float]]):
    """pgvector Vector(dim) on PostgreSQL, JSON-serialized list of floats on SQLite."""

    impl = Text
    cache_ok = True

    def __init__(self, dim: int = 384) -> None:
        super().__init__()
        self.dim = dim

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(Vector(self.dim))
        return dialect.type_descriptor(Text())

    def process_bind_param(self, value: list[float] | None, dialect: Any) -> Any:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return list(value)
        return json.dumps([float(x) for x in value])

    def process_result_value(self, value: Any, dialect: Any) -> list[float] | None:
        if value is None:
            return None
        if dialect.name == "postgresql":
            return [float(x) for x in value]
        if isinstance(value, str):
            return [float(x) for x in json.loads(value)]
        return [float(x) for x in value]


class Base(DeclarativeBase):
    """Base declarative class for all EDG tables."""


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    step: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    git_commit: Mapped[str] = mapped_column(String(64), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    counts_json: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class PolicyDocument(Base):
    __tablename__ = "policy_documents"

    policy_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(EmbeddingVector(384), nullable=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name_token: Mapped[str] = mapped_column(String(64), nullable=False)
    email_token: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    attributes_json: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    customer_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("customers.customer_id"), nullable=False, index=True
    )
    placed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    total_amount: Mapped[float] = mapped_column(Float, nullable=False)
    currency: Mapped[str] = mapped_column(String(8), nullable=False, default="USD")
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class OrderItem(Base):
    __tablename__ = "order_items"

    item_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("orders.order_id"), nullable=False, index=True
    )
    product_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("products.product_id"), nullable=False, index=True
    )
    qty: Mapped[int] = mapped_column(Integer, nullable=False)
    unit_price: Mapped[float] = mapped_column(Float, nullable=False)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class Payment(Base):
    __tablename__ = "payments"

    payment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("orders.order_id"), nullable=False, index=True
    )
    amount: Mapped[float] = mapped_column(Float, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    charged_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    is_duplicate: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class Shipment(Base):
    __tablename__ = "shipments"

    shipment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("orders.order_id"), nullable=False, index=True
    )
    carrier: Mapped[str] = mapped_column(String(64), nullable=False)
    expected_delivery: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    carrier_scan_status: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    carrier_scan_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actual_delivery: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    customer_disputed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class RefundReturn(Base):
    __tablename__ = "refunds_returns"

    return_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("orders.order_id"), nullable=False, index=True
    )
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    reason: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    refund_amount: Mapped[float] = mapped_column(Float, nullable=False)
    refund_issued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class StateSnapshot(Base):
    __tablename__ = "state_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("orders.order_id"), nullable=False, index=True
    )
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    hard_state_tags: Mapped[list[str]] = mapped_column(JSONType, nullable=False, default=list)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class InteractionRaw(Base):
    """Immutable, insert-only raw interaction log."""

    __tablename__ = "interactions_raw"

    interaction_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    ai_response: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    source_tag: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    intent_label: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    snapshot_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("state_snapshots.snapshot_id"), nullable=False, index=True
    )
    reference_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    injected_failure: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    raw_json: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


@event.listens_for(InteractionRaw, "before_update")
def _prevent_raw_update(mapper: Any, connection: Any, target: Any) -> None:
    raise ValueError("interactions_raw is immutable and insert-only; UPDATE is forbidden")


@event.listens_for(InteractionRaw, "before_delete")
def _prevent_raw_delete(mapper: Any, connection: Any, target: Any) -> None:
    raise ValueError("interactions_raw is immutable and insert-only; DELETE is forbidden")


class PIIMapping(Base):
    __tablename__ = "pii_mappings"

    pii_map_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pii_map_ref: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    original_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    token: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class InteractionClean(Base):
    __tablename__ = "interactions_clean"

    interaction_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("interactions_raw.interaction_id"), primary_key=True
    )
    clean_query: Mapped[str] = mapped_column(Text, nullable=False)
    clean_response: Mapped[str] = mapped_column(Text, nullable=False)
    pii_map_ref: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    dup_group_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    dup_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    is_representative: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    valid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    reject_reason: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class Embedding(Base):
    __tablename__ = "embeddings"
    __table_args__ = (
        UniqueConstraint("interaction_id", "kind", "model_name", name="uq_embedding_item_kind_model"),
        Index("ix_embeddings_hash_model", "text_hash", "model_name"),
    )

    embedding_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    interaction_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("interactions_raw.interaction_id"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False, index=True)  # query | response
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    vector: Mapped[list[float]] = mapped_column(EmbeddingVector(384), nullable=False)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class ClusteringRun(Base):
    __tablename__ = "clustering_runs"

    clustering_run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    umap_params: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    hdbscan_params: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    dbcv: Mapped[float | None] = mapped_column(Float, nullable=True)
    n_clusters: Mapped[int] = mapped_column(Integer, nullable=False)
    noise_share: Mapped[float] = mapped_column(Float, nullable=False)
    ari_vs_other_seeds: Mapped[float | None] = mapped_column(Float, nullable=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class Cluster(Base):
    __tablename__ = "clusters"
    __table_args__ = (
        UniqueConstraint("clustering_run_id", "cluster_id", name="uq_cluster_per_run"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    clustering_run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("clustering_runs.clustering_run_id"), nullable=False, index=True
    )
    size: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str | None] = mapped_column(String(256), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    mapped_intent: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    is_new_intent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    central_examples_json: Mapped[list[Any]] = mapped_column(JSONType, nullable=False, default=list)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class ClusterMembership(Base):
    __tablename__ = "cluster_membership"
    __table_args__ = (
        UniqueConstraint("interaction_id", "clustering_run_id", name="uq_membership_per_run"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    interaction_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("interactions_raw.interaction_id"), nullable=False, index=True
    )
    clustering_run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("clustering_runs.clustering_run_id"), nullable=False, index=True
    )
    cluster_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)  # -1 = noise
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    glosh_score: Mapped[float] = mapped_column(Float, nullable=False)
    umap2d_x: Mapped[float] = mapped_column(Float, nullable=False)
    umap2d_y: Mapped[float] = mapped_column(Float, nullable=False)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class PriorityScore(Base):
    __tablename__ = "priority_scores"

    interaction_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("interactions_raw.interaction_id"), primary_key=True
    )
    novelty: Mapped[float] = mapped_column(Float, nullable=False)
    small_cluster: Mapped[float] = mapped_column(Float, nullable=False)
    multi_intent: Mapped[float] = mapped_column(Float, nullable=False)
    state_conflict: Mapped[float] = mapped_column(Float, nullable=False)
    policy_boundary: Mapped[float] = mapped_column(Float, nullable=False)
    response_drift: Mapped[float] = mapped_column(Float, nullable=False)
    total_score: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class JudgeSample(Base):
    __tablename__ = "judge_samples"

    sample_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    interaction_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("interactions_raw.interaction_id"), nullable=False, index=True
    )
    stratum: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    sampling_weight: Mapped[float] = mapped_column(Float, nullable=False)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class Judgement(Base):
    __tablename__ = "judgements"
    __table_args__ = (
        Index("ix_judgements_cache", "interaction_id", "judge_version", "prompt_hash"),
    )

    judgement_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    interaction_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("interactions_raw.interaction_id"), nullable=False, index=True
    )
    judge_model: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    judge_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    evidence_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    claims_json: Mapped[list[Any]] = mapped_column(JSONType, nullable=False, default=list)
    claim_checks_json: Mapped[list[Any]] = mapped_column(JSONType, nullable=False, default=list)

    correctness_verdict: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    correctness_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    relevance_verdict: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    relevance_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    helpfulness_verdict: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    helpfulness_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completeness_verdict: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    completeness_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    safety_verdict: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    safety_score: Mapped[int | None] = mapped_column(Integer, nullable=True)

    failure_types: Mapped[list[str]] = mapped_column(JSONType, nullable=False, default=list)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    raw_output: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class Label(Base):
    __tablename__ = "labels"

    label_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    interaction_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("interactions_raw.interaction_id"), nullable=False, index=True
    )
    judgement_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("judgements.judgement_id"), nullable=False, index=True
    )
    failure_detected: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    failure_types: Mapped[list[str]] = mapped_column(JSONType, nullable=False, default=list)
    severity: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    needs_human_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    review_reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class HumanAnnotation(Base):
    __tablename__ = "human_annotations"

    annotation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    interaction_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("interactions_raw.interaction_id"), nullable=False, index=True
    )
    annotator_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    split: Mapped[str] = mapped_column(String(32), nullable=False, index=True)  # judge_dev | judge_test
    evidence_status: Mapped[str] = mapped_column(String(32), nullable=False, default="sufficient")

    correctness_verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    correctness_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    relevance_verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    relevance_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    helpfulness_verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    helpfulness_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completeness_verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    completeness_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    safety_verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    safety_score: Mapped[int | None] = mapped_column(Integer, nullable=True)

    failure_types: Mapped[list[str]] = mapped_column(JSONType, nullable=False, default=list)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class TestCase(Base):
    __tablename__ = "test_cases"

    test_case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_interaction_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("interactions_raw.interaction_id"), nullable=False, index=True
    )
    input_query: Mapped[str] = mapped_column(Text, nullable=False)
    snapshot_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("state_snapshots.snapshot_id"), nullable=False, index=True
    )
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    expected_checklist_json: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    reference_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    original_failure_types: Mapped[list[str]] = mapped_column(JSONType, nullable=False, default=list)
    severity: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    intent: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    trust_tier: Mapped[str] = mapped_column(String(32), nullable=False, index=True)  # gold | silver | variant
    # candidate | approved | rejected
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    parent_test_case_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("test_cases.test_case_id"), nullable=True, index=True
    )
    provenance_json: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class BenchmarkRelease(Base):
    __tablename__ = "benchmark_releases"

    release_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    frozen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    judge_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    counts_json: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class ReleaseItem(Base):
    __tablename__ = "release_items"
    __table_args__ = (
        UniqueConstraint("release_id", "test_case_id", name="uq_release_item"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    release_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("benchmark_releases.release_id"), nullable=False, index=True
    )
    test_case_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("test_cases.test_case_id"), nullable=False, index=True
    )
    slice: Mapped[str] = mapped_column(String(32), nullable=False, index=True)  # mined | random_control
    split: Mapped[str] = mapped_column(String(32), nullable=False, index=True)  # dev | heldout
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    model_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    base_model: Mapped[str] = mapped_column(String(128), nullable=False)
    system_prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    retrieval_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)
    run_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class EvalResult(Base):
    __tablename__ = "eval_results"
    __table_args__ = (
        UniqueConstraint("release_id", "model_id", "test_case_id", name="uq_eval_result"),
    )

    result_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    release_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("benchmark_releases.release_id"), nullable=False, index=True
    )
    model_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("model_versions.model_id"), nullable=False, index=True
    )
    test_case_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("test_cases.test_case_id"), nullable=False, index=True
    )
    response: Mapped[str] = mapped_column(Text, nullable=False)
    judgement_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("judgements.judgement_id"), nullable=True, index=True
    )
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False, index=True)
    checklist_results_json: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class RegressionReport(Base):
    __tablename__ = "regression_reports"

    report_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    release_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("benchmark_releases.release_id"), nullable=False, index=True
    )
    baseline_model_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("model_versions.model_id"), nullable=False, index=True
    )
    candidate_model_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("model_versions.model_id"), nullable=False, index=True
    )
    scope: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    metric: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    baseline_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    candidate_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    diff: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_adjusted: Mapped[float | None] = mapped_column(Float, nullable=True)
    flagged: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pipeline_runs.run_id"), nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)


class LLMCacheEntry(Base):
    __tablename__ = "llm_cache"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    response_text: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=utcnow)
