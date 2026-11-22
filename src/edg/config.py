"""Configuration loading, validation, and deterministic hashing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


def _find_project_root() -> Path:
    """Locate project root containing pyproject.toml or configs/."""
    cwd = Path.cwd()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "pyproject.toml").exists() or (candidate / "configs").exists():
            return candidate
    return cwd


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dictionary into base dictionary."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class DatabaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = "sqlite:///data/edg.db"
    echo: bool = False


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "text"] = "json"


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama", "openai_compatible", "fake"] = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    openai_base_url: str = "http://localhost:8000/v1"
    openai_api_key: str = ""
    # Model for helper generation: controlled-failure rewrites, checklists, cluster labels, paraphrases
    helper_model: str = "qwen2.5:7b"
    # Context window requested from Ollama; must hold the longest judge prompt plus its reply
    ollama_num_ctx: int = Field(default=8192, ge=2048)
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=3, ge=0)
    retry_base_delay_seconds: float = Field(default=1.0, ge=0.0)
    retry_backoff_factor: float = Field(default=2.0, ge=1.0)
    cache_enabled: bool = True


class WorldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_version: str = "v1"
    catalogue_size: int = Field(default=3000, ge=1)
    n_customers: int = Field(default=2000, ge=1)
    n_orders: int = Field(default=5000, ge=1)
    hard_state_rates: dict[str, float] = Field(default_factory=dict)

    @field_validator("hard_state_rates")
    @classmethod
    def _validate_rates(cls, rates: dict[str, float]) -> dict[str, float]:
        for name, rate in rates.items():
            if not (0.0 <= rate <= 1.0):
                raise ValueError(f"hard_state_rate '{name}' must be in [0, 1], got {rate}")
        return rates


class QueriesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bitext_sample_size: int = Field(default=8000, ge=1)
    hard_query_count: int = Field(default=1500, ge=0)
    hard_query_generator_model: str = "qwen2.5:3b"


class SUTVersionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    provider: Literal["ollama", "openai_compatible", "fake"] = "ollama"
    base_model: str
    retrieval_enabled: bool = False
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    system_prompt: str


class SUTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_tool_calls: int = Field(default=4, ge=1, le=10)
    failure_injection_rate: float = Field(default=0.10, ge=0.0, le=1.0)
    versions: dict[str, SUTVersionConfig] = Field(default_factory=dict)


class CleaningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    near_dup_cosine_threshold: float = Field(default=0.95, ge=0.0, le=1.0)


class EmbeddingsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = "all-MiniLM-L6-v2"
    alternative_model_name: str = "BAAI/bge-small-en-v1.5"
    dimension: int = Field(default=384, ge=1)
    batch_size: int = Field(default=64, ge=1)


class ClusteringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    umap_n_components: int = Field(default=10, ge=2)
    umap_n_neighbors: int = Field(default=30, ge=2)
    umap_min_dist: float = Field(default=0.0, ge=0.0)
    umap_metric: str = "cosine"
    umap2d_min_dist: float = Field(default=0.1, ge=0.0)
    hdbscan_min_cluster_sizes: list[int] = Field(default_factory=lambda: [15, 30, 50, 100])
    hdbscan_min_samples: list[int] = Field(default_factory=lambda: [5, 10])
    max_noise_share: float = Field(default=0.35, ge=0.0, le=1.0)
    stability_seeds: list[int] = Field(default_factory=lambda: [42, 101, 202])
    intent_purity_threshold: float = Field(default=0.50, ge=0.0, le=1.0)


class PriorityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: dict[str, float] = Field(default_factory=dict)
    judge_sample_size: int = Field(default=6000, ge=1)
    random_control_share: float = Field(default=0.20, ge=0.0, le=1.0)
    stratified_share: float = Field(default=0.40, ge=0.0, le=1.0)
    priority_share: float = Field(default=0.40, ge=0.0, le=1.0)


class JudgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama", "openai_compatible", "fake"] = "ollama"
    model_name: str = "llama3.1:8b"
    judge_version: str = "j1.4"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_retries: int = Field(default=3, ge=0)
    top_k_policies: int = Field(default=3, ge=1)
    low_confidence_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    dev_annotation_size: int = Field(default=200, ge=1)
    test_annotation_size: int = Field(default=400, ge=1)


class TestCasesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dedupe_cosine_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    silver_confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checklist_pass_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    bootstrap_resamples: int = Field(default=2000, ge=10)
    min_category_size: int = Field(default=30, ge=1)
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    flip_rate_sample_share: float = Field(default=0.05, gt=0.0, le=1.0)


class AppConfig(BaseModel):
    """Top-level validated application configuration."""

    model_config = ConfigDict(extra="forbid")

    profile: str = "default"
    seed: int = 42
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    world: WorldConfig = Field(default_factory=WorldConfig)
    queries: QueriesConfig = Field(default_factory=QueriesConfig)
    sut: SUTConfig = Field(default_factory=SUTConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    priority: PriorityConfig = Field(default_factory=PriorityConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    testcases: TestCasesConfig = Field(default_factory=TestCasesConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    def config_hash(self) -> str:
        """Compute a deterministic SHA-256 hash of the validated configuration."""
        payload = self.model_dump(mode="json")
        # Exclude secrets (API keys) from lineage hashes while keeping all behavioural params
        if "llm" in payload and "openai_api_key" in payload["llm"]:
            payload["llm"]["openai_api_key"] = ""
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_config(
    config_path: str | Path | None = None,
    *,
    small: bool = False,
    overrides: dict[str, Any] | None = None,
) -> AppConfig:
    """Load configuration from YAML, optionally applying the small profile and env overrides."""
    root = _find_project_root()
    base_path = Path(config_path) if config_path is not None else root / "configs" / "default.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {base_path}")

    with base_path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    if small:
        small_path = root / "configs" / "small.yaml"
        if not small_path.exists():
            raise FileNotFoundError(f"Small profile configuration file not found: {small_path}")
        with small_path.open("r", encoding="utf-8") as fh:
            small_raw: dict[str, Any] = yaml.safe_load(fh) or {}
        raw = _deep_merge(raw, small_raw)

    # Environment variable overrides
    env_overrides: dict[str, Any] = {}
    if db_url := os.getenv("EDG_DATABASE_URL"):
        env_overrides.setdefault("database", {})["url"] = db_url
    if ollama_url := os.getenv("EDG_OLLAMA_BASE_URL"):
        env_overrides.setdefault("llm", {})["ollama_base_url"] = ollama_url
    if openai_url := os.getenv("EDG_OPENAI_BASE_URL"):
        env_overrides.setdefault("llm", {})["openai_base_url"] = openai_url
    if openai_key := os.getenv("EDG_OPENAI_API_KEY"):
        env_overrides.setdefault("llm", {})["openai_api_key"] = openai_key
    if llm_provider := os.getenv("EDG_LLM_PROVIDER"):
        env_overrides.setdefault("llm", {})["provider"] = llm_provider
    if log_level := os.getenv("EDG_LOG_LEVEL"):
        env_overrides.setdefault("logging", {})["level"] = log_level.upper()
    if log_format := os.getenv("EDG_LOG_FORMAT"):
        env_overrides.setdefault("logging", {})["format"] = log_format.lower()

    if env_overrides:
        raw = _deep_merge(raw, env_overrides)

    if overrides:
        raw = _deep_merge(raw, overrides)

    return AppConfig.model_validate(raw)
