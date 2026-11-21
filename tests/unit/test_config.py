"""Unit tests for configuration loading, profiles, overrides, and deterministic hashing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from edg.config import AppConfig, load_config


def test_load_default_config() -> None:
    cfg = load_config()
    assert cfg.profile == "default"
    assert cfg.seed == 42
    assert cfg.world.n_orders == 5000
    assert cfg.queries.bitext_sample_size == 8000
    assert "v1" in cfg.sut.versions and "v2" in cfg.sut.versions
    assert cfg.sut.versions["v1"].retrieval_enabled is False
    assert cfg.sut.versions["v2"].retrieval_enabled is True


def test_load_small_config_overrides_default() -> None:
    cfg = load_config(small=True)
    assert cfg.profile == "small"
    assert cfg.world.n_orders == 300
    assert cfg.queries.bitext_sample_size == 400
    assert cfg.queries.hard_query_count == 100
    # Unchanged fields from default remain intact
    assert cfg.embeddings.model_name == "all-MiniLM-L6-v2"


def test_config_hash_is_stable() -> None:
    cfg1 = load_config(small=True)
    cfg2 = load_config(small=True)
    assert cfg1.config_hash() == cfg2.config_hash()
    assert len(cfg1.config_hash()) == 64


def test_config_hash_changes_on_parameter_change() -> None:
    cfg1 = load_config(small=True)
    cfg2 = load_config(small=True, overrides={"seed": 99})
    assert cfg1.config_hash() != cfg2.config_hash()


def test_config_hash_excludes_api_key() -> None:
    cfg1 = load_config(small=True, overrides={"llm": {"openai_api_key": "secret-a"}})
    cfg2 = load_config(small=True, overrides={"llm": {"openai_api_key": "secret-b"}})
    assert cfg1.config_hash() == cfg2.config_hash()


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EDG_DATABASE_URL", "sqlite:///custom.db")
    monkeypatch.setenv("EDG_LLM_PROVIDER", "fake")
    monkeypatch.setenv("EDG_LOG_LEVEL", "debug")
    cfg = load_config(small=True)
    assert cfg.database.url == "sqlite:///custom.db"
    assert cfg.llm.provider == "fake"
    assert cfg.logging.level == "DEBUG"


def test_invalid_hard_state_rate_raises() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"world": {"hard_state_rates": {"bad": 1.5}}})
