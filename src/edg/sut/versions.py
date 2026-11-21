"""Registration and lookup of SUT model versions in the model_versions table."""

from __future__ import annotations

import hashlib

from sqlalchemy.orm import Session

from edg.config import AppConfig, SUTVersionConfig
from edg.db.models import ModelVersion


def hash_system_prompt(system_prompt: str) -> str:
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()


def register_model_versions(
    session: Session,
    config: AppConfig,
    *,
    run_id: str | None = None,
) -> dict[str, ModelVersion]:
    """Upsert all SUT versions defined in config into model_versions."""
    registered: dict[str, ModelVersion] = {}
    for key, ver_cfg in config.sut.versions.items():
        mv = upsert_model_version(session, ver_cfg, model_id=key, run_id=run_id)
        registered[key] = mv
    session.flush()
    return registered


def upsert_model_version(
    session: Session,
    ver_cfg: SUTVersionConfig,
    *,
    model_id: str | None = None,
    run_id: str | None = None,
) -> ModelVersion:
    resolved_id = model_id or ver_cfg.name
    prompt_hash = hash_system_prompt(ver_cfg.system_prompt)
    existing = session.get(ModelVersion, resolved_id)
    if existing is None:
        mv = ModelVersion(
            model_id=resolved_id,
            name=ver_cfg.name,
            provider=ver_cfg.provider,
            base_model=ver_cfg.base_model,
            system_prompt_hash=prompt_hash,
            retrieval_enabled=ver_cfg.retrieval_enabled,
            config_json=ver_cfg.model_dump(mode="json"),
            run_id=run_id,
        )
        session.add(mv)
        return mv

    existing.name = ver_cfg.name
    existing.provider = ver_cfg.provider
    existing.base_model = ver_cfg.base_model
    existing.system_prompt_hash = prompt_hash
    existing.retrieval_enabled = ver_cfg.retrieval_enabled
    existing.config_json = ver_cfg.model_dump(mode="json")
    if run_id is not None:
        existing.run_id = run_id
    return existing
