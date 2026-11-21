"""Deterministic Fake LLM client for unit tests and offline development."""

from __future__ import annotations

import json
import types
from collections.abc import Callable
from typing import Any, Literal, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel

from edg.llm.cache import LLMCache
from edg.llm.client import BaseLLMClient, LLMRequest, LLMResponse

T = TypeVar("T", bound=BaseModel)


def _sample_value_for_annotation(annotation: Any, field_info: Any | None = None) -> Any:
    """Build a deterministic, schema-valid sample value for a Pydantic field annotation."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        return args[0]

    if origin in (Union, types.UnionType):
        non_none = [a for a in args if a is not type(None)]
        if not non_none:
            return None
        return _sample_value_for_annotation(non_none[0], field_info)

    if origin is list:
        return []

    if origin is dict:
        return {}

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _build_sample_dict(annotation)

    if annotation is str:
        return "sample"
    if annotation is bool:
        return True
    if annotation is int:
        # Respect simple numeric bounds if present on field metadata
        lower = 1
        if field_info is not None:
            for meta in getattr(field_info, "metadata", ()):
                if hasattr(meta, "ge") and meta.ge is not None:
                    lower = max(lower, int(meta.ge))
                if hasattr(meta, "gt") and meta.gt is not None:
                    lower = max(lower, int(meta.gt) + 1)
                if hasattr(meta, "le") and meta.le is not None:
                    lower = min(lower, int(meta.le))
        return lower
    if annotation is float:
        lower_f = 0.5
        if field_info is not None:
            for meta in getattr(field_info, "metadata", ()):
                if hasattr(meta, "ge") and meta.ge is not None:
                    lower_f = max(lower_f, float(meta.ge))
                if hasattr(meta, "le") and meta.le is not None:
                    lower_f = min(lower_f, float(meta.le))
        return lower_f
    return "sample"


def _build_sample_dict(schema_model: type[BaseModel]) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for field_name, field_info in schema_model.model_fields.items():
        if field_info.default is not None and not str(field_info.default).startswith("PydanticUndefined"):
            data[field_name] = field_info.default
        elif field_info.default_factory is not None:
            data[field_name] = field_info.get_default(call_default_factory=True)
        else:
            data[field_name] = _sample_value_for_annotation(field_info.annotation, field_info)
    return data


class FakeLLMClient(BaseLLMClient):
    """Deterministic fake LLM client with programmable failures and schema-valid JSON support."""

    provider = "fake"

    def __init__(
        self,
        *,
        default_model: str = "fake-model",
        default_response: str = "This is a deterministic fake LLM response.",
        responses: list[str] | None = None,
        handler: Callable[[LLMRequest], str] | None = None,
        fail_times: int = 0,
        always_fail: bool = False,
        max_retries: int = 3,
        retry_base_delay_seconds: float = 0.0,
        retry_backoff_factor: float = 1.0,
        cache: LLMCache | None = None,
        cache_enabled: bool = True,
    ) -> None:
        super().__init__(
            default_model=default_model,
            max_retries=max_retries,
            retry_base_delay_seconds=retry_base_delay_seconds,
            retry_backoff_factor=retry_backoff_factor,
            cache=cache,
            cache_enabled=cache_enabled,
        )
        self.default_response = default_response
        self._responses: list[str] = list(responses) if responses else []
        self._handler = handler
        self._remaining_failures = fail_times
        self.always_fail = always_fail
        self.call_count = 0
        self.requests: list[LLMRequest] = []
        self._active_schema: type[BaseModel] | None = None

    def _send(self, request: LLMRequest) -> tuple[str, dict[str, Any]]:
        self.call_count += 1
        self.requests.append(request)

        if self.always_fail or self._remaining_failures > 0:
            if self._remaining_failures > 0:
                self._remaining_failures -= 1
            raise RuntimeError(f"Simulated LLM transport failure on call #{self.call_count}")

        if self._responses:
            text = self._responses.pop(0)
            return text, {"fake": True, "call_count": self.call_count}

        if self._handler is not None:
            text = self._handler(request)
            return text, {"fake": True, "call_count": self.call_count}

        if self._active_schema is not None:
            sample = _build_sample_dict(self._active_schema)
            return json.dumps(sample), {"fake": True, "schema": self._active_schema.__name__}

        return self.default_response, {"fake": True, "call_count": self.call_count}

    def generate_json(
        self,
        prompt: str,
        schema_model: type[T],
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        max_validation_retries: int | None = None,
        bypass_cache: bool = False,
    ) -> tuple[T, LLMResponse]:
        self._active_schema = schema_model
        try:
            return super().generate_json(
                prompt=prompt,
                schema_model=schema_model,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                max_validation_retries=max_validation_retries,
                bypass_cache=bypass_cache,
            )
        finally:
            self._active_schema = None
