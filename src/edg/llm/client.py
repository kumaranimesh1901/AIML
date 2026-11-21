"""Unified LLM client supporting Ollama, OpenAI-compatible HTTP APIs, retries, and JSON validation."""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, Field, ValidationError

from edg.config import LLMConfig
from edg.llm.cache import LLMCache, make_cache_key, make_prompt_hash
from edg.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    """Base exception for LLM client errors."""


class LLMRetryExhaustedError(LLMError):
    """Raised when all retry attempts for an LLM call fail."""


class LLMValidationError(LLMError):
    """Raised when structured JSON output fails Pydantic validation after retries."""


class LLMRequest(BaseModel):
    prompt: str
    system_prompt: str | None = None
    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1)
    json_mode: bool = False
    # Optional JSON Schema for providers that can constrain decoding to it (Ollama `format`)
    json_schema: dict[str, Any] | None = None


class LLMResponse(BaseModel):
    text: str
    model: str
    provider: str
    latency_ms: float = 0.0
    cached: bool = False
    raw: dict[str, Any] = Field(default_factory=dict)


def _strip_json_fences(text: str) -> str:
    """Extract JSON payload if wrapped in markdown code fences."""
    stripped = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped)
    if fence_match:
        return fence_match.group(1).strip()
    return stripped


class BaseLLMClient(ABC):
    """Abstract base client providing retries with backoff, caching, and structured JSON parsing."""

    provider: str = "base"

    def __init__(
        self,
        *,
        default_model: str,
        max_retries: int = 3,
        retry_base_delay_seconds: float = 1.0,
        retry_backoff_factor: float = 2.0,
        cache: LLMCache | None = None,
        cache_enabled: bool = True,
    ) -> None:
        self.default_model = default_model
        self.max_retries = max_retries
        self.retry_base_delay_seconds = retry_base_delay_seconds
        self.retry_backoff_factor = retry_backoff_factor
        self.cache = cache if cache_enabled else None

    def _safe_cache_get(self, cache_key: str) -> tuple[str, float] | None:
        assert self.cache is not None
        try:
            return self.cache.get(cache_key)
        except Exception as exc:
            logger.warning("llm_cache_read_failed", extra={"provider": self.provider, "error": str(exc)})
            return None

    def _safe_cache_set(
        self,
        *,
        cache_key: str,
        model: str,
        prompt_hash: str,
        response_text: str,
        latency_ms: float,
    ) -> None:
        assert self.cache is not None
        try:
            self.cache.set(
                cache_key=cache_key,
                provider=self.provider,
                model=model,
                prompt_hash=prompt_hash,
                response_text=response_text,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            logger.warning("llm_cache_write_failed", extra={"provider": self.provider, "error": str(exc)})

    @abstractmethod
    def _send(self, request: LLMRequest) -> tuple[str, dict[str, Any]]:
        """Perform a single raw call to the underlying provider and return `(text, raw_payload)`."""

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        json_mode: bool = False,
        schema_name: str | None = None,
        bypass_cache: bool = False,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate text with caching and exponential backoff retries."""
        resolved_model = model or self.default_model
        prompt_hash = make_prompt_hash(system_prompt, prompt)
        cache_key = make_cache_key(
            provider=self.provider,
            model=resolved_model,
            prompt_hash=prompt_hash,
            temperature=temperature,
            schema_name=schema_name,
        )

        if self.cache is not None and not bypass_cache:
            cached = self._safe_cache_get(cache_key)
            if cached is not None:
                text, latency_ms = cached
                return LLMResponse(
                    text=text,
                    model=resolved_model,
                    provider=self.provider,
                    latency_ms=latency_ms,
                    cached=True,
                )

        request = LLMRequest(
            prompt=prompt,
            system_prompt=system_prompt,
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            json_schema=json_schema,
        )

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                text, raw = self._send(request)
                latency_ms = (time.perf_counter() - start) * 1000.0
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self.retry_base_delay_seconds * (self.retry_backoff_factor**attempt)
                    logger.warning(
                        "llm_call_retry",
                        extra={
                            "provider": self.provider,
                            "model": resolved_model,
                            "attempt": attempt + 1,
                            "max_retries": self.max_retries,
                            "delay_seconds": delay,
                            "error": str(exc),
                        },
                    )
                    if delay > 0:
                        time.sleep(delay)
                continue

            # The provider call succeeded: a cache backend problem must never be retried as an
            # LLM failure (which would re-send the request and eventually discard a good answer).
            if self.cache is not None and not bypass_cache:
                self._safe_cache_set(
                    cache_key=cache_key,
                    model=resolved_model,
                    prompt_hash=prompt_hash,
                    response_text=text,
                    latency_ms=latency_ms,
                )
            return LLMResponse(
                text=text,
                model=resolved_model,
                provider=self.provider,
                latency_ms=latency_ms,
                cached=False,
                raw=raw,
            )

        raise LLMRetryExhaustedError(
            f"LLM call failed after {self.max_retries + 1} attempts "
            f"({self.provider}/{resolved_model}): {last_exc}"
        ) from last_exc

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
        """Generate and validate a Pydantic-structured JSON response, retrying on parse errors."""
        retries = self.max_retries if max_validation_retries is None else max_validation_retries
        schema_name = schema_model.__name__
        json_schema = schema_model.model_json_schema()
        # Show the exact target shape: small local models cannot infer it from a class name, and
        # providers without schema-constrained decoding only see the prompt.
        base_prompt = (
            f"{prompt}\n\nRespond with ONLY a JSON object that conforms to this JSON Schema:\n"
            f"{json.dumps(json_schema, sort_keys=True)}"
        )
        attempt_prompt = base_prompt
        last_error: Exception | None = None
        last_response: LLMResponse | None = None

        for attempt in range(retries + 1):
            effective_bypass = bypass_cache or (attempt > 0)
            response = self.generate(
                prompt=attempt_prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
                schema_name=schema_name,
                bypass_cache=effective_bypass,
                json_schema=json_schema,
            )
            last_response = response
            try:
                cleaned = _strip_json_fences(response.text)
                data = json.loads(cleaned)
                parsed = schema_model.model_validate(data)
                return parsed, response
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                # Corrective retry: re-sending the identical prompt reproduces the identical invalid
                # output at temperature 0, so tell the model what was wrong.
                attempt_prompt = (
                    f"{base_prompt}\n\nYour previous reply was invalid:\n{response.text[:1000]}\n"
                    f"Validation error: {str(exc)[:600]}\n"
                    "Reply again with ONLY a corrected JSON object that conforms to the schema above."
                )
                logger.warning(
                    "llm_json_validation_retry",
                    extra={
                        "provider": self.provider,
                        "model": response.model,
                        "schema": schema_name,
                        "attempt": attempt + 1,
                        "error": str(exc),
                    },
                )

        raise LLMValidationError(
            f"Failed to parse valid {schema_name} JSON after {retries + 1} attempts: {last_error}"
            + (f" | last_output={last_response.text[:200]!r}" if last_response else "")
        ) from last_error


class OllamaClient(BaseLLMClient):
    """HTTP client for a local Ollama server."""

    provider = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        default_model: str = "qwen2.5:1.5b",
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        retry_base_delay_seconds: float = 1.0,
        retry_backoff_factor: float = 2.0,
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
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _send(self, request: LLMRequest) -> tuple[str, dict[str, Any]]:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": request.temperature},
        }
        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens
        if request.json_mode:
            # Ollama structured outputs: a JSON Schema constrains decoding to that exact shape
            payload["format"] = request.json_schema if request.json_schema else "json"

        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("message", {}).get("content", "")
            return text, data


class OpenAICompatibleClient(BaseLLMClient):
    """HTTP client for OpenAI-compatible chat completion endpoints."""

    provider = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "",
        default_model: str = "qwen2.5:1.5b",
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        retry_base_delay_seconds: float = 1.0,
        retry_backoff_factor: float = 2.0,
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
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def _send(self, request: LLMRequest) -> tuple[str, dict[str, Any]]:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return text, data


def create_llm_client(
    config: LLMConfig,
    *,
    provider: str | None = None,
    default_model: str = "qwen2.5:1.5b",
    cache: LLMCache | None = None,
) -> BaseLLMClient:
    """Instantiate the configured LLM client."""
    resolved_provider = provider or config.provider
    if resolved_provider == "ollama":
        return OllamaClient(
            base_url=config.ollama_base_url,
            default_model=default_model,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            retry_base_delay_seconds=config.retry_base_delay_seconds,
            retry_backoff_factor=config.retry_backoff_factor,
            cache=cache,
            cache_enabled=config.cache_enabled,
        )
    if resolved_provider == "openai_compatible":
        return OpenAICompatibleClient(
            base_url=config.openai_base_url,
            api_key=config.openai_api_key,
            default_model=default_model,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            retry_base_delay_seconds=config.retry_base_delay_seconds,
            retry_backoff_factor=config.retry_backoff_factor,
            cache=cache,
            cache_enabled=config.cache_enabled,
        )
    if resolved_provider == "fake":
        from edg.llm.fake import FakeLLMClient

        return FakeLLMClient(
            default_model=default_model,
            max_retries=config.max_retries,
            retry_base_delay_seconds=0.0,
            retry_backoff_factor=1.0,
            cache=cache,
            cache_enabled=config.cache_enabled,
        )
    raise ValueError(f"Unsupported LLM provider: {resolved_provider}")
