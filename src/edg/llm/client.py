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
            cached = self.cache.get(cache_key)
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
        )

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                text, raw = self._send(request)
                latency_ms = (time.perf_counter() - start) * 1000.0
                if self.cache is not None and not bypass_cache:
                    self.cache.set(
                        cache_key=cache_key,
                        provider=self.provider,
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
        last_error: Exception | None = None
        last_response: LLMResponse | None = None

        for attempt in range(retries + 1):
            effective_bypass = bypass_cache or (attempt > 0)
            response = self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
                schema_name=schema_name,
                bypass_cache=effective_bypass,
            )
            last_response = response
            try:
                cleaned = _strip_json_fences(response.text)
                data = json.loads(cleaned)
                parsed = schema_model.model_validate(data)
                return parsed, response
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
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
            payload["format"] = "json"

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
