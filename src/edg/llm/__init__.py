"""LLM client abstraction, caching, and deterministic fake for tests."""

from edg.llm.cache import LLMCache, make_cache_key, make_prompt_hash
from edg.llm.client import (
    BaseLLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMRetryExhaustedError,
    LLMValidationError,
    OllamaClient,
    OpenAICompatibleClient,
    create_llm_client,
)
from edg.llm.fake import FakeLLMClient

__all__ = [
    "BaseLLMClient",
    "FakeLLMClient",
    "LLMCache",
    "LLMError",
    "LLMRequest",
    "LLMResponse",
    "LLMRetryExhaustedError",
    "LLMValidationError",
    "OllamaClient",
    "OpenAICompatibleClient",
    "create_llm_client",
    "make_cache_key",
    "make_prompt_hash",
]
