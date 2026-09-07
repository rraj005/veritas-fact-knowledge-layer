"""LLM client abstraction: Protocol, real clients (lazy-import), fake, cached wrapper."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from veritas.llm.cache import DiskCache, cache_key


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str: ...


# ---------------------------------------------------------------------------
# Real clients (lazy SDK import inside __init__ so the module is importable
# without keys / SDKs installed).
# ---------------------------------------------------------------------------


class OpenAIClient:
    """Wrapper around the OpenAI Chat Completions API."""

    def __init__(self, api_key: str, model: str) -> None:
        import openai  # lazy import

        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


class OllamaClient:
    """Wrapper around the Ollama local HTTP API (no API key required)."""

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self._model = model
        self._base = (
            base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        import httpx  # lazy import

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        response = httpx.post(f"{self._base}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]


# ---------------------------------------------------------------------------
# Fake client for tests (no SDK / key needed).
# ---------------------------------------------------------------------------


class FakeLLMClient:
    """Test double: accepts either a list of queued responses or a callable."""

    def __init__(self, responses: list[str] | Callable[[str, str], str]) -> None:
        if callable(responses):
            self._fn: Callable[[str, str], str] | None = responses
            self._queue: list[str] = []
        else:
            self._fn = None
            self._queue = list(responses)

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        if self._fn is not None:
            return self._fn(system, user)
        return self._queue.pop(0)


# ---------------------------------------------------------------------------
# Cached wrapper.
# ---------------------------------------------------------------------------


class CachedClient:
    """Wraps any LLMClient and caches identical calls to disk."""

    def __init__(self, inner: LLMClient, cache: DiskCache, model: str) -> None:
        self._inner = inner
        self._cache = cache
        self._model = model

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        key = cache_key(self._model, system, user)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        result = self._inner.complete(system, user, max_tokens=max_tokens, temperature=temperature)
        self._cache.set(key, result)
        return result


# ---------------------------------------------------------------------------
# Factory.
# ---------------------------------------------------------------------------

_PROVIDER_CLASSES: dict[str, tuple] = {
    "openai": (OpenAIClient, "openai_api_key", "OPENAI_API_KEY"),
}

# Providers that do not require an API key.
_NO_KEY_PROVIDERS = {"ollama"}

_SUPPORTED_PROVIDERS = list(_PROVIDER_CLASSES) + list(_NO_KEY_PROVIDERS)


def get_client(settings: object) -> LLMClient:  # type: ignore[return]
    """Return a CachedClient wrapping the configured provider's LLMClient.

    Raises RuntimeError with a BYOK message if the required API key is absent.
    Ollama is key-free; the key check is skipped for that provider.
    Raises RuntimeError if provider is unset or unsupported.
    """
    provider: str = getattr(settings, "llm_provider", "") or ""
    model: str = getattr(settings, "llm_model", "") or ""

    if not provider:
        raise RuntimeError(
            "LLM provider is not configured. "
            f"Set LLM_PROVIDER to one of: {_SUPPORTED_PROVIDERS}."
        )

    if provider not in _PROVIDER_CLASSES and provider not in _NO_KEY_PROVIDERS:
        raise RuntimeError(
            f"Unknown LLM provider '{provider}'. "
            f"Set LLM_PROVIDER to one of: {_SUPPORTED_PROVIDERS}."
        )

    if provider in _NO_KEY_PROVIDERS:
        # Ollama: no API key required — just construct and wrap.
        base_url: str | None = getattr(settings, "ollama_base_url", None)
        inner: LLMClient = OllamaClient(model=model, base_url=base_url)
    else:
        cls, key_attr, env_var = _PROVIDER_CLASSES[provider]
        api_key: str | None = getattr(settings, key_attr, None)
        if not api_key:
            raise RuntimeError(
                f"No API key found for provider '{provider}'. "
                f"Set the {env_var} environment variable "
                f"(BYOK — your key is never stored in the repo)."
            )
        inner = cls(api_key=api_key, model=model)

    cache_dir = getattr(settings, "cache_dir", None)
    if cache_dir is not None:
        return CachedClient(inner, DiskCache(cache_dir), model=model)
    return inner
