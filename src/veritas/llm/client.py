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


class OpenAICompatibleClient:
    """Wrapper around the OpenAI Chat Completions API (or any compatible endpoint).

    Used for providers: openai (no base_url), openrouter (fixed base_url),
    and custom (user-supplied base_url).
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
    ) -> None:
        import openai  # lazy import

        kwargs: dict = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**kwargs)
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


# Keep the original name as an alias so existing tests importing OpenAIClient still work.
class OpenAIClient(OpenAICompatibleClient):
    """Backward-compatible alias for OpenAICompatibleClient (plain OpenAI, no base_url)."""

    def __init__(self, api_key: str, model: str) -> None:
        super().__init__(api_key=api_key, model=model, base_url=None)


class AnthropicClient:
    """Wrapper around the Anthropic Messages API via plain HTTP (no SDK)."""

    _API_URL = "https://api.anthropic.com/v1/messages"
    _API_VERSION = "2023-06-01"

    def __init__(self, model: str, api_key: str) -> None:
        self._model = model
        self._api_key = api_key

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        import httpx  # lazy import

        # temperature is omitted from the request body entirely — some Anthropic
        # models reject an explicit temperature parameter, so we never send it.
        payload = {
            "model": self._model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self._API_VERSION,
            "content-type": "application/json",
        }
        response = httpx.post(self._API_URL, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]


class GeminiClient:
    """Wrapper around the Google Generative Language API via plain HTTP (no SDK)."""

    _API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, model: str, api_key: str) -> None:
        self._model = model
        self._api_key = api_key

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> str:
        import httpx  # lazy import

        url = f"{self._API_BASE}/{self._model}:generateContent?key={self._api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        try:
            response = httpx.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Gemini request failed with HTTP {e.response.status_code}."
            ) from None
        except httpx.RequestError:
            raise RuntimeError("Gemini request failed: could not reach the API.") from None
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


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
# Central client factory.
# ---------------------------------------------------------------------------

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

_PROVIDER_CLASSES: dict[str, tuple] = {
    "openai": (OpenAIClient, "openai_api_key", "OPENAI_API_KEY"),
}

# Providers that do not require an API key.
_NO_KEY_PROVIDERS = {"ollama"}

_SUPPORTED_PROVIDERS = list(_PROVIDER_CLASSES) + list(_NO_KEY_PROVIDERS)


def client_for(
    provider: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLMClient:
    """Return the appropriate LLMClient for *provider*.

    Args:
        provider:  One of "openai", "anthropic", "gemini", "openrouter",
                   "ollama", or "custom".
        model:     The model id to use.
        api_key:   API key (required for openai/anthropic/gemini/openrouter;
                   optional for custom; ignored for ollama).
        base_url:  Required for "custom"; used as-is for "ollama" (defaults to
                   http://localhost:11434 if absent).

    Raises:
        RuntimeError: If a key-requiring provider has no api_key.
        ValueError:   If "custom" is used without base_url, or provider unknown.
    """
    if provider == "openai":
        if not api_key:
            raise RuntimeError(
                "An OpenAI API key is required. "
                "Provide api_key or set OPENAI_API_KEY."
            )
        return OpenAICompatibleClient(api_key=api_key, model=model, base_url=None)

    if provider == "anthropic":
        if not api_key:
            raise RuntimeError(
                "An Anthropic API key is required. "
                "Provide api_key or set ANTHROPIC_API_KEY."
            )
        return AnthropicClient(model=model, api_key=api_key)

    if provider == "gemini":
        if not api_key:
            raise RuntimeError(
                "A Gemini API key is required. "
                "Provide api_key or set GEMINI_API_KEY."
            )
        return GeminiClient(model=model, api_key=api_key)

    if provider == "openrouter":
        if not api_key:
            raise RuntimeError(
                "An OpenRouter API key is required. "
                "Provide api_key or set OPENROUTER_API_KEY."
            )
        return OpenAICompatibleClient(
            api_key=api_key,
            model=model,
            base_url=_OPENROUTER_BASE_URL,
        )

    if provider == "ollama":
        return OllamaClient(model=model, base_url=base_url)

    if provider == "custom":
        if not base_url:
            raise ValueError(
                "A base_url is required for the 'custom' provider. "
                "Provide the base URL of your OpenAI-compatible endpoint."
            )
        return OpenAICompatibleClient(api_key=api_key or "", model=model, base_url=base_url)

    raise ValueError(
        f"Unknown provider '{provider}'. "
        "Supported providers: openai, anthropic, gemini, openrouter, ollama, custom."
    )


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
