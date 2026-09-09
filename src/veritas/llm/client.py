"""LLM client abstraction: Protocol, real clients (lazy-import), fake, cached wrapper."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from veritas.llm.cache import DiskCache, cache_key

# ---------------------------------------------------------------------------
# Shared timeout for raw-httpx LLM completion calls.
# LLM generation is slow — 120 s read timeout, 10 s connect timeout.
# ---------------------------------------------------------------------------
try:
    import httpx as _httpx_mod
    _LLM_TIMEOUT = _httpx_mod.Timeout(120.0, connect=10.0)
except ImportError:  # pragma: no cover — httpx always available
    _LLM_TIMEOUT = None  # type: ignore[assignment]

# HTTP status codes considered transient / retryable.
_RETRY_STATUSES = {429, 500, 502, 503, 504}
_RETRY_BACKOFFS = (1.0, 2.0, 4.0)


def _post_with_retry(
    url: str,
    *,
    json: object,
    headers: dict | None = None,
    timeout: object = None,
    attempts: int = 3,
) -> object:
    """POST *url* with *json* body, retrying on transient errors.

    Retries on:
    - ``httpx.RequestError`` (timeouts, connection errors)
    - HTTP status 429 / 500 / 502 / 503 / 504

    Raises:
        RuntimeError: After all attempts are exhausted.  The message intentionally
            omits the URL and any API key (Gemini embeds the key in the URL).
    """
    import httpx  # lazy import

    if timeout is None:
        timeout = _LLM_TIMEOUT

    for attempt in range(1, attempts + 1):
        is_last = attempt == attempts
        try:
            kw: dict = {"json": json, "timeout": timeout}
            if headers is not None:
                kw["headers"] = headers
            response = httpx.post(url, **kw)
            status = getattr(response, "status_code", None)
            if status in _RETRY_STATUSES and not is_last:
                # Transient HTTP error — sleep then retry.
                backoff = _RETRY_BACKOFFS[attempt - 1] if attempt - 1 < len(_RETRY_BACKOFFS) else _RETRY_BACKOFFS[-1]
                time.sleep(backoff)
                continue
            return response
        except httpx.RequestError:
            if is_last:
                break
            backoff = _RETRY_BACKOFFS[attempt - 1] if attempt - 1 < len(_RETRY_BACKOFFS) else _RETRY_BACKOFFS[-1]
            time.sleep(backoff)

    raise RuntimeError("LLM request failed after retries.") from None


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
        try:
            response = _post_with_retry(
                self._API_URL,
                json=payload,
                headers=headers,
                timeout=_LLM_TIMEOUT,
            )
            response.raise_for_status()  # type: ignore[union-attr]
        except RuntimeError:
            raise RuntimeError("Anthropic request failed after retries.") from None
        data = response.json()  # type: ignore[union-attr]
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
            response = _post_with_retry(url, json=payload, timeout=_LLM_TIMEOUT)
            response.raise_for_status()  # type: ignore[union-attr]
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Gemini request failed with HTTP {e.response.status_code}."
            ) from None
        except RuntimeError:
            raise RuntimeError("Gemini request failed after retries.") from None
        data = response.json()  # type: ignore[union-attr]
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
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        try:
            response = _post_with_retry(
                f"{self._base}/api/chat",
                json=payload,
                timeout=_LLM_TIMEOUT,
            )
            response.raise_for_status()  # type: ignore[union-attr]
        except RuntimeError:
            raise RuntimeError("Ollama request failed after retries.") from None
        data = response.json()  # type: ignore[union-attr]
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
