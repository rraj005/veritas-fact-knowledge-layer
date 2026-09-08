"""Unit tests for veritas.llm.models — all network calls are mocked."""

from __future__ import annotations

import pytest

from veritas.llm.models import list_models

# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


class _FakeModel:
    def __init__(self, model_id: str) -> None:
        self.id = model_id


class _FakeModelList:
    def __init__(self, ids: list[str]) -> None:
        self.data = [_FakeModel(i) for i in ids]


class _FakeModelsResource:
    def __init__(self, ids: list[str]) -> None:
        self._ids = ids

    def list(self) -> _FakeModelList:
        return _FakeModelList(self._ids)


class _FakeOpenAIClient:
    def __init__(self, *, api_key: str) -> None:
        self._key = api_key
        self.models = _FakeModelsResource(["gpt-4o", "gpt-3.5-turbo", "gpt-4"])


def test_openai_list_models_returns_sorted(monkeypatch):
    """list_models('openai', api_key=...) returns sorted model ids."""
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAIClient)

    result = list_models("openai", api_key="sk-test")
    assert result == sorted(result), "Model list must be sorted"
    assert "gpt-4o" in result
    assert "gpt-3.5-turbo" in result


def test_openai_missing_key_raises(monkeypatch):
    """list_models('openai') raises RuntimeError when api_key is absent."""
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAIClient)

    with pytest.raises(RuntimeError, match="API key"):
        list_models("openai", api_key=None)

    with pytest.raises(RuntimeError, match="API key"):
        list_models("openai", api_key="")


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class _FakeHttpResponse:
    def __init__(self, data: dict) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._data


def test_ollama_list_models_returns_names(monkeypatch):
    """list_models('ollama') returns model names from /api/tags."""
    captured: dict = {}

    def fake_get(url: str, **_kw) -> _FakeHttpResponse:
        captured["url"] = url
        return _FakeHttpResponse(
            {"models": [{"name": "llama3.1"}, {"name": "mistral"}]}
        )

    monkeypatch.setattr("httpx.get", fake_get)

    result = list_models("ollama")
    assert result == sorted(result), "Ollama model list must be sorted"
    assert "llama3.1" in result
    assert "mistral" in result
    assert captured["url"].endswith("/api/tags")


def test_ollama_uses_custom_base_url(monkeypatch):
    """list_models('ollama', base_url=...) uses the supplied base URL."""
    captured: dict = {}

    def fake_get(url: str, **_kw) -> _FakeHttpResponse:
        captured["url"] = url
        return _FakeHttpResponse({"models": [{"name": "phi3"}]})

    monkeypatch.setattr("httpx.get", fake_get)

    list_models("ollama", base_url="http://custom-host:9999")
    assert "custom-host:9999" in captured["url"]


def test_ollama_empty_models(monkeypatch):
    """list_models('ollama') returns [] when no models are installed."""
    monkeypatch.setattr(
        "httpx.get",
        lambda url, **_kw: _FakeHttpResponse({"models": []}),
    )
    assert list_models("ollama") == []


# ---------------------------------------------------------------------------
# Unknown provider
# ---------------------------------------------------------------------------


def test_unknown_provider_raises_value_error():
    """list_models with an unknown provider raises ValueError."""
    with pytest.raises(ValueError, match="Unknown provider"):
        list_models("bedrock", api_key="key")


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


def test_anthropic_list_models_returns_sorted(monkeypatch):
    """list_models('anthropic', api_key=...) GETs the models endpoint and returns sorted ids."""
    captured: dict = {}

    def fake_get(url: str, **kwargs) -> _FakeHttpResponse:
        captured["url"] = url
        captured["headers"] = kwargs.get("headers", {})
        return _FakeHttpResponse(
            {"data": [{"id": "claude-3-opus-20240229"}, {"id": "claude-3-5-sonnet-20241022"}]}
        )

    monkeypatch.setattr("httpx.get", fake_get)

    result = list_models("anthropic", api_key="ant-key")
    assert result == sorted(result), "Anthropic model list must be sorted"
    assert "claude-3-opus-20240229" in result
    assert "claude-3-5-sonnet-20241022" in result
    assert "api.anthropic.com" in captured["url"]
    assert captured["headers"].get("x-api-key") == "ant-key"
    assert "anthropic-version" in captured["headers"]


def test_anthropic_missing_key_raises():
    """list_models('anthropic') raises RuntimeError when api_key is absent."""
    with pytest.raises(RuntimeError, match="API key"):
        list_models("anthropic", api_key=None)

    with pytest.raises(RuntimeError, match="API key"):
        list_models("anthropic", api_key="")


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------


def test_gemini_list_models_filters_and_strips_prefix(monkeypatch):
    """list_models('gemini') keeps only generateContent models and strips 'models/' prefix."""
    captured: dict = {}

    def fake_get(url: str, **_kw) -> _FakeHttpResponse:
        captured["url"] = url
        return _FakeHttpResponse(
            {
                "models": [
                    {
                        "name": "models/gemini-1.5-pro",
                        "supportedGenerationMethods": ["generateContent", "countTokens"],
                    },
                    {
                        "name": "models/gemini-embedding-exp",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                    {
                        "name": "models/gemini-1.5-flash",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                ]
            }
        )

    monkeypatch.setattr("httpx.get", fake_get)

    result = list_models("gemini", api_key="gm-key")
    assert result == sorted(result), "Gemini model list must be sorted"
    assert "gemini-1.5-pro" in result
    assert "gemini-1.5-flash" in result
    # Embedding-only model must be excluded
    assert "gemini-embedding-exp" not in result
    # 'models/' prefix must be stripped
    for m in result:
        assert not m.startswith("models/"), f"Prefix not stripped: {m}"
    assert "gm-key" in captured["url"]


def test_gemini_missing_key_raises():
    """list_models('gemini') raises RuntimeError when api_key is absent."""
    with pytest.raises(RuntimeError, match="API key"):
        list_models("gemini", api_key=None)

    with pytest.raises(RuntimeError, match="API key"):
        list_models("gemini", api_key="")


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------


def test_openrouter_list_models_returns_sorted(monkeypatch):
    """list_models('openrouter') GETs the OpenRouter models endpoint."""
    captured: dict = {}

    def fake_get(url: str, **kwargs) -> _FakeHttpResponse:
        captured["url"] = url
        captured["headers"] = kwargs.get("headers", {})
        return _FakeHttpResponse(
            {"data": [{"id": "openai/gpt-4o"}, {"id": "anthropic/claude-3-opus"}]}
        )

    monkeypatch.setattr("httpx.get", fake_get)

    result = list_models("openrouter", api_key="or-key")
    assert result == sorted(result), "OpenRouter model list must be sorted"
    assert "openai/gpt-4o" in result
    assert "anthropic/claude-3-opus" in result
    assert "openrouter.ai" in captured["url"]
    assert captured["headers"].get("Authorization") == "Bearer or-key"


def test_openrouter_no_key_still_works(monkeypatch):
    """list_models('openrouter') can be called without an api_key."""
    monkeypatch.setattr(
        "httpx.get",
        lambda url, **_kw: _FakeHttpResponse({"data": [{"id": "meta-llama/llama-3.1-8b"}]}),
    )
    result = list_models("openrouter")
    assert "meta-llama/llama-3.1-8b" in result


# ---------------------------------------------------------------------------
# Custom (OpenAI-compatible)
# ---------------------------------------------------------------------------


def test_custom_list_models_uses_base_url(monkeypatch):
    """list_models('custom', base_url=...) GETs {base_url}/models."""
    captured: dict = {}

    def fake_get(url: str, **kwargs) -> _FakeHttpResponse:
        captured["url"] = url
        captured["headers"] = kwargs.get("headers", {})
        return _FakeHttpResponse({"data": [{"id": "mistral-7b"}, {"id": "llama-3-8b"}]})

    monkeypatch.setattr("httpx.get", fake_get)

    result = list_models("custom", api_key="my-key", base_url="http://localhost:8000/v1")
    assert result == sorted(result), "Custom model list must be sorted"
    assert "mistral-7b" in result
    assert captured["url"].endswith("/models")
    assert "localhost:8000" in captured["url"]
    assert captured["headers"].get("Authorization") == "Bearer my-key"


def test_custom_missing_base_url_raises():
    """list_models('custom') raises ValueError when base_url is absent."""
    with pytest.raises(ValueError, match="base_url"):
        list_models("custom", api_key="key", base_url=None)

    with pytest.raises(ValueError, match="base_url"):
        list_models("custom", api_key="key", base_url="")
