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
