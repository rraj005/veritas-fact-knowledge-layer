from __future__ import annotations

import pytest

from veritas.llm.cache import DiskCache
from veritas.llm.client import (
    _PROVIDER_CLASSES,
    CachedClient,
    FakeLLMClient,
    OllamaClient,
    get_client,
)


def test_fake_returns_queued():
    c = FakeLLMClient(["hello"])
    assert c.complete("s", "u") == "hello"


def test_fake_callable():
    c = FakeLLMClient(lambda s, u: f"{s}|{u}")
    assert c.complete("sys", "usr") == "sys|usr"


def test_cache_avoids_second_call(tmp_path):
    calls = {"n": 0}

    def gen(s, u):
        calls["n"] += 1
        return "R"

    inner = FakeLLMClient(gen)
    cached = CachedClient(inner, DiskCache(tmp_path), model="m")
    assert cached.complete("s", "u") == "R"
    assert cached.complete("s", "u") == "R"
    assert calls["n"] == 1


def test_cache_different_inputs_hit_inner(tmp_path):
    calls = {"n": 0}

    def gen(s, u):
        calls["n"] += 1
        return f"resp-{calls['n']}"

    inner = FakeLLMClient(gen)
    cached = CachedClient(inner, DiskCache(tmp_path), model="m")
    r1 = cached.complete("s", "u1")
    r2 = cached.complete("s", "u2")
    assert calls["n"] == 2
    assert r1 != r2


# ---------------------------------------------------------------------------
# Finding 1 — get_client wraps provider in CachedClient
# ---------------------------------------------------------------------------


class _FakeSettings:
    """Minimal settings stub for get_client tests."""

    def __init__(
        self, provider: str, api_key: str | None, cache_dir, model: str = "test-model"
    ) -> None:
        self.llm_provider = provider
        self.llm_model = model
        self.openai_api_key = api_key if provider == "openai" else None
        self.ollama_base_url = "http://localhost:11434"
        self.cache_dir = cache_dir


def test_get_client_returns_cached_client_and_caches_calls(tmp_path):
    """get_client() wraps the inner client in CachedClient; repeated identical
    .complete() calls must only reach the inner client once."""
    call_count = {"n": 0}

    class _FakeInner:
        def complete(self, system, user, *, max_tokens=2048, temperature=0.0):
            call_count["n"] += 1
            return "cached-response"

    settings = _FakeSettings("openai", api_key="fake-key", cache_dir=tmp_path)

    # Patch _PROVIDER_CLASSES so OpenAIClient is replaced with our fake.
    original = _PROVIDER_CLASSES.copy()
    _PROVIDER_CLASSES["openai"] = (
        lambda **kw: _FakeInner(),
        "openai_api_key",
        "OPENAI_API_KEY",
    )

    try:
        client = get_client(settings)
        assert isinstance(client, CachedClient), f"Expected CachedClient, got {type(client)}"

        r1 = client.complete("system-prompt", "user-message")
        r2 = client.complete("system-prompt", "user-message")  # identical -> cache hit
        assert r1 == "cached-response"
        assert r2 == "cached-response"
        assert call_count["n"] == 1, (
            "Inner client should only be called once (second call hits disk cache)"
        )
    finally:
        _PROVIDER_CLASSES.clear()
        _PROVIDER_CLASSES.update(original)


def test_get_client_raises_on_missing_key(tmp_path):
    """get_client() raises RuntimeError when the API key is absent."""
    settings = _FakeSettings("openai", api_key=None, cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="BYOK"):
        get_client(settings)


def test_get_client_raises_on_unset_provider(tmp_path):
    """get_client() raises RuntimeError when provider is empty/unset."""
    settings = _FakeSettings("", api_key=None, cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="not configured"):
        get_client(settings)


def test_get_client_raises_on_unknown_provider(tmp_path):
    """get_client() raises RuntimeError for an unrecognised provider."""
    settings = _FakeSettings("badprovider", api_key="x", cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="Unknown LLM provider"):
        get_client(settings)


# ---------------------------------------------------------------------------
# Finding 2 — OllamaClient + get_client for ollama provider
# ---------------------------------------------------------------------------


def test_ollama_client_sends_correct_request_shape(monkeypatch):
    """OllamaClient POSTs the correct payload to Ollama's /api/chat endpoint
    and returns the message content from the response -- no running Ollama needed."""
    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "ollama-reply"}}

    def fake_post(url, *, json=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse()

    monkeypatch.setattr("httpx.post", fake_post)

    client = OllamaClient(model="llama3.1")
    result = client.complete("be helpful", "what is 2+2?")

    assert result == "ollama-reply"
    assert captured["url"].endswith("/api/chat")
    assert captured["json"]["model"] == "llama3.1"
    assert captured["json"]["stream"] is False
    messages = captured["json"]["messages"]
    assert messages[0] == {"role": "system", "content": "be helpful"}
    assert messages[1] == {"role": "user", "content": "what is 2+2?"}


def test_ollama_client_respects_base_url_env(monkeypatch):
    """OllamaClient picks up OLLAMA_BASE_URL from the environment."""
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://custom-host:9999")
    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "ok"}}

    def fake_post(url, *, json=None, **kwargs):
        captured["url"] = url
        return _FakeResponse()

    monkeypatch.setattr("httpx.post", fake_post)

    client = OllamaClient(model="llama3.1")
    client.complete("s", "u")
    assert "custom-host:9999" in captured["url"]


def test_get_client_ollama_no_key_required(tmp_path, monkeypatch):
    """get_client() with provider=ollama must NOT require an API key and must
    return a CachedClient wrapping an OllamaClient."""

    class _FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "hi"}}

    monkeypatch.setattr("httpx.post", lambda *a, **kw: _FakeResponse())

    settings = _FakeSettings("ollama", api_key=None, cache_dir=tmp_path, model="llama3.1")
    client = get_client(settings)
    assert isinstance(client, CachedClient), f"Expected CachedClient, got {type(client)}"
    result = client.complete("sys", "usr")
    assert result == "hi"
