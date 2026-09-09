"""Tests for LLM client timeouts and retry behaviour.

All HTTP calls are monkeypatched — no network access needed.
"""

from __future__ import annotations

import pytest

from veritas.llm.client import (
    _LLM_TIMEOUT,
    AnthropicClient,
    GeminiClient,
    OllamaClient,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gemini_ok_response():
    """A valid Gemini success response object."""

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "candidates": [
                    {"content": {"parts": [{"text": "gemini-answer"}], "role": "model"}}
                ]
            }

    return _Resp()


def _anthropic_ok_response():
    """A valid Anthropic success response object."""

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"content": [{"text": "anthropic-answer", "type": "text"}]}

    return _Resp()


def _ollama_ok_response():
    """A valid Ollama success response object."""

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "ollama-answer"}}

    return _Resp()


# ---------------------------------------------------------------------------
# 1. Timeout is passed to httpx.post — both GeminiClient and AnthropicClient
# ---------------------------------------------------------------------------


def test_gemini_passes_timeout_to_httpx(monkeypatch):
    """GeminiClient.complete must pass a timeout > 30 s to httpx.post."""
    captured: dict = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return _gemini_ok_response()

    monkeypatch.setattr("httpx.post", fake_post)

    client = GeminiClient(model="gemini-1.5-flash", api_key="test-key")
    result = client.complete("system", "user")

    assert result == "gemini-answer"
    assert "timeout" in captured, "timeout kwarg must be passed to httpx.post"
    timeout = captured["timeout"]
    # Accept either an httpx.Timeout object or a plain numeric value.
    import httpx

    if isinstance(timeout, httpx.Timeout):
        # read timeout should be generous (> 30 s)
        assert timeout.read is not None and timeout.read > 30, (
            f"read timeout must be > 30 s, got {timeout.read}"
        )
    else:
        assert float(timeout) > 30, f"timeout must be > 30 s, got {timeout}"


def test_anthropic_passes_timeout_to_httpx(monkeypatch):
    """AnthropicClient.complete must pass a timeout > 30 s to httpx.post."""
    captured: dict = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return _anthropic_ok_response()

    monkeypatch.setattr("httpx.post", fake_post)

    client = AnthropicClient(model="claude-3-haiku-20240307", api_key="test-key")
    result = client.complete("system", "user")

    assert result == "anthropic-answer"
    assert "timeout" in captured, "timeout kwarg must be passed to httpx.post"
    timeout = captured["timeout"]
    import httpx

    if isinstance(timeout, httpx.Timeout):
        assert timeout.read is not None and timeout.read > 30, (
            f"read timeout must be > 30 s, got {timeout.read}"
        )
    else:
        assert float(timeout) > 30, f"timeout must be > 30 s, got {timeout}"


def test_ollama_passes_timeout_to_httpx(monkeypatch):
    """OllamaClient.complete must pass a timeout > 30 s to httpx.post."""
    captured: dict = {}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return _ollama_ok_response()

    monkeypatch.setattr("httpx.post", fake_post)

    client = OllamaClient(model="llama3.1")
    result = client.complete("system", "user")

    assert result == "ollama-answer"
    assert "timeout" in captured, "timeout kwarg must be passed to httpx.post"
    timeout = captured["timeout"]
    import httpx

    if isinstance(timeout, httpx.Timeout):
        assert timeout.read is not None and timeout.read > 30, (
            f"read timeout must be > 30 s, got {timeout.read}"
        )
    else:
        assert float(timeout) > 30, f"timeout must be > 30 s, got {timeout}"


def test_llm_timeout_constant_read_gt_30():
    """The module-level _LLM_TIMEOUT must have a read timeout > 30 s."""
    import httpx

    assert _LLM_TIMEOUT is not None
    assert isinstance(_LLM_TIMEOUT, httpx.Timeout)
    assert _LLM_TIMEOUT.read is not None and _LLM_TIMEOUT.read > 30, (
        f"_LLM_TIMEOUT.read must be > 30 s, got {_LLM_TIMEOUT.read}"
    )
    assert _LLM_TIMEOUT.connect is not None and _LLM_TIMEOUT.connect > 0


# ---------------------------------------------------------------------------
# 2. Retry works: two ConnectErrors then success → returns parsed text
# ---------------------------------------------------------------------------


def test_gemini_retries_on_connect_error(monkeypatch):
    """GeminiClient.complete retries on httpx.ConnectError and returns text on success."""
    import httpx

    call_count = {"n": 0}
    sleeps: list[float] = []

    def fake_post(url, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise httpx.ConnectError("connection refused")
        return _gemini_ok_response()

    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr("veritas.llm.client.time.sleep", lambda s: sleeps.append(s))

    client = GeminiClient(model="gemini-1.5-flash", api_key="k")
    result = client.complete("s", "u")

    assert result == "gemini-answer", "should return parsed text after retries"
    assert call_count["n"] == 3, f"expected 3 calls (2 failures + 1 success), got {call_count['n']}"
    assert len(sleeps) == 2, f"expected 2 sleep calls between retries, got {len(sleeps)}"


def test_anthropic_retries_on_connect_error(monkeypatch):
    """AnthropicClient.complete retries on httpx.ConnectError and returns text on success."""
    import httpx

    call_count = {"n": 0}
    sleeps: list[float] = []

    def fake_post(url, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise httpx.ConnectError("connection refused")
        return _anthropic_ok_response()

    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr("veritas.llm.client.time.sleep", lambda s: sleeps.append(s))

    client = AnthropicClient(model="claude-3-haiku-20240307", api_key="k")
    result = client.complete("s", "u")

    assert result == "anthropic-answer", "should return parsed text after retries"
    assert call_count["n"] == 3, f"expected 3 calls (2 failures + 1 success), got {call_count['n']}"
    assert len(sleeps) == 2, f"expected 2 sleep calls between retries, got {len(sleeps)}"


def test_ollama_retries_on_connect_error(monkeypatch):
    """OllamaClient.complete retries on httpx.ConnectError and returns text on success."""
    import httpx

    call_count = {"n": 0}
    sleeps: list[float] = []

    def fake_post(url, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise httpx.ConnectError("connection refused")
        return _ollama_ok_response()

    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr("veritas.llm.client.time.sleep", lambda s: sleeps.append(s))

    client = OllamaClient(model="llama3.1")
    result = client.complete("s", "u")

    assert result == "ollama-answer", "should return parsed text after retries"
    assert call_count["n"] == 3
    assert len(sleeps) == 2


# ---------------------------------------------------------------------------
# 3. Retry exhausts → RuntimeError with no url/key in message
# ---------------------------------------------------------------------------


def test_gemini_exhausted_retries_raises_runtime_error(monkeypatch):
    """GeminiClient.complete raises RuntimeError (not leaking URL/key) after all retries fail."""
    import httpx

    monkeypatch.setattr(
        "httpx.post",
        lambda *a, **kw: (_ for _ in ()).throw(httpx.ConnectError("timeout")),
    )
    monkeypatch.setattr("veritas.llm.client.time.sleep", lambda s: None)

    client = GeminiClient(model="gemini-1.5-flash", api_key="SECRET_KEY_456")
    with pytest.raises(RuntimeError) as exc_info:
        client.complete("s", "u")

    msg = str(exc_info.value)
    assert "SECRET_KEY_456" not in msg, "API key must not appear in RuntimeError message"
    assert "key=" not in msg, "'key=' query param must not appear in RuntimeError message"
    assert exc_info.value.__cause__ is None, "__cause__ must be None (raised with 'from None')"


def test_anthropic_exhausted_retries_raises_runtime_error(monkeypatch):
    """AnthropicClient.complete raises RuntimeError (no key/url) after all retries fail."""
    import httpx

    monkeypatch.setattr(
        "httpx.post",
        lambda *a, **kw: (_ for _ in ()).throw(httpx.ConnectError("timeout")),
    )
    monkeypatch.setattr("veritas.llm.client.time.sleep", lambda s: None)

    client = AnthropicClient(model="claude-3-haiku-20240307", api_key="MY_SECRET_KEY")
    with pytest.raises(RuntimeError) as exc_info:
        client.complete("s", "u")

    msg = str(exc_info.value)
    assert "MY_SECRET_KEY" not in msg, "API key must not appear in RuntimeError message"
    assert "api.anthropic.com" not in msg, "URL must not appear in RuntimeError message"
    assert exc_info.value.__cause__ is None, "__cause__ must be None (raised with 'from None')"


def test_ollama_exhausted_retries_raises_runtime_error(monkeypatch):
    """OllamaClient.complete raises RuntimeError after all retries fail."""
    import httpx

    monkeypatch.setattr(
        "httpx.post",
        lambda *a, **kw: (_ for _ in ()).throw(httpx.ConnectError("timeout")),
    )
    monkeypatch.setattr("veritas.llm.client.time.sleep", lambda s: None)

    client = OllamaClient(model="llama3.1")
    with pytest.raises(RuntimeError) as exc_info:
        client.complete("s", "u")

    msg = str(exc_info.value)
    assert "localhost" not in msg, "URL must not appear in RuntimeError message"
    assert exc_info.value.__cause__ is None


# ---------------------------------------------------------------------------
# 4. Retry on transient HTTP status codes (429, 503)
# ---------------------------------------------------------------------------


def _make_status_response(status_code: int):
    """Build a fake httpx response with a given status code that raises on raise_for_status."""
    import httpx

    class _Resp:
        def __init__(self):
            self.status_code = status_code

        def raise_for_status(self):
            request = httpx.Request("POST", "http://fake/")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=request,
                response=response,
            )

        def json(self):
            return {}

    return _Resp()


def test_gemini_retries_on_429(monkeypatch):
    """GeminiClient retries on HTTP 429 and returns text when subsequent attempt succeeds."""
    call_count = {"n": 0}
    sleeps: list[float] = []

    def fake_post(url, **kwargs):
        call_count["n"] += 1
        if call_count["n"] < 2:
            return _make_status_response(429)
        return _gemini_ok_response()

    monkeypatch.setattr("httpx.post", fake_post)
    monkeypatch.setattr("veritas.llm.client.time.sleep", lambda s: sleeps.append(s))

    client = GeminiClient(model="gemini-1.5-flash", api_key="k")
    result = client.complete("s", "u")
    assert result == "gemini-answer"
    assert call_count["n"] == 2
    assert len(sleeps) == 1
