"""Tests for AnthropicClient, GeminiClient, and OpenAICompatibleClient.

All HTTP calls are monkeypatched — no network access needed.
"""

from __future__ import annotations

from veritas.llm.client import AnthropicClient, GeminiClient, OpenAICompatibleClient

# ---------------------------------------------------------------------------
# Shared fake HTTP response
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, data: dict) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._data


# ---------------------------------------------------------------------------
# AnthropicClient
# ---------------------------------------------------------------------------


def test_anthropic_client_request_shape(monkeypatch):
    """AnthropicClient POSTs correct URL, headers, and body; returns content text."""
    captured: dict = {}

    def fake_post(url: str, *, json=None, headers=None, **_kw) -> _FakeResponse:
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _FakeResponse(
            {"content": [{"text": "anthropic-reply", "type": "text"}]}
        )

    monkeypatch.setattr("httpx.post", fake_post)

    client = AnthropicClient(model="claude-3-5-sonnet-20241022", api_key="ant-key")
    result = client.complete("You are helpful.", "What is 2+2?", max_tokens=512, temperature=0.1)

    assert result == "anthropic-reply"
    assert "api.anthropic.com" in captured["url"]
    assert "/v1/messages" in captured["url"]

    # Headers
    headers = captured["headers"]
    assert headers.get("x-api-key") == "ant-key"
    assert "anthropic-version" in headers
    assert headers.get("content-type") == "application/json"

    # Body shape
    body = captured["json"]
    assert body["model"] == "claude-3-5-sonnet-20241022"
    assert body["max_tokens"] == 512
    # temperature is intentionally omitted — some Anthropic models reject it
    assert "temperature" not in body
    assert body["system"] == "You are helpful."
    assert body["messages"] == [{"role": "user", "content": "What is 2+2?"}]


def test_anthropic_client_uses_first_content_block(monkeypatch):
    """AnthropicClient returns the text from the first content block."""
    monkeypatch.setattr(
        "httpx.post",
        lambda *a, **kw: _FakeResponse(
            {"content": [{"text": "first block", "type": "text"}, {"text": "second", "type": "text"}]}
        ),
    )
    client = AnthropicClient(model="claude-3-haiku-20240307", api_key="key")
    assert client.complete("s", "u") == "first block"


# ---------------------------------------------------------------------------
# GeminiClient
# ---------------------------------------------------------------------------


def test_gemini_client_request_shape(monkeypatch):
    """GeminiClient POSTs correct URL and body; returns candidate text."""
    captured: dict = {}

    def fake_post(url: str, *, json=None, **_kw) -> _FakeResponse:
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": "gemini-reply"}],
                            "role": "model",
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr("httpx.post", fake_post)

    client = GeminiClient(model="gemini-1.5-flash", api_key="gm-key")
    result = client.complete("Be concise.", "Capital of France?", max_tokens=256, temperature=0.3)

    assert result == "gemini-reply"

    # URL must contain model name, method, and api_key
    assert "gemini-1.5-flash" in captured["url"]
    assert "generateContent" in captured["url"]
    assert "gm-key" in captured["url"]

    # Body shape
    body = captured["json"]
    assert body["system_instruction"] == {"parts": [{"text": "Be concise."}]}
    assert body["contents"] == [{"role": "user", "parts": [{"text": "Capital of France?"}]}]
    gen_cfg = body["generationConfig"]
    assert gen_cfg["maxOutputTokens"] == 256
    assert gen_cfg["temperature"] == 0.3


def test_gemini_client_returns_first_candidate(monkeypatch):
    """GeminiClient returns text from the first candidate's first part."""
    monkeypatch.setattr(
        "httpx.post",
        lambda *a, **kw: _FakeResponse(
            {
                "candidates": [
                    {"content": {"parts": [{"text": "answer"}], "role": "model"}},
                    {"content": {"parts": [{"text": "other"}], "role": "model"}},
                ]
            }
        ),
    )
    client = GeminiClient(model="gemini-1.5-pro", api_key="key")
    assert client.complete("s", "u") == "answer"


# ---------------------------------------------------------------------------
# OpenAICompatibleClient — verifies base_url is forwarded to SDK
# ---------------------------------------------------------------------------


class _FakeCompletionMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeCompletionChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeCompletionMessage(content)


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeCompletionChoice(content)]


class _FakeChatCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    def create(self, **kwargs):
        return _FakeCompletion(self._content)


class _FakeChatResource:
    def __init__(self, content: str) -> None:
        self.completions = _FakeChatCompletions(content)


class _FakeOpenAI:
    """Fake openai.OpenAI that captures constructor kwargs."""

    from typing import ClassVar

    _instances: ClassVar[list] = []

    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs
        self.chat = _FakeChatResource("compat-reply")
        _FakeOpenAI._instances.append(self)


def test_openai_compatible_client_uses_base_url(monkeypatch):
    """OpenAICompatibleClient passes base_url to the OpenAI SDK constructor."""
    _FakeOpenAI._instances.clear()
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)

    client = OpenAICompatibleClient(
        api_key="key",
        model="mistral-7b",
        base_url="http://localhost:8000/v1",
    )
    result = client.complete("system", "user")

    assert result == "compat-reply"
    assert len(_FakeOpenAI._instances) == 1
    instance = _FakeOpenAI._instances[0]
    assert instance._kwargs.get("base_url") == "http://localhost:8000/v1"
    assert instance._kwargs.get("api_key") == "key"


def test_openai_compatible_client_no_base_url(monkeypatch):
    """OpenAICompatibleClient without base_url does not pass base_url to SDK."""
    _FakeOpenAI._instances.clear()
    monkeypatch.setattr("openai.OpenAI", _FakeOpenAI)

    client = OpenAICompatibleClient(api_key="key", model="gpt-4o", base_url=None)
    client.complete("system", "user")

    instance = _FakeOpenAI._instances[0]
    # base_url should not be set (or falsy) when None is passed
    assert not instance._kwargs.get("base_url")
