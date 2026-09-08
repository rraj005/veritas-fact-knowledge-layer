"""Provider metadata — single source of truth for all supported LLM providers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderInfo:
    """Metadata about a single LLM provider."""

    id: str
    label: str
    needs_key: bool
    needs_base_url: bool
    default_base_url: str | None


PROVIDERS: list[ProviderInfo] = [
    ProviderInfo(
        id="openai",
        label="OpenAI",
        needs_key=True,
        needs_base_url=False,
        default_base_url=None,
    ),
    ProviderInfo(
        id="anthropic",
        label="Anthropic (Claude)",
        needs_key=True,
        needs_base_url=False,
        default_base_url=None,
    ),
    ProviderInfo(
        id="gemini",
        label="Google Gemini",
        needs_key=True,
        needs_base_url=False,
        default_base_url=None,
    ),
    ProviderInfo(
        id="openrouter",
        label="OpenRouter",
        needs_key=True,
        needs_base_url=False,
        default_base_url=None,
    ),
    ProviderInfo(
        id="ollama",
        label="Ollama (local)",
        needs_key=False,
        needs_base_url=True,
        default_base_url="http://localhost:11434",
    ),
    ProviderInfo(
        id="custom",
        label="Custom (OpenAI-compatible)",
        needs_key=True,
        needs_base_url=True,
        default_base_url=None,
    ),
]

# Lookup by id for fast access
PROVIDER_BY_ID: dict[str, ProviderInfo] = {p.id: p for p in PROVIDERS}
