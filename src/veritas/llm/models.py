"""Model discovery: query a provider for the list of models accessible with a given key."""

from __future__ import annotations

_DEFAULT_OLLAMA_BASE = "http://localhost:11434"


def list_models(
    provider: str,
    api_key: str | None = None,
    base_url: str | None = None,
) -> list[str]:
    """Return the list of model ids available to the caller for *provider*.

    Args:
        provider: "openai" or "ollama".
        api_key:  Required for "openai"; ignored for "ollama".
        base_url: Optional Ollama base URL (defaults to http://localhost:11434).

    Returns:
        Sorted list of model id strings.

    Raises:
        RuntimeError: If provider is "openai" and api_key is missing.
        ValueError:   If provider is unknown.
    """
    if provider == "openai":
        if not api_key:
            raise RuntimeError(
                "An OpenAI API key is required to list models. "
                "Provide api_key or set OPENAI_API_KEY."
            )
        import openai  # lazy import

        client = openai.OpenAI(api_key=api_key)
        resp = client.models.list()
        return sorted(m.id for m in resp.data)

    if provider == "ollama":
        import httpx  # lazy import

        base = (base_url or _DEFAULT_OLLAMA_BASE).rstrip("/")
        response = httpx.get(f"{base}/api/tags")
        response.raise_for_status()
        data = response.json()
        return sorted(m["name"] for m in data.get("models", []))

    raise ValueError(
        f"Unknown provider '{provider}'. Supported providers: openai, ollama."
    )
