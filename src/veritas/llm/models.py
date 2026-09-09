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
        provider: One of "openai", "ollama", "anthropic", "gemini",
                  "openrouter", or "custom".
        api_key:  Required for "openai", "anthropic", "gemini".
                  Optional for "openrouter". Ignored for "ollama".
        base_url: Required for "custom". Optional for "ollama"
                  (defaults to http://localhost:11434).

    Returns:
        Sorted list of model id strings.

    Raises:
        RuntimeError: If a provider requires an api_key and it is missing.
        ValueError:   If provider is unknown, or "custom" is used without base_url.
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
        response = httpx.get(f"{base}/api/tags", timeout=30.0)
        response.raise_for_status()
        data = response.json()
        return sorted(m["name"] for m in data.get("models", []))

    if provider == "anthropic":
        if not api_key:
            raise RuntimeError(
                "An Anthropic API key is required to list models. "
                "Provide api_key or set ANTHROPIC_API_KEY."
            )
        import httpx  # lazy import

        response = httpx.get(
            "https://api.anthropic.com/v1/models",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()
        return sorted(m["id"] for m in data.get("data", []))

    if provider == "gemini":
        if not api_key:
            raise RuntimeError(
                "A Gemini API key is required to list models. "
                "Provide api_key or set GEMINI_API_KEY."
            )
        import httpx  # lazy import

        try:
            response = httpx.get(
                f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
                timeout=30.0,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Gemini returned HTTP {e.response.status_code}. Check your API key."
            ) from None
        except httpx.RequestError:
            raise RuntimeError("Could not reach the Gemini API.") from None
        data = response.json()
        models = [
            m["name"].removeprefix("models/")
            for m in data.get("models", [])
            if "generateContent" in m.get("supportedGenerationMethods", [])
        ]
        return sorted(models)

    if provider == "openrouter":
        import httpx  # lazy import

        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = httpx.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        return sorted(m["id"] for m in data.get("data", []))

    if provider == "custom":
        if not base_url:
            raise ValueError(
                "A base_url is required for the 'custom' provider. "
                "Provide the base URL of your OpenAI-compatible endpoint."
            )
        import httpx  # lazy import

        url = base_url.rstrip("/") + "/models"
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        response = httpx.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        return sorted(m["id"] for m in data.get("data", []))

    raise ValueError(
        f"Unknown provider '{provider}'. "
        "Supported providers: openai, ollama, anthropic, gemini, openrouter, custom."
    )
