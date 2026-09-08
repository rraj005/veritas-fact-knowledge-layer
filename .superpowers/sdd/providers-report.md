# LLM Providers Report

## What Changed

### New Providers Added
- **anthropic**: Plain HTTP via httpx to `https://api.anthropic.com/v1/messages`. Discovery GETs `/v1/models` with `x-api-key` + `anthropic-version` headers.
- **gemini**: Plain HTTP via httpx to Google Generative Language API. Discovery filters `generateContent` methods and strips `models/` prefix from ids.
- **openrouter**: OpenAI-compatible via `openai` SDK with fixed `base_url=https://openrouter.ai/api/v1`. Discovery GETs `/api/v1/models`.
- **custom**: OpenAI-compatible via `openai` SDK with user-supplied `base_url`. Requires `base_url`. Covers Groq/Together/DeepSeek/Mistral/vLLM/etc.

### Modified Files
- `src/veritas/llm/models.py` — Added branches for anthropic, gemini, openrouter, custom. Updated docstring.
- `src/veritas/llm/client.py` — Added `AnthropicClient`, `GeminiClient`, `OpenAICompatibleClient` (accepts `base_url`). Added `OpenAIClient` as backward-compatible alias. Added central `client_for()` factory. `get_client()` unchanged for env-based path.
- `src/veritas/llm/providers.py` — New file. Single source of truth: list of `ProviderInfo` dataclasses with id, label, needs_key, needs_base_url, default_base_url.
- `src/veritas/api/schemas.py` — Added `LLMProviderInfo` Pydantic model.
- `src/veritas/api/app.py` — Imported `LLMProviderInfo`. Updated `_current_llm()` to use `client_for()` (all 6 providers work at runtime). Added `GET /config/llm/providers` endpoint.
- `src/veritas/web/index.html` — Replaced hardcoded `openai`/`ollama` option elements and static field divs with dynamic placeholders (`llm-key-fields`, `llm-url-fields`).
- `src/veritas/web/app.js` — `initLlmSetup()` now fetches `/config/llm/providers` on load and builds the dropdown + conditionally shows/hides key/base-url fields based on `needs_key`/`needs_base_url`. `refreshLlmStatus()` updated to handle ollama (base_url provider without key).

### New Endpoint
- `GET /config/llm/providers` — Returns `[{id, label, needs_key, needs_base_url, default_base_url}]` for all 6 providers. Used by the frontend to render fields dynamically.

### New Tests
- `tests/llm/test_models.py` — Added 12 new tests: anthropic (sorted + missing key), gemini (filter + strip prefix + missing key), openrouter (sorted + no key works), custom (base_url used + missing base_url raises).
- `tests/llm/test_new_clients.py` — New file: AnthropicClient request shape + first-content-block extraction, GeminiClient request shape + first-candidate extraction, OpenAICompatibleClient base_url forwarding (with and without).
- `tests/api/test_app.py` — Added 3 tests: `GET /config/llm/providers` returns all 6 ids, ollama entry has default_base_url, openai entry needs_key=true.

## Test Results
- **Before**: 165 passed
- **After**: 182 passed (+17 new), 1 warning (unrelated starlette deprecation)
- `ruff check src tests`: All checks passed

## Commit Hashes
To be filled after git commit.

## Concerns
- None. The `get_client()` env-based factory path still only supports `openai` and `ollama` via `_PROVIDER_CLASSES`/`_NO_KEY_PROVIDERS` (intentionally unchanged to not break env-based config). Runtime path via `_current_llm()` → `client_for()` supports all 6 providers.
- OpenRouter and custom providers pass `api_key=""` to the OpenAI SDK for custom when no key is given; this is safe since custom endpoints may not require auth.
