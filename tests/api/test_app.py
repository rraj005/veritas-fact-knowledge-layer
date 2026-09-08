"""Tests for the FastAPI application (Task 14)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fixtures.make_pdf import make_pdf
from veritas.llm.client import FakeLLMClient
from veritas.models import Fact, new_id
from veritas.store.db import Store
from veritas.store.embeddings import FakeEmbedder
from veritas.store.vectors import VectorIndex

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _extraction_response(text: str, value: str = "100") -> str:
    """Return a JSON extraction response for a single fact.

    The evidence_span is set to the full document text so it is guaranteed
    to be a substring of the chunk text fed to the extractor.
    """
    return json.dumps(
        [
            {
                "subject": "Company",
                "attribute": "revenue",
                "value": value,
                "unit": "INR Cr",
                "temporal_context": "FY24",
                "scope_qualifiers": [],
                "evidence_span": text.strip(),
                "claim_text": f"Revenue FY24 {value} INR Cr",
                "fact_kind": "numerical",
                "confidence": 0.9,
            }
        ]
    )


def _adjudication_response() -> str:
    return json.dumps(
        {
            "relation": "corroborate",
            "reasoning": "Same values",
            "reconciling_dimension": "none",
            "confidence": 0.85,
        }
    )


def _make_fake_llm(doc_text: str) -> FakeLLMClient:
    """Return a FakeLLMClient that handles extraction, adjudication, and Q&A."""

    def _respond(system: str, user: str) -> str:
        if "atomic factual claim" in system:
            # Extraction prompt — return a valid fact whose evidence_span is
            # a substring of the chunk text that contains doc_text.
            if doc_text in user:
                return _extraction_response(doc_text)
            return "[]"
        if "grounded question-answering" in system:
            # Q&A prompt — return a placeholder that resolves fact ids from user.
            # We extract any fact ids present in the user prompt and cite one.
            import re

            ids = re.findall(r"\[(fact_[a-z0-9]+)\]", user)
            cited = ids[:1] if ids else []
            return json.dumps({"answer": "Revenue was 100 Cr.", "cited_fact_ids": cited})
        return _adjudication_response()

    return FakeLLMClient(_respond)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_store(tmp_path: Path) -> Store:
    s = Store(tmp_path / "veritas.db")
    s.init_schema()
    return s


@pytest.fixture()
def tmp_index(tmp_path: Path) -> VectorIndex:
    return VectorIndex(tmp_path / "chroma", FakeEmbedder())


@pytest.fixture()
def tiny_pdf_bytes(tmp_path: Path) -> bytes:
    """Create a tiny in-memory PDF and return its bytes."""
    p = tmp_path / "tiny.pdf"
    make_pdf(p, ["Revenue was 100 crore in FY24."])
    return p.read_bytes()


@pytest.fixture()
def doc_text() -> str:
    return "Revenue was 100 crore in FY24."


@pytest.fixture()
def client_with_fakes(tmp_path: Path, tmp_store: Store, tmp_index: VectorIndex, doc_text: str):
    """TestClient with injected fakes (no real LLM, no real embedder)."""
    from fastapi.testclient import TestClient

    from veritas.api.app import build_app
    from veritas.config import get_settings
    from veritas.pipeline import Pipeline

    llm = _make_fake_llm(doc_text)
    settings = get_settings()

    # Use a tmp upload dir so tests don't pollute the real data dir.
    import dataclasses

    settings = dataclasses.replace(settings, upload_dir=tmp_path / "uploads")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(store=tmp_store, index=tmp_index, llm=llm, settings=settings)
    app = build_app(
        store=tmp_store,
        index=tmp_index,
        pipeline=pipeline,
        llm=llm,
    )
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health(client_with_fakes) -> None:
    """GET /health returns status ok."""
    r = client_with_fakes.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "counts" in body


def test_upload_and_list_facts(client_with_fakes, tiny_pdf_bytes: bytes, doc_text: str) -> None:
    """POST /documents uploads, creates a job; GET /facts returns extracted facts."""
    r = client_with_fakes.post(
        "/documents",
        files={"file": ("a.pdf", tiny_pdf_bytes, "application/pdf")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert body.get("filename") == "a.pdf"

    # Under TestClient, background tasks run inline before the response is returned.
    facts = client_with_fakes.get("/facts").json()
    assert isinstance(facts, list)
    # The FakeLLM extracts exactly one fact whose evidence_span is the doc text.
    assert len(facts) >= 1, "Expected at least one extracted fact after upload"
    # Grounding check: evidence_span must be a substring of the PDF text
    first_fact = facts[0]
    assert first_fact["evidence_span"] in doc_text or doc_text in first_fact["evidence_span"]


def test_facts_404(client_with_fakes) -> None:
    """GET /facts/{id} returns 404 for an unknown id."""
    r = client_with_fakes.get("/facts/nonexistent_id")
    assert r.status_code == 404


def test_list_documents(client_with_fakes, tiny_pdf_bytes: bytes) -> None:
    """GET /documents returns a list."""
    client_with_fakes.post(
        "/documents",
        files={"file": ("b.pdf", tiny_pdf_bytes, "application/pdf")},
    )
    r = client_with_fakes.get("/documents")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_job_status(client_with_fakes, tiny_pdf_bytes: bytes) -> None:
    """GET /jobs/{id} returns job details."""
    r = client_with_fakes.post(
        "/documents",
        files={"file": ("c.pdf", tiny_pdf_bytes, "application/pdf")},
    )
    job_id = r.json()["job_id"]
    r2 = client_with_fakes.get(f"/jobs/{job_id}")
    assert r2.status_code == 200
    body = r2.json()
    assert "status" in body
    assert "progress" in body


def test_relationships_list(client_with_fakes) -> None:
    """GET /relationships returns a list."""
    r = client_with_fakes.get("/relationships")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_relationships_enriched(client_with_fakes, tmp_store: Store) -> None:
    """GET /relationships returns edges enriched with full fact details."""
    # Seed two facts and an edge.
    fa = Fact(
        new_id("fact"),
        "doc1",
        "Co",
        "revenue",
        "100",
        "INR Cr",
        "FY24",
        [],
        "Revenue 100 Cr",
        1,
        "Revenue FY24 100",
        "numerical",
        0.9,
    )
    fb = Fact(
        new_id("fact"),
        "doc2",
        "Co",
        "revenue",
        "100",
        "INR Cr",
        "FY24",
        [],
        "Revenue 100 Cr",
        2,
        "Revenue FY24 100",
        "numerical",
        0.9,
    )
    from veritas.models import Edge

    edge = Edge(
        new_id("edge"),
        fa.id,
        fb.id,
        "corroborate",
        "Same value",
        "none",
        0.85,
    )
    tmp_store.add_facts([fa, fb])
    tmp_store.add_edges([edge])

    r = client_with_fakes.get("/relationships")
    assert r.status_code == 200
    edges = r.json()
    assert len(edges) >= 1
    # Find the seeded edge in the response
    e = next((x for x in edges if x["id"] == edge.id), None)
    assert e is not None
    # fact_a must be fully populated (not null)
    assert e["fact_a"] is not None, "fact_a should be resolved in the enriched response"
    assert e["fact_a"]["evidence_span"] != "", "fact_a.evidence_span must be non-empty"
    assert e["fact_b"] is not None, "fact_b should be resolved in the enriched response"


def test_ask_endpoint(client_with_fakes, tmp_store: Store, tmp_index: VectorIndex) -> None:
    """POST /ask returns a grounded answer citing the seeded fact."""
    # Seed a fact and an embedding so Q&A has something to retrieve.
    fa = Fact(
        new_id("fact"),
        "doc1",
        "Co",
        "revenue",
        "100",
        "INR Cr",
        "FY24",
        [],
        "Revenue 100 Cr",
        1,
        "Revenue FY24 100",
        "numerical",
        0.9,
    )
    tmp_store.add_facts([fa])
    tmp_index.add([fa.id], [fa.claim_text], [{"doc_id": fa.doc_id, "fact_id": fa.id}])

    r = client_with_fakes.post("/ask", json={"question": "What is the revenue?"})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "citations" in body
    # The FakeLLM returns a specific answer string when answering QA prompts
    assert body["answer"] == "Revenue was 100 Cr."
    # The citation must include the seeded fact id and its evidence_span
    assert len(body["citations"]) >= 1
    cite = body["citations"][0]
    assert cite["fact_id"] == fa.id
    assert cite["evidence_span"] == fa.evidence_span
    assert cite["page"] == fa.page


def test_cases_endpoint(client_with_fakes) -> None:
    """GET /cases returns the four-cases structure."""
    r = client_with_fakes.get("/cases")
    assert r.status_code == 200
    body = r.json()
    assert "corroborate" in body
    assert "contradict" in body
    assert "reconcilable" in body
    assert "failure" in body


def test_export_json(client_with_fakes) -> None:
    """GET /export?format=json returns JSON with facts and edges."""
    r = client_with_fakes.get("/export?format=json")
    assert r.status_code == 200
    body = r.json()
    assert "facts" in body
    assert "edges" in body


def test_export_csv(client_with_fakes, tiny_pdf_bytes: bytes) -> None:
    """GET /export?format=csv returns a non-empty CSV with expected header columns."""
    # Upload a document first so the CSV has actual data rows.
    client_with_fakes.post(
        "/documents",
        files={"file": ("exp.pdf", tiny_pdf_bytes, "application/pdf")},
    )

    r = client_with_fakes.get("/export?format=csv")
    assert r.status_code == 200
    content = r.text
    assert len(content) > 0, "CSV output must not be empty"
    lines = content.strip().splitlines()
    header = lines[0].lower()
    assert "id" in header, "CSV header must contain 'id'"
    assert "claim_text" in header, "CSV header must contain 'claim_text'"
    # Should have at least header + one data row
    assert len(lines) >= 2, "CSV must have at least a header row and one data row"


# ---------------------------------------------------------------------------
# LLM runtime config endpoint tests (Part C)
# ---------------------------------------------------------------------------


def _make_injected_client(tmp_path, tmp_store, tmp_index, doc_text):
    """Build a TestClient with a pre-injected fake LLM (no config/llm calls needed)."""
    import dataclasses

    from fastapi.testclient import TestClient

    from veritas.api.app import build_app
    from veritas.config import get_settings
    from veritas.pipeline import Pipeline

    llm = _make_fake_llm(doc_text)
    settings = get_settings()
    settings = dataclasses.replace(settings, upload_dir=tmp_path / "uploads")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(store=tmp_store, index=tmp_index, llm=llm, settings=settings)
    app = build_app(store=tmp_store, index=tmp_index, pipeline=pipeline, llm=llm)
    return TestClient(app)


def test_get_llm_config_injected(tmp_path, tmp_store, tmp_index, doc_text) -> None:
    """GET /config/llm returns configured=True when a client was injected."""
    client = _make_injected_client(tmp_path, tmp_store, tmp_index, doc_text)
    r = client.get("/config/llm")
    assert r.status_code == 200
    body = r.json()
    assert body["configured"] is True
    assert "api_key" not in body, "api_key must never appear in the response"


def test_connect_llm_success(tmp_path, tmp_store, tmp_index, doc_text) -> None:
    """POST /config/llm/connect with a fake list_models_fn succeeds and stores config."""
    import dataclasses

    from fastapi.testclient import TestClient

    from veritas.api.app import build_app
    from veritas.config import get_settings
    from veritas.pipeline import Pipeline

    llm = _make_fake_llm(doc_text)
    settings = get_settings()
    settings = dataclasses.replace(settings, upload_dir=tmp_path / "uploads")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(store=tmp_store, index=tmp_index, llm=llm, settings=settings)

    def fake_list_models(provider, api_key=None, base_url=None):
        return ["model-a", "model-b"]

    app = build_app(
        store=tmp_store,
        index=tmp_index,
        pipeline=pipeline,
        llm=None,
        list_models_fn=fake_list_models,
    )
    client = TestClient(app)

    r = client.post(
        "/config/llm/connect",
        json={"provider": "openai", "api_key": "sk-test"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["provider"] == "openai"
    assert "model-a" in body["available_models"]
    assert "api_key" not in body, "api_key must never appear in the response"

    # GET /config/llm should now show not yet configured (model not selected)
    r2 = client.get("/config/llm")
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["configured"] is False  # model not selected yet
    assert body2["provider"] == "openai"
    assert body2["has_key"] is True
    assert "api_key" not in body2


def test_select_model_success(tmp_path, tmp_store, tmp_index, doc_text) -> None:
    """POST /config/llm/select succeeds after connect and marks config ready."""
    import dataclasses

    from fastapi.testclient import TestClient

    from veritas.api.app import build_app
    from veritas.config import get_settings
    from veritas.pipeline import Pipeline

    llm = _make_fake_llm(doc_text)
    settings = get_settings()
    settings = dataclasses.replace(settings, upload_dir=tmp_path / "uploads")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(store=tmp_store, index=tmp_index, llm=llm, settings=settings)

    def fake_list_models(provider, api_key=None, base_url=None):
        return ["gpt-4o", "gpt-3.5-turbo"]

    app = build_app(
        store=tmp_store,
        index=tmp_index,
        pipeline=pipeline,
        llm=None,
        list_models_fn=fake_list_models,
    )
    client = TestClient(app)

    # Connect first
    client.post("/config/llm/connect", json={"provider": "openai", "api_key": "sk-test"})

    # Select a model
    r = client.post("/config/llm/select", json={"model": "gpt-4o"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"] == "gpt-4o"
    assert body["provider"] == "openai"

    # GET /config/llm should now show configured=True
    r2 = client.get("/config/llm")
    assert r2.status_code == 200
    body2 = r2.json()
    assert body2["configured"] is True
    assert body2["model"] == "gpt-4o"


def test_select_model_invalid_returns_400(tmp_path, tmp_store, tmp_index, doc_text) -> None:
    """POST /config/llm/select with unknown model returns 400."""
    import dataclasses

    from fastapi.testclient import TestClient

    from veritas.api.app import build_app
    from veritas.config import get_settings
    from veritas.pipeline import Pipeline

    llm = _make_fake_llm(doc_text)
    settings = get_settings()
    settings = dataclasses.replace(settings, upload_dir=tmp_path / "uploads")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(store=tmp_store, index=tmp_index, llm=llm, settings=settings)

    def fake_list_models(provider, api_key=None, base_url=None):
        return ["gpt-4o"]

    app = build_app(
        store=tmp_store,
        index=tmp_index,
        pipeline=pipeline,
        llm=None,
        list_models_fn=fake_list_models,
    )
    client = TestClient(app)

    client.post("/config/llm/connect", json={"provider": "openai", "api_key": "sk-test"})
    r = client.post("/config/llm/select", json={"model": "nonexistent-model"})
    assert r.status_code == 400


def test_connect_llm_failure_returns_400(tmp_path, tmp_store, tmp_index, doc_text) -> None:
    """POST /config/llm/connect returns 400 when list_models raises."""
    import dataclasses

    from fastapi.testclient import TestClient

    from veritas.api.app import build_app
    from veritas.config import get_settings
    from veritas.pipeline import Pipeline

    llm = _make_fake_llm(doc_text)
    settings = get_settings()
    settings = dataclasses.replace(settings, upload_dir=tmp_path / "uploads")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(store=tmp_store, index=tmp_index, llm=llm, settings=settings)

    def failing_list_models(provider, api_key=None, base_url=None):
        raise RuntimeError("Invalid API key")

    app = build_app(
        store=tmp_store,
        index=tmp_index,
        pipeline=pipeline,
        llm=None,
        list_models_fn=failing_list_models,
    )
    client = TestClient(app)

    r = client.post("/config/llm/connect", json={"provider": "openai", "api_key": "bad-key"})
    assert r.status_code == 400
    # Key must not appear in error response
    assert "bad-key" not in r.text


def test_connect_llm_sdk_error_does_not_leak_key(tmp_path, tmp_store, tmp_index, doc_text) -> None:
    """POST /config/llm/connect with an SDK-level Exception must not leak key material."""
    import dataclasses

    from fastapi.testclient import TestClient

    from veritas.api.app import build_app
    from veritas.config import get_settings
    from veritas.pipeline import Pipeline

    llm = _make_fake_llm(doc_text)
    settings = get_settings()
    settings = dataclasses.replace(settings, upload_dir=tmp_path / "uploads")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(store=tmp_store, index=tmp_index, llm=llm, settings=settings)

    def sdk_error_list_models(provider, api_key=None, base_url=None):
        # Simulate an SDK auth error that embeds key material in its message.
        raise Exception("Incorrect API key sk-secret-123")  # noqa: TRY002

    app = build_app(
        store=tmp_store,
        index=tmp_index,
        pipeline=pipeline,
        llm=None,
        list_models_fn=sdk_error_list_models,
    )
    client = TestClient(app)

    r = client.post("/config/llm/connect", json={"provider": "openai", "api_key": "sk-secret-123"})
    assert r.status_code == 400
    # The raw SDK message must NOT appear in the response body.
    assert "sk-secret-123" not in r.text
    # The safe generic message must be present instead.
    assert "Failed to connect to the provider" in r.text


def test_get_llm_providers_endpoint(client_with_fakes) -> None:
    """GET /config/llm/providers returns all expected provider ids."""
    r = client_with_fakes.get("/config/llm/providers")
    assert r.status_code == 200
    providers = r.json()
    assert isinstance(providers, list)
    ids = [p["id"] for p in providers]
    for expected in ("openai", "anthropic", "gemini", "openrouter", "ollama", "custom"):
        assert expected in ids, f"Provider '{expected}' missing from /config/llm/providers"
    # Each entry must have the required fields
    for p in providers:
        assert "id" in p
        assert "label" in p
        assert "needs_key" in p
        assert "needs_base_url" in p
        assert "default_base_url" in p or p.get("default_base_url") is None


def test_get_llm_providers_ollama_has_default_base_url(client_with_fakes) -> None:
    """GET /config/llm/providers — Ollama entry must have a default_base_url."""
    r = client_with_fakes.get("/config/llm/providers")
    providers = {p["id"]: p for p in r.json()}
    ollama = providers["ollama"]
    assert ollama["needs_base_url"] is True
    assert ollama["needs_key"] is False
    assert ollama["default_base_url"] is not None
    assert "localhost" in ollama["default_base_url"]


def test_get_llm_providers_openai_needs_key(client_with_fakes) -> None:
    """GET /config/llm/providers — OpenAI entry must need a key and no base_url."""
    r = client_with_fakes.get("/config/llm/providers")
    providers = {p["id"]: p for p in r.json()}
    openai_p = providers["openai"]
    assert openai_p["needs_key"] is True
    assert openai_p["needs_base_url"] is False


def test_select_model_empty_available_list_returns_400(tmp_path, tmp_store, tmp_index, doc_text) -> None:
    """POST /config/llm/select returns 400 when provider returned no models."""
    import dataclasses

    from fastapi.testclient import TestClient

    from veritas.api.app import build_app
    from veritas.config import get_settings
    from veritas.pipeline import Pipeline

    llm = _make_fake_llm(doc_text)
    settings = get_settings()
    settings = dataclasses.replace(settings, upload_dir=tmp_path / "uploads")
    settings.upload_dir.mkdir(parents=True, exist_ok=True)

    pipeline = Pipeline(store=tmp_store, index=tmp_index, llm=llm, settings=settings)

    def empty_list_models(provider, api_key=None, base_url=None):
        return []

    app = build_app(
        store=tmp_store,
        index=tmp_index,
        pipeline=pipeline,
        llm=None,
        list_models_fn=empty_list_models,
    )
    client = TestClient(app)

    # Connect succeeds (empty model list is valid at connect time).
    r = client.post("/config/llm/connect", json={"provider": "openai", "api_key": "sk-test"})
    assert r.status_code == 200

    # Selecting any model must fail when the provider returned no models.
    r2 = client.post("/config/llm/select", json={"model": "any-model"})
    assert r2.status_code == 400
