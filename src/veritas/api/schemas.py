"""Pydantic request/response schemas for the Veritas API (Task 14)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    """Request body for POST /ask."""

    question: str


# ---------------------------------------------------------------------------
# LLM config request/response schemas (Part C)
# ---------------------------------------------------------------------------


class LLMConnectRequest(BaseModel):
    """Request body for POST /config/llm/connect."""

    provider: str
    api_key: str | None = None
    base_url: str | None = None


class LLMConnectResponse(BaseModel):
    """Response for POST /config/llm/connect."""

    provider: str
    available_models: list[str]


class LLMSelectRequest(BaseModel):
    """Request body for POST /config/llm/select."""

    model: str


class LLMSelectResponse(BaseModel):
    """Response for POST /config/llm/select."""

    provider: str
    model: str


class LLMConfigResponse(BaseModel):
    """Response for GET /config/llm."""

    configured: bool
    provider: str | None
    model: str | None
    available_models: list[str]
    has_key: bool


# ---------------------------------------------------------------------------
# Response models — Documents & Jobs
# ---------------------------------------------------------------------------


class DocumentResponse(BaseModel):
    """A single ingested document."""

    id: str
    filename: str
    content_hash: str
    num_pages: int
    status: str
    created_at: str
    fact_count: int = 0


class UploadResponse(BaseModel):
    """Response for POST /documents."""

    job_id: str
    filename: str


class JobResponse(BaseModel):
    """A background ingestion job."""

    id: str
    doc_id: str
    status: str
    progress: float
    message: str
    created_at: str


# ---------------------------------------------------------------------------
# Response models — Facts
# ---------------------------------------------------------------------------


class FactResponse(BaseModel):
    """A single extracted fact."""

    id: str
    doc_id: str
    subject: str
    attribute: str
    value: str
    unit: str
    temporal_context: str
    scope_qualifiers: list[str]
    evidence_span: str
    page: int
    claim_text: str
    fact_kind: str
    confidence: float
    normalized_value: float | None = None
    normalized_unit: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    canonical_subject: str | None = None


# ---------------------------------------------------------------------------
# Response models — Relationships / Edges
# ---------------------------------------------------------------------------


class RelationshipResponse(BaseModel):
    """An edge between two facts, enriched with fact details."""

    id: str
    fact_a_id: str
    fact_b_id: str
    relation: str
    reasoning: str
    reconciling_dimension: str
    confidence: float
    # Enriched fact details (None if the fact was not found in the store)
    fact_a: FactResponse | None = None
    fact_b: FactResponse | None = None


# ---------------------------------------------------------------------------
# Response models — Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness + config response."""

    status: str
    provider: str
    counts: dict[str, int]


# ---------------------------------------------------------------------------
# Response models — Q&A
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A single grounded citation."""

    fact_id: str
    doc_id: str
    page: int
    evidence_span: str


class AskResponse(BaseModel):
    """Response for POST /ask."""

    answer: str
    citations: list[Citation]


# ---------------------------------------------------------------------------
# Response models — Cases
# ---------------------------------------------------------------------------


class CasesResponse(BaseModel):
    """Response for GET /cases — one example per relation type plus failure."""

    corroborate: dict[str, Any] | None = None
    contradict: dict[str, Any] | None = None
    reconcilable: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Response models — Export
# ---------------------------------------------------------------------------


class ExportResponse(BaseModel):
    """JSON export of facts + edges."""

    facts: list[dict[str, Any]]
    edges: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Response models — Provider metadata
# ---------------------------------------------------------------------------


class LLMProviderInfo(BaseModel):
    """Metadata about a single LLM provider."""

    id: str
    label: str
    needs_key: bool
    needs_base_url: bool
    default_base_url: str | None = None
    # When True, the key field is shown but not required (for keyless local servers).
    key_optional: bool = False
