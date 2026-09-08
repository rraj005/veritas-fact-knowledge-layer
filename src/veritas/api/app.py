"""FastAPI application factory and routes for the Veritas API (Task 14)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from veritas.api.schemas import (
    AskRequest,
    AskResponse,
    CasesResponse,
    Citation,
    DocumentResponse,
    ExportResponse,
    FactResponse,
    HealthResponse,
    JobResponse,
    LLMConfigResponse,
    LLMConnectRequest,
    LLMConnectResponse,
    LLMProviderInfo,
    LLMSelectRequest,
    LLMSelectResponse,
    RelationshipResponse,
    UploadResponse,
)
from veritas.cases import build_cases, export_csv, export_layer
from veritas.llm.models import list_models as _default_list_models
from veritas.models import Job, new_id

if TYPE_CHECKING:
    from veritas.llm.client import LLMClient
    from veritas.pipeline import Pipeline
    from veritas.store.db import Store
    from veritas.store.vectors import VectorIndex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Web static-files directory
# ---------------------------------------------------------------------------

_WEB_DIR = Path(__file__).parent.parent / "web"


# ---------------------------------------------------------------------------
# Runtime LLM config — held in memory, NEVER persisted or logged
# ---------------------------------------------------------------------------


@dataclass
class RuntimeLLMConfig:
    """Mutable holder for the active LLM provider/key/model at runtime.

    The api_key is NEVER logged or returned in any response.
    """

    provider: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    available_models: list[str] = field(default_factory=list)
    # Optional pre-built client (used when build_app injects an llm for tests)
    _injected_client: object = field(default=None, repr=False)

    def is_ready(self) -> bool:
        """Return True when provider and model are both set (or a client was injected)."""
        if self._injected_client is not None:
            return True
        return bool(self.provider and self.model)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def build_app(
    store: Store,
    index: VectorIndex,
    pipeline: Pipeline,
    llm: LLMClient,
    list_models_fn: Callable[..., list[str]] = _default_list_models,
) -> FastAPI:
    """Create and return a configured FastAPI application.

    This factory allows tests to inject fakes without touching any real
    settings or API keys.

    Args:
        store:          The SQLite Store instance to use.
        index:          The VectorIndex instance for embedding lookups.
        pipeline:       The Pipeline instance for document ingestion.
        llm:            The LLMClient used by Q&A (injected by tests; production
                        uses the runtime config instead).
        list_models_fn: Injectable model-discovery function (default: real impl).

    Returns:
        A fully configured FastAPI application.
    """
    app = FastAPI(
        title="Veritas Fact Knowledge Layer",
        description=(
            "A domain-agnostic RAG system that ingests PDFs, extracts grounded facts, "
            "and detects cross-document relationships."
        ),
        version="1.0.0",
    )

    # ------------------------------------------------------------------
    # CORS — permissive for local development.
    # allow_credentials must be False when allow_origins=["*"] (browser spec).
    # ------------------------------------------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Runtime LLM config stored on app.state
    # If build_app was given a real injected llm (tests), pre-populate config
    # so existing tests never need to call /config/llm/connect.
    # ------------------------------------------------------------------
    runtime_cfg = RuntimeLLMConfig()
    if llm is not None:
        runtime_cfg._injected_client = llm
        runtime_cfg.provider = "test"
        runtime_cfg.model = "test"
    app.state.llm_config = runtime_cfg

    # ------------------------------------------------------------------
    # Helper: resolve the active LLM client from runtime config
    # ------------------------------------------------------------------
    def _current_llm() -> LLMClient:
        cfg: RuntimeLLMConfig = app.state.llm_config
        if not cfg.is_ready():
            raise HTTPException(
                status_code=400,
                detail=(
                    "LLM not configured — connect a provider and select a model "
                    "via /config/llm/connect and /config/llm/select."
                ),
            )
        # If a client was pre-injected (test path), return it directly.
        if cfg._injected_client is not None:
            return cfg._injected_client  # type: ignore[return-value]
        # Build client from runtime config at request time.
        from veritas.llm.cache import DiskCache
        from veritas.llm.client import CachedClient, client_for

        provider = cfg.provider or ""
        model = cfg.model or ""
        try:
            inner: LLMClient = client_for(
                provider,
                model,
                api_key=cfg.api_key,
                base_url=cfg.base_url,
            )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Wrap with cache if pipeline.settings has a cache_dir
        try:
            cache_dir = pipeline.settings.cache_dir
            return CachedClient(inner, DiskCache(cache_dir), model=model)
        except AttributeError:
            return inner

    # ------------------------------------------------------------------
    # Helper: upload dir from pipeline settings
    # ------------------------------------------------------------------
    def _upload_dir() -> Path:
        p: Path = pipeline.settings.upload_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------------
    # Helper: convert Fact -> FactResponse
    # ------------------------------------------------------------------
    def _fact_to_response(fact) -> FactResponse:
        return FactResponse(
            id=fact.id,
            doc_id=fact.doc_id,
            subject=fact.subject,
            attribute=fact.attribute,
            value=fact.value,
            unit=fact.unit,
            temporal_context=fact.temporal_context,
            scope_qualifiers=fact.scope_qualifiers,
            evidence_span=fact.evidence_span,
            page=fact.page,
            claim_text=fact.claim_text,
            fact_kind=fact.fact_kind,
            confidence=fact.confidence,
            normalized_value=fact.normalized_value,
            normalized_unit=fact.normalized_unit,
            period_start=fact.period_start,
            period_end=fact.period_end,
            canonical_subject=fact.canonical_subject,
        )

    # ------------------------------------------------------------------
    # Helper: convert Edge -> RelationshipResponse (enriched)
    # ------------------------------------------------------------------
    def _edge_to_response(edge) -> RelationshipResponse:
        fact_a = store.get_fact(edge.fact_a_id)
        fact_b = store.get_fact(edge.fact_b_id)
        return RelationshipResponse(
            id=edge.id,
            fact_a_id=edge.fact_a_id,
            fact_b_id=edge.fact_b_id,
            relation=edge.relation,
            reasoning=edge.reasoning,
            reconciling_dimension=edge.reconciling_dimension,
            confidence=edge.confidence,
            fact_a=_fact_to_response(fact_a) if fact_a else None,
            fact_b=_fact_to_response(fact_b) if fact_b else None,
        )

    # ==================================================================
    # Routes
    # ==================================================================

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        """Return liveness + provider + aggregate counts."""
        cfg: RuntimeLLMConfig = app.state.llm_config
        provider = cfg.provider or ""
        return HealthResponse(
            status="ok",
            provider=provider,
            counts=store.counts(),
        )

    # ------------------------------------------------------------------
    # LLM Runtime Config
    # ------------------------------------------------------------------

    @app.get("/config/llm", response_model=LLMConfigResponse, tags=["config"])
    def get_llm_config() -> LLMConfigResponse:
        """Return current runtime LLM configuration (never returns the api_key)."""
        cfg: RuntimeLLMConfig = app.state.llm_config
        return LLMConfigResponse(
            configured=cfg.is_ready(),
            provider=cfg.provider,
            model=cfg.model,
            available_models=cfg.available_models,
            has_key=bool(cfg.api_key),
        )

    @app.post("/config/llm/connect", response_model=LLMConnectResponse, tags=["config"])
    def connect_llm(body: LLMConnectRequest) -> LLMConnectResponse:
        """Connect to an LLM provider and discover available models.

        Stores provider/key/base_url in runtime config (never logs the key).
        Clears any previously-selected model.
        Returns the list of available models for the user to choose from.
        """
        try:
            models = list_models_fn(
                body.provider,
                api_key=body.api_key,
                base_url=body.base_url,
            )
        except (ValueError, RuntimeError) as exc:
            # Our own exceptions carry safe, user-facing messages.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            # SDK / network errors may embed API-key fragments — never echo them.
            raise HTTPException(
                status_code=400,
                detail=(
                    "Failed to connect to the provider. Check the provider name, "
                    "API key, and base URL, then try again."
                ),
            ) from exc

        cfg: RuntimeLLMConfig = app.state.llm_config
        cfg.provider = body.provider
        cfg.api_key = body.api_key
        cfg.base_url = body.base_url
        cfg.model = None  # cleared — user must select via /config/llm/select
        cfg.available_models = models
        cfg._injected_client = None  # clear any pre-injected test client

        return LLMConnectResponse(provider=body.provider, available_models=models)

    @app.post("/config/llm/select", response_model=LLMSelectResponse, tags=["config"])
    def select_model(body: LLMSelectRequest) -> LLMSelectResponse:
        """Select the active model from the last-fetched available_models list.

        The model must be one returned by the most recent /config/llm/connect call.
        """
        cfg: RuntimeLLMConfig = app.state.llm_config
        if not cfg.provider:
            raise HTTPException(
                status_code=400,
                detail="No provider connected. Call /config/llm/connect first.",
            )
        if body.model not in cfg.available_models:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{body.model}' is not in the available models list. "
                    "Call /config/llm/connect to refresh the list."
                ),
            )
        cfg.model = body.model
        return LLMSelectResponse(provider=cfg.provider or "", model=body.model)

    @app.get(
        "/config/llm/providers",
        response_model=list[LLMProviderInfo],
        tags=["config"],
    )
    def list_providers() -> list[LLMProviderInfo]:
        """Return metadata for all supported LLM providers.

        The frontend uses this to dynamically build the provider dropdown and
        show/hide the API key and base URL fields based on provider requirements.
        """
        from veritas.llm.providers import PROVIDERS

        return [
            LLMProviderInfo(
                id=p.id,
                label=p.label,
                needs_key=p.needs_key,
                needs_base_url=p.needs_base_url,
                default_base_url=p.default_base_url,
            )
            for p in PROVIDERS
        ]

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    @app.post("/documents", response_model=UploadResponse, tags=["ingest"])
    async def upload_document(
        file: UploadFile,
        background_tasks: BackgroundTasks,
    ) -> UploadResponse:
        """Upload a PDF for background ingestion.

        Saves the uploaded file to the configured upload directory, creates a
        Job record with status="queued", and schedules ingestion via
        BackgroundTasks. Returns the job_id immediately.
        """
        # Resolve the LLM client at request time (raises 400 if not configured).
        active_llm = _current_llm()

        filename = file.filename or "upload.pdf"
        data = await file.read()

        # Persist the upload.
        upload_path = _upload_dir() / f"{new_id('up')}_{filename}"
        upload_path.write_bytes(data)

        # Create a Job row so callers can poll progress.
        job_id = new_id("job")
        now = datetime.now(tz=UTC).isoformat()
        job = Job(
            id=job_id,
            doc_id="",  # will be linked once ingestion starts
            status="queued",
            progress=0.0,
            message="Queued for ingestion",
            created_at=now,
        )
        store.create_job(job)

        # Build a per-request pipeline with the current LLM.
        from veritas.pipeline import Pipeline

        request_pipeline = Pipeline(
            store=store,
            index=index,
            llm=active_llm,
            settings=pipeline.settings,
        )

        # Schedule ingestion in the background.
        background_tasks.add_task(
            request_pipeline.ingest_document,
            upload_path,
            filename,
            job_id,
        )

        return UploadResponse(job_id=job_id, filename=filename)

    @app.get("/documents", response_model=list[DocumentResponse], tags=["ingest"])
    def list_documents() -> list[DocumentResponse]:
        """Return all ingested documents."""
        docs = store.list_documents()
        return [
            DocumentResponse(
                id=d.id,
                filename=d.filename,
                content_hash=d.content_hash,
                num_pages=d.num_pages,
                status=d.status,
                created_at=d.created_at,
            )
            for d in docs
        ]

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    @app.get("/jobs/{job_id}", response_model=JobResponse, tags=["ingest"])
    def get_job(job_id: str) -> JobResponse:
        """Return job status and progress."""
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
        return JobResponse(
            id=job.id,
            doc_id=job.doc_id,
            status=job.status,
            progress=job.progress,
            message=job.message,
            created_at=job.created_at,
        )

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------

    @app.get("/facts", response_model=list[FactResponse], tags=["facts"])
    def list_facts(
        doc_id: str | None = Query(default=None),
        subject: str | None = Query(default=None),
        kind: str | None = Query(default=None),
        min_conf: float = Query(default=0.0),
        limit: int = Query(default=200),
        offset: int = Query(default=0),
    ) -> list[FactResponse]:
        """List facts with optional filters."""
        facts = store.list_facts(
            doc_id=doc_id,
            subject=subject,
            kind=kind,
            min_conf=min_conf,
            limit=limit,
            offset=offset,
        )
        return [_fact_to_response(f) for f in facts]

    @app.get("/facts/{fact_id}", response_model=FactResponse, tags=["facts"])
    def get_fact(fact_id: str) -> FactResponse:
        """Return a single fact by id, including verbatim evidence and page."""
        fact = store.get_fact(fact_id)
        if fact is None:
            raise HTTPException(status_code=404, detail=f"Fact {fact_id!r} not found")
        return _fact_to_response(fact)

    # ------------------------------------------------------------------
    # Relationships
    # ------------------------------------------------------------------

    @app.get(
        "/relationships",
        response_model=list[RelationshipResponse],
        tags=["relationships"],
    )
    def list_relationships(
        relation: str | None = Query(default=None),
        doc_id: str | None = Query(default=None),
    ) -> list[RelationshipResponse]:
        """List relationship edges, each enriched with both facts' details."""
        edges = store.list_edges(relation=relation, doc_id=doc_id)
        return [_edge_to_response(e) for e in edges]

    @app.get(
        "/relationships/{edge_id}",
        response_model=RelationshipResponse,
        tags=["relationships"],
    )
    def get_relationship(edge_id: str) -> RelationshipResponse:
        """Return a single edge with both facts' full detail."""
        edge = store.get_edge(edge_id)
        if edge is None:
            raise HTTPException(status_code=404, detail=f"Relationship {edge_id!r} not found")
        return _edge_to_response(edge)

    # ------------------------------------------------------------------
    # Four Cases
    # ------------------------------------------------------------------

    @app.get("/cases", response_model=CasesResponse, tags=["cases"])
    def cases() -> CasesResponse:
        """Return up to one example of each relation type plus a failure slot."""
        return CasesResponse(**build_cases(store))

    # ------------------------------------------------------------------
    # Ask (grounded Q&A)
    # ------------------------------------------------------------------

    @app.post("/ask", response_model=AskResponse, tags=["qa"])
    def ask(body: AskRequest) -> AskResponse:
        """Answer a question grounded in the indexed facts with citations."""
        from veritas.qa import answer

        active_llm = _current_llm()
        result = answer(body.question, store, index, active_llm)
        citations = [
            Citation(
                fact_id=c["fact_id"],
                doc_id=c["doc_id"],
                page=c["page"],
                evidence_span=c["evidence_span"],
            )
            for c in result.get("citations", [])
        ]
        return AskResponse(answer=result["answer"], citations=citations)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @app.get("/export", response_model=None, tags=["export"])
    def export(
        fmt: str = Query(default="json", description="json or csv", alias="format"),
        kind: str = Query(default="facts", description="facts or edges (csv only)"),
    ):
        """Export the full knowledge layer.

        - ``?format=json`` (default) — returns ``{"facts": [...], "edges": [...]}``
        - ``?format=csv`` — returns facts as text/csv
        - ``?format=csv&kind=edges`` — returns edges as text/csv
        """
        if fmt == "csv":
            csv_str = export_csv(store, kind=kind)
            return Response(content=csv_str, media_type="text/csv")
        # Default: JSON — wire ExportResponse shape
        data = export_layer(store)
        return ExportResponse(facts=data["facts"], edges=data["edges"])

    # ------------------------------------------------------------------
    # Static files (web UI)
    # ------------------------------------------------------------------

    if _WEB_DIR.is_dir():
        from fastapi.staticfiles import StaticFiles

        app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")

    return app


# ---------------------------------------------------------------------------
# Module-level app — built lazily so import does NOT require an API key.
# ---------------------------------------------------------------------------

_app_singleton: FastAPI | None = None


def _build_real_app() -> FastAPI:
    """Build the production app from real settings/deps.

    The runtime LLM config starts unconfigured — users must call
    /config/llm/connect then /config/llm/select before using LLM features.
    """
    from veritas.config import get_settings
    from veritas.pipeline import Pipeline
    from veritas.store.db import Store
    from veritas.store.embeddings import FakeEmbedder, SentenceTransformerEmbedder
    from veritas.store.vectors import VectorIndex

    settings = get_settings()

    store = Store(settings.db_path)
    store.init_schema()

    try:
        embedder = SentenceTransformerEmbedder(settings.embedding_model)
    except (ImportError, OSError, RuntimeError):
        logger.warning(
            "SentenceTransformerEmbedder failed to load; falling back to FakeEmbedder. "
            "Install sentence-transformers for real embeddings."
        )
        embedder = FakeEmbedder()

    index = VectorIndex(settings.chroma_dir, embedder)

    # Pipeline is constructed with a placeholder LLM; the actual LLM is
    # resolved per-request from app.state.llm_config.
    class _UnconfiguredLLM:
        def complete(self, system: str, user: str, **_kw) -> str:
            raise RuntimeError(
                "LLM not configured. Use /config/llm/connect and /config/llm/select."
            )

    placeholder_llm = _UnconfiguredLLM()  # type: ignore[assignment]
    pipeline = Pipeline(store=store, index=index, llm=placeholder_llm, settings=settings)

    # Pass llm=None to signal no pre-injection (runtime config starts empty).
    return build_app(
        store=store,
        index=index,
        pipeline=pipeline,
        llm=None,  # type: ignore[arg-type]
    )


def __getattr__(name: str) -> object:
    """Lazy module-level ``app`` — only built when first accessed."""
    if name == "app":
        global _app_singleton
        if _app_singleton is None:
            _app_singleton = _build_real_app()
        return _app_singleton
    raise AttributeError(name)
