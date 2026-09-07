"""Integration tests for the ingestion pipeline (Task 12)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.fixtures.make_pdf import make_pdf
from veritas.llm.client import FakeLLMClient
from veritas.models import new_id
from veritas.store.db import Store
from veritas.store.embeddings import FakeEmbedder
from veritas.store.vectors import VectorIndex

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_extraction_response(
    text: str,
    subject: str = "Company",
    attribute: str = "revenue",
    value: str = "100",
    unit: str = "INR Cr",
    period: str = "FY24",
) -> str:
    """Return a JSON array string simulating LLM extraction for a chunk."""
    evidence_span = text.strip()
    return json.dumps(
        [
            {
                "subject": subject,
                "attribute": attribute,
                "value": value,
                "unit": unit,
                "temporal_context": period,
                "scope_qualifiers": [],
                "evidence_span": evidence_span,
                "claim_text": f"{subject} {attribute} {period} {value} {unit}",
                "fact_kind": "numerical",
                "confidence": 0.9,
            }
        ]
    )


def _contradiction_adjudication() -> str:
    """Return a contradict adjudication JSON."""
    return json.dumps(
        {
            "relation": "contradict",
            "reasoning": "Same period, different values.",
            "reconciling_dimension": "none",
            "confidence": 0.85,
        }
    )


def _build_fake_llm(doc_a_text: str, doc_b_text: str) -> FakeLLMClient:
    """Build a FakeLLMClient keyed on extraction vs. adjudication prompts.

    Discrimination strategy: the extraction system prompt includes the phrase
    "atomic factual claim" while the adjudication system prompt includes
    "classifies their logical relationship". This is reliable because those
    phrases come from the actual prompt builders.

    - Extraction call for doc A → return extraction JSON for doc A.
    - Extraction call for doc B → return extraction JSON for doc B.
    - Adjudication call → return contradict JSON.
    """

    def _respond(system: str, user: str) -> str:
        # Extraction system prompts reference "atomic factual claim"
        if "atomic factual claim" in system:
            # doc A or doc B extraction — check which doc text is in the user prompt
            if doc_a_text in user:
                return _make_extraction_response(doc_a_text, value="100")
            if doc_b_text in user:
                return _make_extraction_response(doc_b_text, value="200")
            # Fallback extraction (empty)
            return "[]"
        # Adjudication system prompt references "classifies their logical relationship"
        return _contradiction_adjudication()

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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ingest_two_docs_produces_facts_and_edges(
    tmp_path: Path, tmp_store: Store, tmp_index: VectorIndex
) -> None:
    """Ingest two docs; expect 2 facts and >=1 contradict edge."""
    from veritas.config import get_settings
    from veritas.pipeline import Pipeline

    doc_a_text = "Revenue was 100 crore in FY24."
    doc_b_text = "Revenue was 200 crore in FY24."

    pdf_a = tmp_path / "doc_a.pdf"
    pdf_b = tmp_path / "doc_b.pdf"
    make_pdf(pdf_a, [doc_a_text])
    make_pdf(pdf_b, [doc_b_text])

    llm = _build_fake_llm(doc_a_text, doc_b_text)

    settings = get_settings()
    pipeline = Pipeline(store=tmp_store, index=tmp_index, llm=llm, settings=settings)

    doc_a = pipeline.ingest_document(pdf_a, "doc_a.pdf")
    doc_b = pipeline.ingest_document(pdf_b, "doc_b.pdf")

    assert doc_a.status == "done"
    assert doc_b.status == "done"

    counts = tmp_store.counts()
    assert counts["facts"] == 2, f"Expected 2 facts, got {counts['facts']}"
    assert counts["edges"] >= 1, f"Expected >=1 edge, got {counts['edges']}"

    # At least one contradict edge
    edges = tmp_store.list_edges()
    relations = {e.relation for e in edges}
    assert "contradict" in relations, f"Expected a contradict edge; got {relations}"


def test_reingest_same_doc_is_noop(
    tmp_path: Path, tmp_store: Store, tmp_index: VectorIndex
) -> None:
    """Re-ingesting a document with the same bytes must be a no-op."""
    from veritas.config import get_settings
    from veritas.pipeline import Pipeline

    doc_a_text = "Revenue was 100 crore in FY24."
    doc_b_text = "Revenue was 200 crore in FY24."

    pdf_a = tmp_path / "doc_a.pdf"
    pdf_b = tmp_path / "doc_b.pdf"
    make_pdf(pdf_a, [doc_a_text])
    make_pdf(pdf_b, [doc_b_text])

    llm = _build_fake_llm(doc_a_text, doc_b_text)
    settings = get_settings()
    pipeline = Pipeline(store=tmp_store, index=tmp_index, llm=llm, settings=settings)

    pipeline.ingest_document(pdf_a, "doc_a.pdf")
    pipeline.ingest_document(pdf_b, "doc_b.pdf")

    counts_after_first = tmp_store.counts()

    # Re-ingest doc_b (same bytes)
    pipeline.ingest_document(pdf_b, "doc_b.pdf")

    counts_after_reingestion = tmp_store.counts()

    assert counts_after_first == counts_after_reingestion, (
        f"Re-ingestion changed counts: {counts_after_first} → {counts_after_reingestion}"
    )


def test_content_hash_is_sha256() -> None:
    """content_hash returns a 64-char hex string."""
    import hashlib
    import os
    import tempfile

    from veritas.pipeline import content_hash

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(b"hello world")
        tmpfile = f.name

    try:
        result = content_hash(tmpfile)
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert result == expected
        assert len(result) == 64
    finally:
        os.unlink(tmpfile)


def test_progress_callback_called(tmp_path: Path, tmp_store: Store, tmp_index: VectorIndex) -> None:
    """progress_cb should be called at each pipeline stage."""
    from veritas.config import get_settings
    from veritas.pipeline import Pipeline

    doc_text = "Revenue was 100 crore in FY24."
    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf, [doc_text])

    def _respond(system: str, user: str) -> str:
        if "atomic factual claim" in system:
            return _make_extraction_response(doc_text)
        return _contradiction_adjudication()

    llm = FakeLLMClient(_respond)
    settings = get_settings()
    pipeline = Pipeline(store=tmp_store, index=tmp_index, llm=llm, settings=settings)

    calls: list[tuple[str, float, str]] = []

    def cb(stage: str, pct: float, msg: str) -> None:
        calls.append((stage, pct, msg))

    pipeline.ingest_document(pdf, "doc.pdf", progress_cb=cb)

    stages = [c[0] for c in calls]
    assert stages, "No progress callbacks were received"
    assert "done" in stages


def test_job_updated_to_done(tmp_path: Path, tmp_store: Store, tmp_index: VectorIndex) -> None:
    """If job_id is provided, the job should be updated to 'done' on success."""
    from veritas.config import get_settings
    from veritas.models import Job
    from veritas.pipeline import Pipeline

    doc_text = "Revenue was 100 crore in FY24."
    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf, [doc_text])

    def _respond(system: str, user: str) -> str:
        if "atomic factual claim" in system:
            return _make_extraction_response(doc_text)
        return _contradiction_adjudication()

    llm = FakeLLMClient(_respond)
    settings = get_settings()
    pipeline = Pipeline(store=tmp_store, index=tmp_index, llm=llm, settings=settings)

    job_id = new_id("job")
    job = Job(
        id=job_id,
        doc_id="placeholder",
        status="queued",
        progress=0.0,
        message="",
        created_at="2026-09-07T00:00:00",
    )
    tmp_store.create_job(job)

    pipeline.ingest_document(pdf, "doc.pdf", job_id=job_id)

    updated_job = tmp_store.get_job(job_id)
    assert updated_job is not None
    assert updated_job.status == "done"


def test_job_doc_id_linked_after_ingestion(
    tmp_path: Path, tmp_store: Store, tmp_index: VectorIndex
) -> None:
    """After ingestion, the job's doc_id must equal the created document's id."""
    from veritas.config import get_settings
    from veritas.models import Job
    from veritas.pipeline import Pipeline

    doc_text = "Revenue was 100 crore in FY24."
    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf, [doc_text])

    def _respond(system: str, user: str) -> str:
        if "atomic factual claim" in system:
            return _make_extraction_response(doc_text)
        return _contradiction_adjudication()

    llm = FakeLLMClient(_respond)
    settings = get_settings()
    pipeline = Pipeline(store=tmp_store, index=tmp_index, llm=llm, settings=settings)

    job_id = new_id("job")
    job = Job(
        id=job_id,
        doc_id="placeholder",
        status="queued",
        progress=0.0,
        message="",
        created_at="2026-09-07T00:00:00",
    )
    tmp_store.create_job(job)

    result_doc = pipeline.ingest_document(pdf, "doc.pdf", job_id=job_id)

    updated_job = tmp_store.get_job(job_id)
    assert updated_job is not None
    assert updated_job.doc_id == result_doc.id, (
        f"Expected job.doc_id={result_doc.id!r}, got {updated_job.doc_id!r}"
    )


def test_pipeline_error_sets_document_and_job_to_error(
    tmp_path: Path, tmp_store: Store, tmp_index: VectorIndex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing pipeline stage must mark both the Document and Job as 'error'
    and re-raise the exception.

    Injection point: ``store.add_facts`` is monkeypatched to raise so that the
    error occurs inside the pipeline's try block and propagates to the caller.
    (``extract_facts`` is resilient per-chunk; faults there are swallowed.
    ``add_facts`` is a discrete store call that propagates normally.)
    """
    from veritas.config import get_settings
    from veritas.models import Job
    from veritas.pipeline import Pipeline

    doc_text = "Revenue was 100 crore in FY24."
    pdf = tmp_path / "doc.pdf"
    make_pdf(pdf, [doc_text])

    def _respond(system: str, user: str) -> str:
        if "atomic factual claim" in system:
            return _make_extraction_response(doc_text)
        return _contradiction_adjudication()

    llm = FakeLLMClient(_respond)
    settings = get_settings()
    pipeline = Pipeline(store=tmp_store, index=tmp_index, llm=llm, settings=settings)

    # Inject failure into the persist-facts stage.
    def _boom(*_a: object, **_kw: object) -> None:
        raise RuntimeError("injected add_facts failure")

    monkeypatch.setattr(tmp_store, "add_facts", _boom)

    job_id = new_id("job")
    job = Job(
        id=job_id,
        doc_id="placeholder",
        status="queued",
        progress=0.0,
        message="",
        created_at="2026-09-07T00:00:00",
    )
    tmp_store.create_job(job)

    with pytest.raises(RuntimeError, match="injected add_facts failure"):
        pipeline.ingest_document(pdf, "doc.pdf", job_id=job_id)

    # The document created during the pipeline run must end up with status='error'.
    docs = tmp_store.list_documents()
    assert len(docs) == 1, f"Expected 1 document record, got {len(docs)}"
    assert docs[0].status == "error", f"Expected doc status='error', got {docs[0].status!r}"

    # The job must also end up with status='error'.
    updated_job = tmp_store.get_job(job_id)
    assert updated_job is not None
    assert updated_job.status == "error", f"Expected job status='error', got {updated_job.status!r}"
