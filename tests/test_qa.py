"""Tests for grounded Q&A (Task 13)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from veritas.llm.client import FakeLLMClient
from veritas.models import Fact
from veritas.store.db import Store
from veritas.store.embeddings import FakeEmbedder
from veritas.store.vectors import VectorIndex

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


def _seed_fact(store: Store, index: VectorIndex) -> Fact:
    """Insert a single fact into the store and index, return it."""
    fact = Fact(
        id="f1",
        doc_id="doc1",
        subject="Company",
        attribute="revenue",
        value="100",
        unit="INR Cr",
        temporal_context="FY24",
        scope_qualifiers=[],
        evidence_span="Revenue was 100 crore in FY24.",
        page=3,
        claim_text="Company revenue FY24 100 INR Cr",
        fact_kind="numerical",
        confidence=0.9,
        normalized_value=1_000_000_000.0,
        normalized_unit="INR",
        period_start="2023-04-01",
        period_end="2024-03-31",
    )
    store.add_facts([fact])
    index.add(
        ids=["f1"],
        texts=["Company revenue FY24 100 INR Cr"],
        metadatas=[{"doc_id": "doc1", "fact_id": "f1"}],
    )
    return fact


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_answer_with_citation(tmp_store: Store, tmp_index: VectorIndex) -> None:
    """answer() should return an answer and a citation resolved to the stored fact."""
    from veritas.qa import answer

    seeded = _seed_fact(tmp_store, tmp_index)

    # The LLM response includes the fact id f1 in its citation list.
    llm_response = json.dumps(
        {
            "answer": "Revenue was 100 crore in FY24.",
            "cited_fact_ids": ["f1"],
        }
    )
    llm = FakeLLMClient([llm_response])

    result = answer("What was the revenue in FY24?", tmp_store, tmp_index, llm, top_k=5)

    assert "answer" in result
    assert result["answer"] == "Revenue was 100 crore in FY24."

    assert "citations" in result
    citations = result["citations"]
    assert len(citations) == 1

    citation = citations[0]
    assert citation["fact_id"] == "f1"
    assert citation["doc_id"] == "doc1"
    assert citation["page"] == seeded.page
    assert citation["evidence_span"] == seeded.evidence_span


def test_empty_retrieval_returns_refusal(tmp_store: Store, tmp_index: VectorIndex) -> None:
    """answer() should refuse (no LLM call) when retrieval returns nothing."""
    from veritas.qa import answer

    # Empty store + index: no facts to retrieve.
    llm = FakeLLMClient([])  # queue is empty; calling complete() would raise

    result = answer("What was the revenue in FY24?", tmp_store, tmp_index, llm, top_k=5)

    assert result["answer"] == "Not enough grounded evidence."
    assert result["citations"] == []


def test_answer_cites_only_retrieved_facts(tmp_store: Store, tmp_index: VectorIndex) -> None:
    """Citations list only includes fact ids that can be resolved via store.get_fact."""
    from veritas.qa import answer

    _seed_fact(tmp_store, tmp_index)

    # LLM tries to cite a non-existent fact id alongside the real one.
    llm_response = json.dumps(
        {
            "answer": "Revenue was 100 crore in FY24.",
            "cited_fact_ids": ["f1", "nonexistent_id"],
        }
    )
    llm = FakeLLMClient([llm_response])

    result = answer("What was the revenue?", tmp_store, tmp_index, llm, top_k=5)

    # Only f1 resolves; nonexistent_id should be silently dropped.
    fact_ids_in_citations = {c["fact_id"] for c in result["citations"]}
    assert "f1" in fact_ids_in_citations
    assert "nonexistent_id" not in fact_ids_in_citations
