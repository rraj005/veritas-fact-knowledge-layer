"""Tests for Task 8: domain-neutral LLM fact extraction with grounding."""

from __future__ import annotations

import json

from veritas.extraction.extractor import extract_facts, extract_from_chunk
from veritas.llm.client import FakeLLMClient
from veritas.models import Chunk


def test_extracts_and_grounds() -> None:
    text = "Revenue was Rs 8,142 crore in FY24. The sky is blue."
    resp = json.dumps(
        [
            {
                "subject": "Company",
                "attribute": "revenue",
                "value": "8142",
                "unit": "INR Cr",
                "temporal_context": "FY24",
                "scope_qualifiers": [],
                "evidence_span": "Revenue was Rs 8,142 crore in FY24",
                "claim_text": "Revenue in FY24 was 8142 INR Cr",
                "fact_kind": "numerical",
                "confidence": 0.95,
            }
        ]
    )
    c = Chunk("d1", 5, text, 0, len(text))
    facts = extract_from_chunk(c, FakeLLMClient([resp]))
    assert len(facts) == 1
    assert facts[0].page == 5 and facts[0].doc_id == "d1"
    assert facts[0].evidence_span in text  # grounding enforced


def test_drops_ungrounded() -> None:
    text = "Revenue was 100 crore."
    resp = json.dumps(
        [
            {
                "subject": "X",
                "attribute": "y",
                "value": "z",
                "unit": "",
                "temporal_context": "",
                "scope_qualifiers": [],
                "evidence_span": "NOT IN TEXT",
                "claim_text": "c",
                "fact_kind": "semantic",
                "confidence": 0.5,
            }
        ]
    )
    facts = extract_from_chunk(Chunk("d1", 1, text, 0, len(text)), FakeLLMClient([resp]))
    assert facts == []


def test_extract_facts_thread_pool() -> None:
    """extract_facts should aggregate facts from multiple chunks concurrently."""
    text1 = "Population was 1.4 billion in 2023."
    text2 = "Growth rate was 7 percent in 2023."
    resp1 = json.dumps(
        [
            {
                "subject": "Country",
                "attribute": "population",
                "value": "1.4",
                "unit": "billion",
                "temporal_context": "2023",
                "scope_qualifiers": [],
                "evidence_span": "Population was 1.4 billion in 2023",
                "claim_text": "Population in 2023 was 1.4 billion",
                "fact_kind": "numerical",
                "confidence": 0.9,
            }
        ]
    )
    resp2 = json.dumps(
        [
            {
                "subject": "Country",
                "attribute": "growth rate",
                "value": "7",
                "unit": "%",
                "temporal_context": "2023",
                "scope_qualifiers": [],
                "evidence_span": "Growth rate was 7 percent in 2023",
                "claim_text": "Growth rate in 2023 was 7 percent",
                "fact_kind": "numerical",
                "confidence": 0.85,
            }
        ]
    )

    chunks = [
        Chunk("d1", 1, text1, 0, len(text1)),
        Chunk("d1", 2, text2, 0, len(text2)),
    ]

    # Use a callable-based fake keyed on the user prompt text so it's deterministic
    responses_map = {text1: resp1, text2: resp2}

    def fake_fn(system: str, user: str) -> str:
        for key, val in responses_map.items():
            if key in user:
                return val
        return "[]"

    facts = extract_facts(chunks, FakeLLMClient(fake_fn), concurrency=2)
    assert len(facts) == 2
    attributes = {f.attribute for f in facts}
    assert "population" in attributes
    assert "growth rate" in attributes


def test_malformed_row_dropped() -> None:
    """Rows missing required fields should be silently dropped."""
    text = "Exports grew by 12 percent."
    # missing evidence_span field entirely
    resp = json.dumps(
        [
            {
                "subject": "Country",
                "attribute": "exports",
                "value": "12",
                "unit": "%",
                "temporal_context": "",
                "scope_qualifiers": [],
                # evidence_span missing!
                "claim_text": "Exports grew 12%",
                "fact_kind": "numerical",
                "confidence": 0.8,
            }
        ]
    )
    facts = extract_from_chunk(Chunk("d1", 1, text, 0, len(text)), FakeLLMClient([resp]))
    assert facts == []


def test_empty_llm_response() -> None:
    """Empty array response produces no facts."""
    text = "The document is mostly formatting."
    facts = extract_from_chunk(Chunk("d1", 1, text, 0, len(text)), FakeLLMClient(["[]"]))
    assert facts == []


def test_empty_evidence_span_dropped() -> None:
    """A fact with evidence_span present but empty string should be dropped."""
    text = "Sales increased by 15 percent."
    resp = json.dumps(
        [
            {
                "subject": "Company",
                "attribute": "sales",
                "value": "15",
                "unit": "%",
                "temporal_context": "",
                "scope_qualifiers": [],
                "evidence_span": "",  # present but empty
                "claim_text": "Sales increased by 15 percent",
                "fact_kind": "numerical",
                "confidence": 0.8,
            }
        ]
    )
    facts = extract_from_chunk(Chunk("d1", 1, text, 0, len(text)), FakeLLMClient([resp]))
    assert facts == []


def test_whitespace_collapsed_grounding() -> None:
    """Evidence span with collapsed whitespace should still be grounded."""
    text = "Total  assets  were  500  crore."
    # LLM collapses whitespace in the span
    evidence = "Total assets were 500 crore"
    resp = json.dumps(
        [
            {
                "subject": "Company",
                "attribute": "total assets",
                "value": "500",
                "unit": "INR Cr",
                "temporal_context": "",
                "scope_qualifiers": [],
                "evidence_span": evidence,
                "claim_text": "Total assets were 500 crore",
                "fact_kind": "numerical",
                "confidence": 0.88,
            }
        ]
    )
    facts = extract_from_chunk(Chunk("d1", 1, text, 0, len(text)), FakeLLMClient([resp]))
    # Should be accepted since whitespace-collapsed form matches
    assert len(facts) == 1
