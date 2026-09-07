"""Tests for Task 10: Relationship engine (RAG core)."""

from __future__ import annotations

import json

from veritas.llm.client import FakeLLMClient
from veritas.models import Fact, new_id
from veritas.relate.relationships import adjudicate, find_relationships

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _f(doc: str, val: str, period: str) -> Fact:
    return Fact(
        new_id("f"),
        doc,
        "Co",
        "revenue",
        val,
        "INR Cr",
        period,
        [],
        f"rev {val}",
        1,
        f"Revenue {period} {val}",
        "numerical",
        0.9,
    )


# ---------------------------------------------------------------------------
# Provided tests (from the plan)
# ---------------------------------------------------------------------------


def test_adjudicate_contradiction():
    resp = json.dumps(
        {
            "relation": "contradict",
            "reasoning": "Same period, different value",
            "reconciling_dimension": "none",
            "confidence": 0.8,
        }
    )
    e = adjudicate(_f("a", "100", "FY24"), _f("b", "200", "FY24"), FakeLLMClient([resp]))
    assert e is not None
    assert e.relation == "contradict"


def test_adjudicate_reconcilable():
    resp = json.dumps(
        {
            "relation": "reconcilable",
            "reasoning": "Different periods",
            "reconciling_dimension": "time",
            "confidence": 0.85,
        }
    )
    e = adjudicate(_f("a", "100", "FY23"), _f("b", "200", "FY24"), FakeLLMClient([resp]))
    assert e is not None
    assert e.relation == "reconcilable" and e.reconciling_dimension == "time"


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------


def test_adjudicate_parse_failure_returns_none():
    """Returns None on unparseable LLM response."""
    e = adjudicate(_f("a", "100", "FY24"), _f("b", "200", "FY24"), FakeLLMClient(["NOT JSON"]))
    assert e is None


def test_adjudicate_returns_edge_for_unrelated():
    """adjudicate returns Edge even if relation is 'unrelated'; caller filters."""
    resp = json.dumps(
        {
            "relation": "unrelated",
            "reasoning": "Different topics entirely",
            "reconciling_dimension": "none",
            "confidence": 0.3,
        }
    )
    e = adjudicate(_f("a", "100", "FY24"), _f("b", "200", "FY24"), FakeLLMClient([resp]))
    assert e is not None
    assert e.relation == "unrelated"


class _FakeStore:
    """Minimal store stub for find_relationships tests."""

    def __init__(self, facts: list[Fact]) -> None:
        self._facts = {f.id: f for f in facts}

    def get_fact(self, fact_id: str) -> Fact | None:
        return self._facts.get(fact_id)


class _RecordingIndex:
    """Fake index that records the where-filter passed to query()."""

    def __init__(self, results: list[tuple[str, float, dict]]) -> None:
        self._results = results
        self.last_where: dict | None = None

    def query(
        self, text: str, top_k: int, where: dict | None = None
    ) -> list[tuple[str, float, dict]]:
        self.last_where = where
        return self._results


def test_find_relationships_filters_out_unrelated():
    """find_relationships drops edges whose relation is 'unrelated'."""
    fact_a = _f("doc_a", "100", "FY24")
    fact_b = _f("doc_b", "200", "FY24")

    store = _FakeStore([fact_a, fact_b])

    unrelated_resp = json.dumps(
        {
            "relation": "unrelated",
            "reasoning": "Not relevant",
            "reconciling_dimension": "none",
            "confidence": 0.2,
        }
    )
    llm = FakeLLMClient([unrelated_resp])
    index = _RecordingIndex([(fact_b.id, 0.1, {"doc_id": "doc_b"})])

    edges = find_relationships(fact_a, store, index, llm, top_k=8)
    assert edges == []


def test_find_relationships_deduplicates_repeated_candidate_pair():
    """If the same candidate pair appears twice in results, adjudicate only once."""
    fact_a = _f("doc_a", "100", "FY24")
    fact_b = _f("doc_b", "200", "FY24")

    store = _FakeStore([fact_a, fact_b])

    contradict_resp = json.dumps(
        {
            "relation": "contradict",
            "reasoning": "Different values",
            "reconciling_dimension": "none",
            "confidence": 0.9,
        }
    )
    # Two LLM responses; if dedup fails, the second one gets consumed too
    llm = FakeLLMClient([contradict_resp, contradict_resp])

    # Return the same fact_b id twice (simulating duplicate retrieval)
    index = _RecordingIndex(
        [
            (fact_b.id, 0.1, {"doc_id": "doc_b"}),
            (fact_b.id, 0.15, {"doc_id": "doc_b"}),
        ]
    )

    edges = find_relationships(fact_a, store, index, llm, top_k=8)
    # Only one edge should result (deduplicated)
    assert len(edges) == 1
    assert edges[0].relation == "contradict"


def test_find_relationships_passes_ne_where_filter():
    """find_relationships passes {'doc_id': {'$ne': fact.doc_id}} to index.query."""
    fact_a = _f("doc_a", "100", "FY24")
    store = _FakeStore([fact_a])

    # index returns no results; we just inspect the where arg
    index = _RecordingIndex([])
    llm = FakeLLMClient([])

    find_relationships(fact_a, store, index, llm, top_k=8)

    assert index.last_where == {"doc_id": {"$ne": fact_a.doc_id}}


def test_adjudicate_reasoning_with_brace_inside_string():
    """adjudicate succeeds when the LLM reasoning value contains a '}' character.

    The lazy regex ``\\{.*?\\}`` would stop at the first ``}`` inside the
    reasoning string and produce a truncated (invalid) JSON snippet.  The
    greedy ``\\{[\\s\\S]*\\}`` must capture the whole outermost object.
    """
    resp = json.dumps(
        {
            "relation": "corroborate",
            "reasoning": "see note {a} for details",
            "reconciling_dimension": "none",
            "confidence": 0.9,
        }
    )
    e = adjudicate(_f("a", "100", "FY24"), _f("b", "100", "FY24"), FakeLLMClient([resp]))
    assert e is not None
    assert e.relation == "corroborate"
    assert "}" in e.reasoning
