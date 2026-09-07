"""Tests for Task 11: Fact canonicalization via corroboration clustering."""

from __future__ import annotations

from veritas.models import Edge, Fact, new_id
from veritas.relate.canonicalize import canonical_clusters, canonical_label

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MinimalFact:
    """Minimal stand-in satisfying the `.id` attribute requirement."""

    def __init__(self, fact_id: str) -> None:
        self.id = fact_id


def _make_fact(fact_id: str, claim_text: str, confidence: float) -> Fact:
    return Fact(
        id=fact_id,
        doc_id="doc1",
        subject="Co",
        attribute="revenue",
        value="100",
        unit="INR Cr",
        temporal_context="FY24",
        scope_qualifiers=[],
        evidence_span=f"Evidence for {fact_id}",
        page=1,
        claim_text=claim_text,
        fact_kind="numerical",
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Provided test (from the plan)
# ---------------------------------------------------------------------------


def test_union_find_clusters():
    """Corroborating facts are clustered; unconnected facts are singletons."""
    facts = [_MinimalFact("a"), _MinimalFact("b"), _MinimalFact("c")]
    edges = [Edge(new_id("e"), "a", "b", "corroborate", "same", "none", 0.9)]
    clusters = canonical_clusters(facts, edges)
    assert sorted(sorted(c) for c in clusters) == [["a", "b"], ["c"]]


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------


def test_all_singletons_when_no_corroborate_edges():
    """Every fact is its own cluster when there are no corroborate edges."""
    facts = [_MinimalFact("x"), _MinimalFact("y"), _MinimalFact("z")]
    edges = [
        Edge(new_id("e"), "x", "y", "contradict", "conflict", "none", 0.8),
        Edge(new_id("e"), "y", "z", "reconcilable", "diff period", "time", 0.7),
    ]
    clusters = canonical_clusters(facts, edges)
    assert sorted(sorted(c) for c in clusters) == [["x"], ["y"], ["z"]]


def test_transitive_corroboration():
    """Union-find correctly merges transitively: a-b corroborate, b-c corroborate → {a,b,c}."""
    facts = [_MinimalFact("a"), _MinimalFact("b"), _MinimalFact("c")]
    edges = [
        Edge(new_id("e"), "a", "b", "corroborate", "same", "none", 0.9),
        Edge(new_id("e"), "b", "c", "corroborate", "same", "none", 0.85),
    ]
    clusters = canonical_clusters(facts, edges)
    assert sorted(sorted(c) for c in clusters) == [["a", "b", "c"]]


def test_every_fact_id_in_exactly_one_cluster():
    """Every fact id appears in exactly one cluster (no duplicates, no omissions)."""
    ids = ["f1", "f2", "f3", "f4", "f5"]
    facts = [_MinimalFact(i) for i in ids]
    edges = [
        Edge(new_id("e"), "f1", "f2", "corroborate", "same", "none", 0.9),
        Edge(new_id("e"), "f3", "f4", "corroborate", "same", "none", 0.85),
    ]
    clusters = canonical_clusters(facts, edges)

    all_ids = [fid for cluster in clusters for fid in cluster]
    assert sorted(all_ids) == sorted(ids)
    # Each id appears exactly once
    assert len(all_ids) == len(set(all_ids))


def test_canonical_label_picks_highest_confidence():
    """canonical_label returns the claim_text of the highest-confidence fact."""
    facts = [
        _make_fact("f1", "Low confidence claim", 0.5),
        _make_fact("f2", "High confidence claim", 0.95),
        _make_fact("f3", "Medium confidence claim", 0.7),
    ]
    label = canonical_label(facts)
    assert label == "High confidence claim"


def test_canonical_label_single_fact():
    """canonical_label works with a single-element cluster."""
    facts = [_make_fact("f1", "Only claim", 0.8)]
    assert canonical_label(facts) == "Only claim"
