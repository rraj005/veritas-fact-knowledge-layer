"""Tests for eval/evaluate.py — metric math with a synthetic store + gold set.

This test does NOT require any LLM, real data, or API keys.
It seeds a small in-memory Store, adds known Facts and Edges, defines a
synthetic gold dict, calls evaluate(), and asserts the metric values
are arithmetically correct.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make eval/ importable from tests/.
sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

import pytest

# Import the module under test.
from evaluate import evaluate, format_report  # type: ignore[import]

from veritas.models import Edge, Fact, new_id
from veritas.store.db import Store

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_store(tmp_path: Path) -> Store:
    """Return a fresh, initialised Store with known facts and edges."""
    store = Store(tmp_path / "test_eval.db")
    store.init_schema()
    return store


def _fact(
    doc_id: str,
    claim_text: str,
    evidence_span: str,
    normalized_value: float | None = 42.0,
    confidence: float = 0.9,
) -> Fact:
    return Fact(
        id=new_id("fact"),
        doc_id=doc_id,
        subject="TestSubject",
        attribute="revenue",
        value="42",
        unit="USD",
        temporal_context="FY24",
        scope_qualifiers=[],
        evidence_span=evidence_span,
        page=1,
        claim_text=claim_text,
        fact_kind="numerical",
        confidence=confidence,
        normalized_value=normalized_value,
        normalized_unit="USD",
        period_start="2023-04-01",
        period_end="2024-03-31",
        canonical_subject="testsubject",
    )


# ---------------------------------------------------------------------------
# Concrete seeded scenario
# ---------------------------------------------------------------------------

#   fact_a  — "annual revenue grew" — has evidence_span, has normalized_value
#   fact_b  — "revenue increased"   — has evidence_span, has normalized_value
#   fact_c  — "gdp expanded"        — has evidence_span, NO normalized_value
#
#   edge_ab  — corroborate between fact_a and fact_b
#
#   gold dict:
#     expected_relation 1: a_contains="annual revenue",  b_contains="revenue increased"
#                           relation="reconcilable_or_corroborate"  → MATCHED (edge_ab is corroborate)
#     expected_relation 2: a_contains="gdp expanded",    b_contains="annual revenue"
#                           relation="contradict"                   → NOT MATCHED (no such edge)

GOLD = {
    "_comment": "Small hand-checked gold set for unit tests.",
    "expected_relations": [
        {
            "a_contains": "annual revenue",
            "b_contains": "revenue increased",
            "relation": "reconcilable_or_corroborate",
            "note": "same metric, same period — should be corroborated",
        },
        {
            "a_contains": "gdp expanded",
            "b_contains": "annual revenue",
            "relation": "contradict",
            "note": "no contradict edge in test store — should be unmatched",
        },
    ],
}


@pytest.fixture()
def seeded_store(tmp_path: Path) -> tuple[Store, Fact, Fact, Fact, Edge]:
    """Return a store pre-loaded with 3 facts and 1 corroborate edge."""
    store = _make_store(tmp_path)

    fa = _fact(
        "doc_a",
        "annual revenue grew significantly in FY24",
        "annual revenue grew significantly in FY24",
    )
    fb = _fact("doc_b", "revenue increased year on year", "revenue increased year on year")
    fc = _fact(
        "doc_c", "gdp expanded by 6 percent", "gdp expanded by 6 percent", normalized_value=None
    )

    store.add_facts([fa, fb, fc])

    edge = Edge(
        id=new_id("edge"),
        fact_a_id=fa.id,
        fact_b_id=fb.id,
        relation="corroborate",
        reasoning="Both claim revenue grew in FY24.",
        reconciling_dimension="none",
        confidence=0.92,
    )
    store.add_edges([edge])

    return store, fa, fb, fc, edge


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractionSanity:
    def test_total_facts(self, seeded_store: tuple) -> None:
        store, *_ = seeded_store
        metrics = evaluate(store, GOLD)
        assert metrics["total_facts"] == 3

    def test_evidence_span_pct(self, seeded_store: tuple) -> None:
        """All 3 facts have non-empty evidence_span → 100 %."""
        store, *_ = seeded_store
        metrics = evaluate(store, GOLD)
        assert metrics["pct_with_evidence_span"] == pytest.approx(100.0)

    def test_normalized_value_pct(self, seeded_store: tuple) -> None:
        """2 of 3 facts have normalized_value → ~66.67 %."""
        store, *_ = seeded_store
        metrics = evaluate(store, GOLD)
        assert metrics["pct_with_normalized_value"] == pytest.approx(200 / 3, rel=1e-3)


class TestRelationshipMetrics:
    def test_matched_count(self, seeded_store: tuple) -> None:
        """1 of 2 expected relations is matched."""
        store, *_ = seeded_store
        metrics = evaluate(store, GOLD)
        assert metrics["gold_matched"] == 1
        assert metrics["gold_total"] == 2

    def test_precision(self, seeded_store: tuple) -> None:
        """50 % of gold relations matched → precision 50.0."""
        store, *_ = seeded_store
        metrics = evaluate(store, GOLD)
        assert metrics["gold_match_pct"] == pytest.approx(50.0)

    def test_per_item_breakdown(self, seeded_store: tuple) -> None:
        """Per-item list has 2 entries; first matched, second not."""
        store, *_ = seeded_store
        metrics = evaluate(store, GOLD)
        breakdown = metrics["per_item"]
        assert len(breakdown) == 2
        assert breakdown[0]["matched"] is True
        assert breakdown[1]["matched"] is False

    def test_no_gold_edge_case(self, seeded_store: tuple) -> None:
        """Empty gold list → 0 total, 0 matched, 0 %."""
        store, *_ = seeded_store
        empty_gold = {"expected_relations": []}
        metrics = evaluate(store, empty_gold)
        assert metrics["gold_total"] == 0
        assert metrics["gold_matched"] == 0
        assert metrics["gold_match_pct"] == pytest.approx(0.0)


class TestFormatReport:
    def test_format_report_is_string(self, seeded_store: tuple) -> None:
        store, *_ = seeded_store
        metrics = evaluate(store, GOLD)
        report = format_report(metrics)
        assert isinstance(report, str)
        assert "total_facts" in report or "Total facts" in report

    def test_format_report_contains_pct(self, seeded_store: tuple) -> None:
        store, *_ = seeded_store
        metrics = evaluate(store, GOLD)
        report = format_report(metrics)
        # Should mention some percentage
        assert "%" in report


class TestEdgeCases:
    def test_empty_store(self, tmp_path: Path) -> None:
        """Empty store → 0 facts, 0 %, 0 % matched."""
        store = _make_store(tmp_path)
        metrics = evaluate(store, GOLD)
        assert metrics["total_facts"] == 0
        assert metrics["pct_with_evidence_span"] == pytest.approx(0.0)
        assert metrics["pct_with_normalized_value"] == pytest.approx(0.0)
        assert metrics["gold_matched"] == 0

    def test_reconcilable_satisfies_reconcilable_or_corroborate(self, tmp_path: Path) -> None:
        """'reconcilable_or_corroborate' gold relation is satisfied by a 'reconcilable' edge."""
        store = _make_store(tmp_path)

        fa = _fact("doc_x", "total exports rose last year", "total exports rose last year")
        fb = _fact("doc_y", "exports increased significantly", "exports increased significantly")
        store.add_facts([fa, fb])

        edge = Edge(
            id=new_id("edge"),
            fact_a_id=fa.id,
            fact_b_id=fb.id,
            relation="reconcilable",
            reasoning="Different periods but same trend.",
            reconciling_dimension="time",
            confidence=0.80,
        )
        store.add_edges([edge])

        gold = {
            "expected_relations": [
                {
                    "a_contains": "total exports",
                    "b_contains": "exports increased",
                    "relation": "reconcilable_or_corroborate",
                    "note": "should match reconcilable edge",
                }
            ]
        }
        metrics = evaluate(store, gold)
        assert metrics["gold_matched"] == 1

    def test_corroborate_satisfies_reconcilable_or_corroborate(self, tmp_path: Path) -> None:
        """'reconcilable_or_corroborate' gold relation is satisfied by a 'corroborate' edge."""
        store = _make_store(tmp_path)

        fa = _fact("doc_p", "inflation rate was 4 percent", "inflation rate was 4 percent")
        fb = _fact("doc_q", "consumer price inflation 4 pct", "consumer price inflation 4 pct")
        store.add_facts([fa, fb])

        edge = Edge(
            id=new_id("edge"),
            fact_a_id=fa.id,
            fact_b_id=fb.id,
            relation="corroborate",
            reasoning="Same figure, different sources.",
            reconciling_dimension="none",
            confidence=0.95,
        )
        store.add_edges([edge])

        gold = {
            "expected_relations": [
                {
                    "a_contains": "inflation rate",
                    "b_contains": "consumer price inflation",
                    "relation": "reconcilable_or_corroborate",
                    "note": "should match corroborate edge",
                }
            ]
        }
        metrics = evaluate(store, gold)
        assert metrics["gold_matched"] == 1

    def test_contradict_gold_matched_when_contradict_edge_exists(self, tmp_path: Path) -> None:
        """A 'contradict' gold item IS matched when a contradict edge exists between the facts."""
        store = _make_store(tmp_path)

        # Two facts about the same metric for the same period with incompatible values.
        fa = _fact(
            "doc_r",
            "fiscal deficit was 5.9 percent of gdp in fy23",
            "fiscal deficit was 5.9 percent of gdp in fy23",
        )
        fb = _fact(
            "doc_s",
            "fiscal deficit reached 6.4 percent of gdp fy23",
            "fiscal deficit reached 6.4 percent of gdp fy23",
        )
        store.add_facts([fa, fb])

        edge = Edge(
            id=new_id("edge"),
            fact_a_id=fa.id,
            fact_b_id=fb.id,
            relation="contradict",
            reasoning="One source states 5.9%, another states 6.4% for the same fiscal year and scope.",
            reconciling_dimension="none",
            confidence=0.88,
        )
        store.add_edges([edge])

        gold = {
            "expected_relations": [
                {
                    "a_contains": "fiscal deficit",
                    "b_contains": "fiscal deficit",
                    "relation": "contradict",
                    "note": "incompatible numerical values for the same period",
                }
            ]
        }
        metrics = evaluate(store, gold)
        assert metrics["gold_matched"] == 1, "contradict gold item should match a contradict edge"
        assert metrics["per_item"][0]["matched"] is True

    def test_contradict_gold_unmatched_when_only_corroborate_edge_exists(
        self, tmp_path: Path
    ) -> None:
        """A 'contradict' gold item is NOT matched when only a corroborate edge exists."""
        store = _make_store(tmp_path)

        fa = _fact(
            "doc_t",
            "fiscal deficit was 5.9 percent of gdp in fy23",
            "fiscal deficit was 5.9 percent of gdp in fy23",
        )
        fb = _fact(
            "doc_u",
            "fiscal deficit reached 5.9 percent of gdp fy23",
            "fiscal deficit reached 5.9 percent of gdp fy23",
        )
        store.add_facts([fa, fb])

        edge = Edge(
            id=new_id("edge"),
            fact_a_id=fa.id,
            fact_b_id=fb.id,
            relation="corroborate",
            reasoning="Both sources agree on 5.9% for FY23.",
            reconciling_dimension="none",
            confidence=0.95,
        )
        store.add_edges([edge])

        gold = {
            "expected_relations": [
                {
                    "a_contains": "fiscal deficit",
                    "b_contains": "fiscal deficit",
                    "relation": "contradict",
                    "note": "gold expects contradiction but only corroborate edge exists",
                }
            ]
        }
        metrics = evaluate(store, gold)
        assert metrics["gold_matched"] == 0, (
            "contradict gold item must NOT match a corroborate edge"
        )
        assert metrics["per_item"][0]["matched"] is False

    def test_multiple_edges_same_pair_both_checked(self, tmp_path: Path) -> None:
        """When two edges exist for the same fact pair, both relations are checked (no overwrite)."""
        store = _make_store(tmp_path)

        fa = _fact("doc_v", "market share was 30 percent", "market share was 30 percent")
        fb = _fact("doc_w", "market share reached 35 percent", "market share reached 35 percent")
        store.add_facts([fa, fb])

        # First edge: corroborate (would overwrite the second if we used a plain dict)
        edge1 = Edge(
            id=new_id("edge"),
            fact_a_id=fa.id,
            fact_b_id=fb.id,
            relation="corroborate",
            reasoning="Same trend, similar figure.",
            reconciling_dimension="none",
            confidence=0.70,
        )
        # Second edge: contradict (the one we want to find via the gold item)
        edge2 = Edge(
            id=new_id("edge"),
            fact_a_id=fa.id,
            fact_b_id=fb.id,
            relation="contradict",
            reasoning="30% vs 35% for the same period is a meaningful numerical gap.",
            reconciling_dimension="none",
            confidence=0.85,
        )
        store.add_edges([edge1, edge2])

        gold = {
            "expected_relations": [
                {
                    "a_contains": "market share",
                    "b_contains": "market share",
                    "relation": "contradict",
                    "note": "second edge should be found even when first edge is corroborate",
                }
            ]
        }
        metrics = evaluate(store, gold)
        assert metrics["gold_matched"] == 1, (
            "contradict edge must be found even when a corroborate edge also exists for the same pair"
        )
