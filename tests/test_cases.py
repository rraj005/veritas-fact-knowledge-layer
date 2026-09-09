"""Tests for the four-cases builder and export (Task 15)."""

from __future__ import annotations

from pathlib import Path

import pytest

from veritas.models import Document, Edge, Fact, new_id
from veritas.store.db import Store

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def seeded_store(tmp_path: Path) -> Store:
    """Return a Store pre-seeded with one edge of each relation type."""
    s = Store(tmp_path / "t.db")
    s.init_schema()

    # Documents
    doc_a = Document(new_id("doc"), "a.pdf", "hash_a", 5, "done", "2026-01-01T00:00:00")
    doc_b = Document(new_id("doc"), "b.pdf", "hash_b", 3, "done", "2026-01-01T00:00:00")
    s.upsert_document(doc_a)
    s.upsert_document(doc_b)

    # Fact helpers
    def _f(doc_id: str, attr: str = "revenue", value: str = "100", conf: float = 0.9) -> Fact:
        return Fact(
            new_id("fact"),
            doc_id,
            "Company",
            attr,
            value,
            "INR Cr",
            "FY24",
            [],
            f"{attr} {value} Cr",
            1,
            f"{attr} FY24 {value}",
            "numerical",
            conf,
        )

    # Six facts — two per edge type
    fa1 = _f(doc_a.id, "revenue", "100")
    fb1 = _f(doc_b.id, "revenue", "100")  # corroborate (same value)

    fa2 = _f(doc_a.id, "profit", "50")
    fb2 = _f(doc_b.id, "profit", "200")  # contradict (different value)

    fa3 = _f(doc_a.id, "gdp", "5000")
    fb3 = _f(doc_b.id, "gdp", "5100")  # reconcilable (slightly different)

    # Low-confidence fact for the failure slot
    fa4 = _f(doc_a.id, "misc", "0", conf=0.1)

    s.add_facts([fa1, fb1, fa2, fb2, fa3, fb3, fa4])

    # Edges
    e_corr = Edge(new_id("edge"), fa1.id, fb1.id, "corroborate", "Same value", "none", 0.9)
    e_cont = Edge(new_id("edge"), fa2.id, fb2.id, "contradict", "Different values", "none", 0.85)
    e_rec = Edge(new_id("edge"), fa3.id, fb3.id, "reconcilable", "Slight diff", "time", 0.8)
    s.add_edges([e_corr, e_cont, e_rec])

    # Store IDs for test assertions
    s._test_edge_corr_id = e_corr.id  # type: ignore[attr-defined]
    s._test_edge_cont_id = e_cont.id  # type: ignore[attr-defined]
    s._test_edge_rec_id = e_rec.id  # type: ignore[attr-defined]
    s._test_fact_low_conf_id = fa4.id  # type: ignore[attr-defined]

    return s


# ---------------------------------------------------------------------------
# Tests for build_cases
# ---------------------------------------------------------------------------


def test_build_cases_structure(seeded_store: Store) -> None:
    """build_cases returns the four expected keys."""
    from veritas.cases import build_cases

    result = build_cases(seeded_store)
    assert set(result.keys()) >= {"corroborate", "contradict", "reconcilable", "failure"}


def test_build_cases_corroborate_has_edge(seeded_store: Store) -> None:
    """build_cases corroborate slot contains the seeded corroborate edge id."""
    from veritas.cases import build_cases

    result = build_cases(seeded_store)
    corr = result["corroborate"]
    assert corr is not None
    assert corr["edge"]["id"] == seeded_store._test_edge_corr_id  # type: ignore[attr-defined]


def test_build_cases_contradict_has_evidence(seeded_store: Store) -> None:
    """build_cases contradict slot has both facts with evidence populated."""
    from veritas.cases import build_cases

    result = build_cases(seeded_store)
    cont = result["contradict"]
    assert cont is not None
    assert "fact_a" in cont
    assert "fact_b" in cont
    fa = cont["fact_a"]
    assert fa["evidence_span"] != ""


def test_build_cases_reconcilable_has_dimension(seeded_store: Store) -> None:
    """build_cases reconcilable slot has reconciling_dimension populated."""
    from veritas.cases import build_cases

    result = build_cases(seeded_store)
    rec = result["reconcilable"]
    assert rec is not None
    assert rec["edge"]["reconciling_dimension"] == "time"


def test_build_cases_failure_is_lowest_confidence(seeded_store: Store) -> None:
    """build_cases failure slot contains the lowest-confidence fact."""
    from veritas.cases import build_cases

    result = build_cases(seeded_store)
    failure = result["failure"]
    assert failure is not None
    assert failure["id"] == seeded_store._test_fact_low_conf_id  # type: ignore[attr-defined]


def test_build_cases_empty_store(tmp_path: Path) -> None:
    """build_cases returns nulls for all slots when the store is empty."""
    from veritas.cases import build_cases

    s = Store(tmp_path / "empty.db")
    s.init_schema()
    result = build_cases(s)
    assert result["corroborate"] is None
    assert result["contradict"] is None
    assert result["reconcilable"] is None
    assert result["failure"] is None


# ---------------------------------------------------------------------------
# Tests for export_layer and export_csv
# ---------------------------------------------------------------------------


def test_export_layer_returns_facts_and_edges(seeded_store: Store) -> None:
    """export_layer returns a dict with facts and edges lists."""
    from veritas.cases import export_layer

    result = export_layer(seeded_store)
    assert "facts" in result
    assert "edges" in result
    assert len(result["facts"]) >= 1
    assert len(result["edges"]) >= 1


def test_export_csv_has_header(seeded_store: Store) -> None:
    """export_csv returns a non-empty CSV string with a header row."""
    from veritas.cases import export_csv

    csv_str = export_csv(seeded_store)
    lines = csv_str.strip().splitlines()
    assert len(lines) >= 2  # header + at least one data row
    # Header should contain at least 'id'
    assert "id" in lines[0].lower()


def test_export_csv_edges(seeded_store: Store) -> None:
    """export_csv with kind='edges' returns an edges CSV."""
    from veritas.cases import export_csv

    csv_str = export_csv(seeded_store, kind="edges")
    lines = csv_str.strip().splitlines()
    assert len(lines) >= 2
    # Header should reference relation
    assert "relation" in lines[0].lower()


# ---------------------------------------------------------------------------
# Tests for Fix 3 — distinct, highest-confidence examples across slots
# ---------------------------------------------------------------------------


@pytest.fixture()
def shared_fact_store(tmp_path: Path) -> Store:
    """Store where the same fact appears in both a corroborate and a
    reconcilable edge, plus a cleaner alternative reconcilable edge.

    Layout
    ------
    fact_shared  – appears in e_corr (corroborate, conf=0.95)
                   AND e_rec_bad (reconcilable, conf=0.85)
    fact_corr_b  – paired with fact_shared in e_corr
    fact_rec_alt – paired with fact_rec_alt_b in e_rec_good (reconcilable, conf=0.75)
    fact_rec_alt_b
    fact_cont_a / fact_cont_b – contradict edge
    fact_low_conf – lowest confidence, used for failure slot
    """
    s = Store(tmp_path / "shared.db")
    s.init_schema()

    doc = Document(new_id("doc"), "x.pdf", "hx", 2, "done", "2026-01-01T00:00:00")
    s.upsert_document(doc)

    def _f(attr: str, value: str = "1", conf: float = 0.8) -> Fact:
        return Fact(
            new_id("fact"),
            doc.id,
            "Company",
            attr,
            value,
            "USD",
            "FY24",
            [],
            f"{attr} {value}",
            1,
            f"{attr} is {value}",
            "numerical",
            conf,
        )

    fact_shared = _f("revenue", "100", conf=0.9)
    fact_corr_b = _f("revenue", "100", conf=0.85)
    fact_rec_alt = _f("gdp", "500", conf=0.78)
    fact_rec_alt_b = _f("gdp", "510", conf=0.76)
    fact_cont_a = _f("profit", "50", conf=0.88)
    fact_cont_b = _f("profit", "200", conf=0.82)
    fact_low_conf = _f("misc", "0", conf=0.05)

    s.add_facts([
        fact_shared, fact_corr_b, fact_rec_alt, fact_rec_alt_b,
        fact_cont_a, fact_cont_b, fact_low_conf,
    ])

    # e_corr uses fact_shared (conf=0.95 — highest corroborate)
    e_corr = Edge(new_id("edge"), fact_shared.id, fact_corr_b.id, "corroborate",
                  "Same value", "none", 0.95)
    # e_rec_bad also uses fact_shared → should be skipped when filling reconcilable
    e_rec_bad = Edge(new_id("edge"), fact_shared.id, fact_rec_alt.id, "reconcilable",
                     "Slight diff", "time", 0.85)
    # e_rec_good uses fully distinct facts → should be preferred
    e_rec_good = Edge(new_id("edge"), fact_rec_alt.id, fact_rec_alt_b.id, "reconcilable",
                      "Slight diff", "scope", 0.75)
    e_cont = Edge(new_id("edge"), fact_cont_a.id, fact_cont_b.id, "contradict",
                  "Different values", "none", 0.88)

    s.add_edges([e_corr, e_rec_bad, e_rec_good, e_cont])

    # Expose IDs for assertions
    s._test_fact_shared_id = fact_shared.id  # type: ignore[attr-defined]
    s._test_e_rec_good_id = e_rec_good.id  # type: ignore[attr-defined]
    s._test_fact_low_conf_id = fact_low_conf.id  # type: ignore[attr-defined]

    return s


def test_build_cases_distinct_fact_ids(shared_fact_store: Store) -> None:
    """build_cases must NOT show the same fact_id in more than one relation slot."""
    from veritas.cases import build_cases

    result = build_cases(shared_fact_store)

    used_ids: list[str] = []
    for slot_name in ("corroborate", "contradict", "reconcilable"):
        slot = result[slot_name]
        if slot is not None:
            used_ids.append(slot["fact_a"]["id"])
            used_ids.append(slot["fact_b"]["id"])

    # No duplicates across the three relation slots
    assert len(used_ids) == len(set(used_ids)), (
        f"Fact ids are not distinct across slots: {used_ids}"
    )


def test_build_cases_picks_non_overlapping_reconcilable(shared_fact_store: Store) -> None:
    """When a fact is already used in the corroborate slot, build_cases must
    pick the non-overlapping reconcilable edge (e_rec_good)."""
    from veritas.cases import build_cases

    result = build_cases(shared_fact_store)

    rec = result["reconcilable"]
    assert rec is not None
    shared_id = shared_fact_store._test_fact_shared_id  # type: ignore[attr-defined]
    # The shared fact must NOT appear in the reconcilable slot
    assert rec["fact_a"]["id"] != shared_id
    assert rec["fact_b"]["id"] != shared_id


def test_build_cases_highest_confidence_per_type(tmp_path: Path) -> None:
    """build_cases must pick the highest-confidence edge for each relation type."""
    from veritas.cases import build_cases

    s = Store(tmp_path / "conf.db")
    s.init_schema()

    doc = Document(new_id("doc"), "d.pdf", "hd", 1, "done", "2026-01-01T00:00:00")
    s.upsert_document(doc)

    def _f(attr: str, value: str = "1", conf: float = 0.5) -> Fact:
        return Fact(
            new_id("fact"), doc.id, "Subj", attr, value, "", "", [],
            f"{attr} {value}", 1, f"{attr} is {value}", "numerical", conf,
        )

    # Three corroborate edges with different confidences; each uses distinct facts.
    fa1, fb1 = _f("a", "1", 0.9), _f("a", "1", 0.9)
    fa2, fb2 = _f("b", "2", 0.6), _f("b", "2", 0.6)
    fa3, fb3 = _f("c", "3", 0.3), _f("c", "3", 0.3)

    s.add_facts([fa1, fb1, fa2, fb2, fa3, fb3])

    e_low = Edge(new_id("edge"), fa3.id, fb3.id, "corroborate", "r", "none", 0.3)
    e_mid = Edge(new_id("edge"), fa2.id, fb2.id, "corroborate", "r", "none", 0.6)
    e_high = Edge(new_id("edge"), fa1.id, fb1.id, "corroborate", "r", "none", 0.9)

    s.add_edges([e_low, e_mid, e_high])

    result = build_cases(s)

    corr = result["corroborate"]
    assert corr is not None
    assert corr["edge"]["confidence"] == pytest.approx(0.9)


def test_build_cases_failure_excludes_relation_facts(shared_fact_store: Store) -> None:
    """The failure slot should not reuse facts already shown in relation slots
    when unused facts are available."""
    from veritas.cases import build_cases

    result = build_cases(shared_fact_store)

    failure = result["failure"]
    # The lowest-confidence overall fact is fact_low_conf — which is NOT used
    # in any relation slot, so it should appear in failure.
    assert failure is not None
    assert failure["id"] == shared_fact_store._test_fact_low_conf_id  # type: ignore[attr-defined]
