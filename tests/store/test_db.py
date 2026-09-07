"""Tests for SQLite Store (Task 7)."""

from __future__ import annotations

import pytest

from veritas.models import Document, Edge, Fact, Job, new_id
from veritas.store.db import Store

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc(filename: str = "a.pdf", content_hash: str = "hash123") -> Document:
    return Document(new_id("doc"), filename, content_hash, 10, "done", "2026-01-01")


def _fact(
    doc: str = "d1", attr: str = "revenue", confidence: float = 0.9, kind: str = "numerical"
) -> Fact:
    return Fact(
        new_id("fact"),
        doc,
        "Co",
        attr,
        "100",
        "INR Cr",
        "FY24",
        [],
        "Revenue 100 Cr",
        3,
        "Revenue FY24 100 INR Cr",
        kind,
        confidence,
    )


def _edge(fact_a_id: str, fact_b_id: str, relation: str = "corroborate") -> Edge:
    return Edge(
        new_id("edge"),
        fact_a_id,
        fact_b_id,
        relation,
        "Both report the same figure",
        "none",
        0.9,
    )


def _job(doc_id: str = "d1") -> Job:
    return Job(new_id("job"), doc_id, "queued", 0.0, "", "2026-01-01")


# ---------------------------------------------------------------------------
# Provided tests (must pass unchanged)
# ---------------------------------------------------------------------------


def test_idempotent_document(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    d = Document(new_id("doc"), "a.pdf", "hash123", 10, "done", "2026-01-01")
    s.upsert_document(d)
    assert s.get_document_by_hash("hash123").filename == "a.pdf"


def test_facts_and_vocab(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    s.add_facts([_fact(attr="revenue"), _fact(attr="profit")])
    s.record_attributes(["revenue", "profit", "revenue"])
    assert len(s.list_facts()) == 2
    vocab = dict(s.attribute_vocab())
    assert vocab["revenue"] >= 1


# ---------------------------------------------------------------------------
# Additional tests
# ---------------------------------------------------------------------------


def test_upsert_document_idempotent(tmp_path):
    """Upserting the same doc twice does not create duplicates."""
    s = Store(tmp_path / "t.db")
    s.init_schema()
    d = _doc()
    s.upsert_document(d)
    s.upsert_document(d)  # second upsert
    docs = s.list_documents()
    assert len(docs) == 1
    assert docs[0].id == d.id


def test_list_documents(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    s.upsert_document(_doc("a.pdf", "h1"))
    s.upsert_document(_doc("b.pdf", "h2"))
    docs = s.list_documents()
    assert len(docs) == 2
    filenames = {d.filename for d in docs}
    assert filenames == {"a.pdf", "b.pdf"}


def test_get_document_by_hash_missing(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    assert s.get_document_by_hash("nonexistent") is None


def test_get_fact(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    f = _fact()
    s.add_facts([f])
    retrieved = s.get_fact(f.id)
    assert retrieved is not None
    assert retrieved.id == f.id
    assert retrieved.attribute == f.attribute


def test_get_fact_missing(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    assert s.get_fact("nonexistent") is None


def test_all_facts(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    s.add_facts([_fact(), _fact(), _fact()])
    assert len(s.all_facts()) == 3


def test_list_facts_filter_kind(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    s.add_facts([_fact(kind="numerical"), _fact(kind="semantic")])
    numerical = s.list_facts(kind="numerical")
    assert len(numerical) == 1
    assert numerical[0].fact_kind == "numerical"


def test_list_facts_filter_min_conf(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    s.add_facts([_fact(confidence=0.9), _fact(confidence=0.3)])
    high_conf = s.list_facts(min_conf=0.8)
    assert len(high_conf) == 1
    assert high_conf[0].confidence >= 0.8


def test_list_facts_filter_doc_id(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    s.add_facts([_fact(doc="doc_a"), _fact(doc="doc_b")])
    facts_a = s.list_facts(doc_id="doc_a")
    assert len(facts_a) == 1
    assert facts_a[0].doc_id == "doc_a"


def test_list_facts_filter_subject(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    f1 = Fact(
        new_id("fact"),
        "d1",
        "Apple",
        "revenue",
        "100",
        "USD",
        "FY24",
        [],
        "Apple revenue 100",
        1,
        "Apple revenue FY24 100",
        "numerical",
        0.9,
    )
    f2 = Fact(
        new_id("fact"),
        "d1",
        "Google",
        "revenue",
        "200",
        "USD",
        "FY24",
        [],
        "Google revenue 200",
        2,
        "Google revenue FY24 200",
        "numerical",
        0.9,
    )
    s.add_facts([f1, f2])
    results = s.list_facts(subject="Apple")
    assert len(results) == 1
    assert results[0].subject == "Apple"


def test_list_facts_limit_offset(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    for _ in range(5):
        s.add_facts([_fact()])
    page1 = s.list_facts(limit=2, offset=0)
    page2 = s.list_facts(limit=2, offset=2)
    assert len(page1) == 2
    assert len(page2) == 2
    # ids should be different across pages
    assert {f.id for f in page1}.isdisjoint({f.id for f in page2})


def test_fact_scope_qualifiers_roundtrip(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    f = Fact(
        new_id("fact"),
        "d1",
        "Co",
        "revenue",
        "100",
        "INR Cr",
        "FY24",
        ["consolidated", "audited"],
        "Revenue 100 Cr",
        3,
        "Revenue FY24 100 INR Cr",
        "numerical",
        0.9,
    )
    s.add_facts([f])
    out = s.get_fact(f.id)
    assert out.scope_qualifiers == ["consolidated", "audited"]


# ---------------------------------------------------------------------------
# Edge tests
# ---------------------------------------------------------------------------


def test_edge_roundtrip(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    fa = _fact(doc="d1")
    fb = _fact(doc="d2")
    s.add_facts([fa, fb])
    e = _edge(fa.id, fb.id, "contradict")
    s.add_edges([e])
    retrieved = s.get_edge(e.id)
    assert retrieved is not None
    assert retrieved.relation == "contradict"
    assert retrieved.fact_a_id == fa.id


def test_list_edges_relation_filter(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    fa = _fact(doc="d1")
    fb = _fact(doc="d2")
    fc = _fact(doc="d3")
    s.add_facts([fa, fb, fc])
    e1 = _edge(fa.id, fb.id, "corroborate")
    e2 = _edge(fb.id, fc.id, "contradict")
    s.add_edges([e1, e2])
    corroborations = s.list_edges(relation="corroborate")
    assert len(corroborations) == 1
    assert corroborations[0].relation == "corroborate"


def test_list_edges_doc_id_filter(tmp_path):
    """list_edges(doc_id=...) filters edges where either endpoint belongs to that doc."""
    s = Store(tmp_path / "t.db")
    s.init_schema()
    # fa and fb are the same doc "d1"; fc is "d2"; fd is "d3"
    fa = _fact(doc="d1")
    fb = _fact(doc="d1")
    fc = _fact(doc="d2")
    fd = _fact(doc="d3")
    s.add_facts([fa, fb, fc, fd])
    # e1: d1↔d2 (should appear for d1 and d2)
    e1 = _edge(fa.id, fc.id, "corroborate")
    # e2: d2↔d3 (should NOT appear for d1)
    e2 = _edge(fc.id, fd.id, "contradict")
    s.add_edges([e1, e2])
    edges_d1 = s.list_edges(doc_id="d1")
    assert len(edges_d1) == 1
    assert edges_d1[0].id == e1.id


def test_get_edge_missing(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    assert s.get_edge("nonexistent") is None


def test_add_edges_idempotent(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    fa = _fact(doc="d1")
    fb = _fact(doc="d2")
    s.add_facts([fa, fb])
    e = _edge(fa.id, fb.id)
    s.add_edges([e])
    s.add_edges([e])  # second insert
    assert len(s.list_edges()) == 1


# ---------------------------------------------------------------------------
# Job tests
# ---------------------------------------------------------------------------


def test_job_create_get(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    j = _job()
    s.create_job(j)
    retrieved = s.get_job(j.id)
    assert retrieved is not None
    assert retrieved.id == j.id
    assert retrieved.status == "queued"
    assert retrieved.progress == 0.0


def test_job_update(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    j = _job()
    s.create_job(j)
    s.update_job(j.id, status="extracting", progress=0.5, message="halfway")
    updated = s.get_job(j.id)
    assert updated.status == "extracting"
    assert updated.progress == 0.5
    assert updated.message == "halfway"


def test_get_job_missing(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    assert s.get_job("nonexistent") is None


# ---------------------------------------------------------------------------
# Vocab tests
# ---------------------------------------------------------------------------


def test_attribute_vocab_accumulates(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    s.record_attributes(["revenue", "profit"])
    s.record_attributes(["revenue"])
    vocab = dict(s.attribute_vocab())
    assert vocab["revenue"] == 2
    assert vocab["profit"] == 1


def test_attribute_vocab_empty(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    assert s.attribute_vocab() == []


# ---------------------------------------------------------------------------
# Security: update_job whitelist tests
# ---------------------------------------------------------------------------


def test_update_job_rejects_unknown_field(tmp_path):
    """update_job must raise ValueError for any column name not in the whitelist."""
    s = Store(tmp_path / "t.db")
    s.init_schema()
    j = _job()
    s.create_job(j)
    with pytest.raises(ValueError, match="Unknown job fields"):
        s.update_job(j.id, **{"status = 1; DROP TABLE facts; --": "x"})


def test_update_job_injection_leaves_facts_intact(tmp_path):
    """After a rejected update_job call, the facts table must still exist and be intact."""
    s = Store(tmp_path / "t.db")
    s.init_schema()
    j = _job()
    s.create_job(j)
    # Insert a fact so we can verify the table survives.
    f = _fact()
    s.add_facts([f])
    with pytest.raises(ValueError):
        s.update_job(j.id, **{"status = 1; DROP TABLE facts; --": "x"})
    # facts table must still exist and contain our row.
    surviving = s.all_facts()
    assert len(surviving) == 1
    assert surviving[0].id == f.id


def test_update_job_legitimate_fields(tmp_path):
    """All three whitelisted fields (status, progress, message) still update correctly."""
    s = Store(tmp_path / "t.db")
    s.init_schema()
    j = _job()
    s.create_job(j)
    s.update_job(j.id, status="running", progress=0.75, message="in progress")
    updated = s.get_job(j.id)
    assert updated.status == "running"
    assert updated.progress == 0.75
    assert updated.message == "in progress"


# ---------------------------------------------------------------------------
# counts() test
# ---------------------------------------------------------------------------


def test_counts(tmp_path):
    s = Store(tmp_path / "t.db")
    s.init_schema()
    s.upsert_document(_doc("a.pdf", "h1"))
    s.upsert_document(_doc("b.pdf", "h2"))
    fa = _fact(doc="d1")
    fb = _fact(doc="d2")
    s.add_facts([fa, fb])
    e = _edge(fa.id, fb.id)
    s.add_edges([e])
    c = s.counts()
    assert c["docs"] == 2
    assert c["facts"] == 2
    assert c["edges"] == 1
