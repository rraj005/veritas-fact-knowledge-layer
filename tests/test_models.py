from veritas.models import Fact, new_id


def test_fact_roundtrip():
    f = Fact(
        id=new_id("fact"),
        doc_id="d1",
        subject="Company",
        attribute="revenue",
        value="8142",
        unit="INR Cr",
        temporal_context="FY24",
        scope_qualifiers=["consolidated"],
        evidence_span="Revenue was Rs 8,142 Cr",
        page=12,
        claim_text="Revenue FY24 was 8142 INR Cr",
        fact_kind="numerical",
        confidence=0.9,
    )
    assert Fact.from_dict(f.to_dict()) == f


def test_new_id_prefix():
    assert new_id("edge").startswith("edge_")
