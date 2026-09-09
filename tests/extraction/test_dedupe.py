"""Tests for near-duplicate fact suppression (Fix 1)."""

from __future__ import annotations

from veritas.extraction.dedupe import _jaccard, _normalise_span, dedupe_facts
from veritas.models import Fact, new_id

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fact(
    subject: str,
    evidence_span: str,
    confidence: float = 0.9,
    doc_id: str = "doc_test",
) -> Fact:
    """Construct a minimal Fact for testing."""
    return Fact(
        id=new_id("fact"),
        doc_id=doc_id,
        subject=subject,
        attribute="attr",
        value="1",
        unit="",
        temporal_context="",
        scope_qualifiers=[],
        evidence_span=evidence_span,
        page=1,
        claim_text=f"{subject}: {evidence_span}",
        fact_kind="semantic",
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------


def test_jaccard_identical_sets() -> None:
    a = {"foo", "bar", "baz"}
    assert _jaccard(a, a) == 1.0


def test_jaccard_disjoint_sets() -> None:
    a = {"foo", "bar"}
    b = {"baz", "qux"}
    assert _jaccard(a, b) == 0.0


def test_jaccard_partial_overlap() -> None:
    a = {"a", "b", "c"}
    b = {"a", "b", "d"}
    # intersection=2, union=4 → 0.5
    assert abs(_jaccard(a, b) - 0.5) < 1e-9


def test_normalise_span_collapses_whitespace() -> None:
    assert _normalise_span("  Foo  BAR\t\nbaz  ") == "foo bar baz"


# ---------------------------------------------------------------------------
# dedupe_facts: same subject, overlapping spans → keep higher confidence
# ---------------------------------------------------------------------------


def test_dedupe_keeps_higher_confidence() -> None:
    """Two facts with heavily-overlapping evidence spans and same subject:
    only the higher-confidence one should survive.

    Spans chosen so their token-set Jaccard >= 0.8:
      intersection = 8 tokens shared, union = 9 → 8/9 ≈ 0.889
    """
    span_a = "near-term cooling at any particular location due to aerosol effects"
    span_b = "near-term cooling at any particular location due to aerosol effects and feedback"
    fa = _fact("climate", span_a, confidence=0.7)
    fb = _fact("climate", span_b, confidence=0.9)

    result = dedupe_facts([fa, fb])

    assert len(result) == 1
    assert result[0].id == fb.id  # higher confidence wins


def test_dedupe_keeps_first_on_tie() -> None:
    """When confidence is equal, the first fact (by insertion order) is kept.

    Spans chosen so their token-set Jaccard >= 0.8:
      span_b is span_a + one extra token → 8/9 ≈ 0.889
    """
    span_a = "near-term cooling at any particular location due to aerosol effects"
    span_b = "near-term cooling at any particular location due to aerosol effects revised"
    fa = _fact("climate", span_a, confidence=0.8)
    fb = _fact("climate", span_b, confidence=0.8)

    result = dedupe_facts([fa, fb])

    assert len(result) == 1
    assert result[0].id == fa.id  # first wins on tie


# ---------------------------------------------------------------------------
# dedupe_facts: different subjects → both kept
# ---------------------------------------------------------------------------


def test_dedupe_different_subjects_both_kept() -> None:
    """Two facts with the same span but different subjects must both be kept."""
    span = "near-term cooling at any particular location"
    fa = _fact("climate system", span, confidence=0.8)
    fb = _fact("local weather", span, confidence=0.9)

    result = dedupe_facts([fa, fb])

    assert len(result) == 2
    ids = {f.id for f in result}
    assert fa.id in ids
    assert fb.id in ids


# ---------------------------------------------------------------------------
# dedupe_facts: disjoint spans → both kept
# ---------------------------------------------------------------------------


def test_dedupe_disjoint_spans_both_kept() -> None:
    """Two facts with the same subject but completely different evidence spans
    must not be merged."""
    fa = _fact("climate", "global average temperature increased by 1.1 degrees", confidence=0.8)
    fb = _fact("climate", "sea level has risen by 20 centimetres since 1900", confidence=0.75)

    result = dedupe_facts([fa, fb])

    assert len(result) == 2


# ---------------------------------------------------------------------------
# dedupe_facts: substring containment → near-duplicate
# ---------------------------------------------------------------------------


def test_dedupe_substring_containment() -> None:
    """When one span is a substring of the other (same subject), keep the
    higher-confidence one."""
    short_span = "temperature rise of 1.5 degrees"
    long_span = "temperature rise of 1.5 degrees Celsius above pre-industrial levels"
    fa = _fact("global temperature", short_span, confidence=0.6)
    fb = _fact("global temperature", long_span, confidence=0.85)

    result = dedupe_facts([fa, fb])

    assert len(result) == 1
    assert result[0].id == fb.id


# ---------------------------------------------------------------------------
# dedupe_facts: empty list
# ---------------------------------------------------------------------------


def test_dedupe_empty_list() -> None:
    assert dedupe_facts([]) == []


# ---------------------------------------------------------------------------
# dedupe_facts: single fact
# ---------------------------------------------------------------------------


def test_dedupe_single_fact() -> None:
    fa = _fact("subject", "some evidence span")
    result = dedupe_facts([fa])
    assert len(result) == 1
    assert result[0].id == fa.id


# ---------------------------------------------------------------------------
# dedupe_facts: case / whitespace insensitivity
# ---------------------------------------------------------------------------


def test_dedupe_case_insensitive() -> None:
    """Span comparison is case-insensitive; subject matching is also case-insensitive."""
    fa = _fact("Climate System", "Near-Term Cooling At Any Particular Location", confidence=0.7)
    fb = _fact("climate system", "near-term cooling at any particular location", confidence=0.9)

    result = dedupe_facts([fa, fb])

    assert len(result) == 1
    assert result[0].id == fb.id


# ---------------------------------------------------------------------------
# dedupe_facts: preserves relative order of survivors
# ---------------------------------------------------------------------------


def test_dedupe_preserves_survivor_order() -> None:
    """Survivors must appear in their original insertion order."""
    fa = _fact("subjectA", "alpha beta gamma delta", confidence=0.5)
    fb = _fact("subjectB", "completely different text here", confidence=0.9)
    fc = _fact("subjectA", "epsilon zeta eta theta", confidence=0.4)

    result = dedupe_facts([fa, fb, fc])

    # fa and fb survive (different subjects or disjoint spans), fc is disjoint
    # from fa so all three survive; order must be fa, fb, fc.
    assert len(result) == 3
    assert result[0].id == fa.id
    assert result[1].id == fb.id
    assert result[2].id == fc.id
