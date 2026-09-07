"""Tests for Task 9: normalization of facts (numeric/date/unit/entity)."""

from __future__ import annotations

import pytest

from veritas.models import Fact


def test_number_and_period() -> None:
    from veritas.normalization.normalize import parse_number, parse_period

    assert parse_number("8,142") == 8142.0
    assert parse_number("1.2") == 1.2
    assert parse_period("FY24") == ("2023-04-01", "2024-03-31")
    assert parse_period("Q4FY24") == ("2024-01-01", "2024-03-31")


def test_normalize_crore() -> None:
    from veritas.normalization.normalize import normalize_fact

    f = Fact(
        "i",
        "d",
        "Co",
        "revenue",
        "8,142",
        "INR Cr",
        "FY24",
        [],
        "ev",
        1,
        "c",
        "numerical",
        0.9,
    )
    out = normalize_fact(f)
    assert out.normalized_value == 8142 * 10**7  # crore -> absolute
    assert out.period_start == "2023-04-01"
    # Currency code case must be preserved — "INR Cr" → normalized_unit "INR"
    assert out.normalized_unit == "INR"


# ---------------------------------------------------------------------------
# Additional tests (Q4FY24, physical units, percentage, canonical_subject)
# ---------------------------------------------------------------------------


def test_q4fy24_period() -> None:
    from veritas.normalization.normalize import parse_period

    start, end = parse_period("Q4FY24")
    assert start == "2024-01-01"
    assert end == "2024-03-31"


def test_calendar_year_period() -> None:
    from veritas.normalization.normalize import parse_period

    assert parse_period("CY2024") == ("2024-01-01", "2024-12-31")
    assert parse_period("2024") == ("2024-01-01", "2024-12-31")


def test_unknown_period() -> None:
    from veritas.normalization.normalize import parse_period

    assert parse_period("sometime last year") == (None, None)


def test_physical_unit_kg() -> None:
    """A fact with unit 'kg' should normalize via pint (or pass-through number)."""
    from veritas.normalization.normalize import normalize_fact

    f = Fact(
        "i",
        "d",
        "Shipment",
        "weight",
        "500",
        "kg",
        "",
        [],
        "Weight was 500 kg",
        1,
        "Weight was 500 kg",
        "numerical",
        0.9,
    )
    out = normalize_fact(f)
    assert out.normalized_value == 500.0
    # pint returns "kilogram" for "kg"; accept either abbreviated or full form.
    assert out.normalized_unit in ("kilogram", "kg")


def test_percentage() -> None:
    """A fact with unit '%' should have normalized_value = parsed number."""
    from veritas.normalization.normalize import normalize_fact

    f = Fact(
        "i",
        "d",
        "Economy",
        "growth",
        "7.5",
        "%",
        "2024",
        [],
        "Growth was 7.5%",
        1,
        "Growth was 7.5%",
        "numerical",
        0.9,
    )
    out = normalize_fact(f)
    assert out.normalized_value == 7.5
    assert out.normalized_unit == "%"


def test_canonical_subject_stripping() -> None:
    """canonical_subject should be lowercase, trimmed, punctuation-stripped."""
    from veritas.normalization.normalize import normalize_fact

    f = Fact(
        "i",
        "d",
        "  Acme Corp.  ",
        "revenue",
        "100",
        "INR Cr",
        "",
        [],
        "Acme Corp. revenue 100 Cr",
        1,
        "Acme revenue 100 Cr",
        "numerical",
        0.8,
    )
    out = normalize_fact(f)
    assert out.canonical_subject == "acme corp"


def test_lakh_unit() -> None:
    from veritas.normalization.normalize import normalize_fact

    f = Fact(
        "i",
        "d",
        "Company",
        "employees",
        "5",
        "lakh",
        "",
        [],
        "5 lakh employees",
        1,
        "5 lakh employees",
        "numerical",
        0.9,
    )
    out = normalize_fact(f)
    assert out.normalized_value == 5 * 10**5


def test_million_unit() -> None:
    from veritas.normalization.normalize import normalize_fact

    f = Fact(
        "i",
        "d",
        "Company",
        "revenue",
        "2.5",
        "million USD",
        "",
        [],
        "2.5 million USD revenue",
        1,
        "2.5 million USD revenue",
        "numerical",
        0.9,
    )
    out = normalize_fact(f)
    assert out.normalized_value == pytest.approx(2.5 * 10**6)


def test_parse_number_unparseable() -> None:
    from veritas.normalization.normalize import parse_number

    assert parse_number("n/a") is None
    assert parse_number("") is None
    assert parse_number("abc") is None


def test_as_of_date_period() -> None:
    from veritas.normalization.normalize import parse_period

    start, end = parse_period("as of 2024-03-31")
    assert start == "2024-03-31"
    assert end == "2024-03-31"


# ---------------------------------------------------------------------------
# Q1/Q2/Q3 quarter date tests (Finding 4 — vacuous test gap)
# ---------------------------------------------------------------------------


def test_q1fy24_period() -> None:
    """Q1FY24 → Indian FY Q1 Apr-Jun of the FY start year (2023)."""
    from veritas.normalization.normalize import _quarter_dates, parse_period

    # Direct function test
    assert _quarter_dates(1, 2024) == ("2023-04-01", "2023-06-30")
    # Via parse_period
    start, end = parse_period("Q1FY24")
    assert start == "2023-04-01"
    assert end == "2023-06-30"


def test_q2fy24_period() -> None:
    """Q2FY24 → Indian FY Q2 Jul-Sep of the FY start year (2023)."""
    from veritas.normalization.normalize import _quarter_dates, parse_period

    assert _quarter_dates(2, 2024) == ("2023-07-01", "2023-09-30")
    start, end = parse_period("Q2FY24")
    assert start == "2023-07-01"
    assert end == "2023-09-30"


def test_q3fy24_period() -> None:
    """Q3FY24 → Indian FY Q3 Oct-Dec of the FY start year (2023)."""
    from veritas.normalization.normalize import _quarter_dates, parse_period

    assert _quarter_dates(3, 2024) == ("2023-10-01", "2023-12-31")
    start, end = parse_period("Q3FY24")
    assert start == "2023-10-01"
    assert end == "2023-12-31"
