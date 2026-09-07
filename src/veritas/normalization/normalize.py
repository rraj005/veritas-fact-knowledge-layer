"""General normalization helpers for the Veritas fact knowledge layer.

Fills normalized_value, normalized_unit, period_start, period_end, and
canonical_subject on Fact objects without any domain-specific assumptions.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import string
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from veritas.models import Fact

logger = logging.getLogger(__name__)

# Module-level pint UnitRegistry singleton — created once and reused, so that
# repeated calls to canonical_unit do not pay the registry-construction cost.
try:
    import pint as _pint

    _UREG = _pint.UnitRegistry()
except Exception:  # noqa: BLE001
    _UREG = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Custom scale map (handles Indian and common financial/statistical units that
# are not in pint's default unit registry).
# Keys are lowercase strings that may appear as standalone words in a unit str.
# ---------------------------------------------------------------------------
_SCALE_MAP: dict[str, float] = {
    # Indian numeric units
    "crore": 1e7,
    "cr": 1e7,
    "lakh": 1e5,
    "lac": 1e5,
    # Common large-number words
    "million": 1e6,
    "mn": 1e6,
    "m": 1e6,  # Only matched when unit is EXACTLY "m" (ambiguous otherwise)
    "billion": 1e9,
    "bn": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
    "thousand": 1e3,
    "k": 1e3,
}

# Units whose "m" abbreviation we want to skip for scaling (physical length)
_SKIP_M_AS_MILLION = {"m", "metre", "meter", "metres", "meters"}

# ---------------------------------------------------------------------------
# parse_number
# ---------------------------------------------------------------------------


def parse_number(text: str) -> float | None:
    """Parse a numeric string that may contain commas or spaces as thousands
    separators.

    Returns None if the string cannot be parsed as a number.

    Examples::

        parse_number("8,142")  → 8142.0
        parse_number("1.2")    → 1.2
        parse_number("n/a")    → None
    """
    if not text or not text.strip():
        return None
    # Remove thousands-separator commas and spaces
    cleaned = text.strip().replace(",", "").replace(" ", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# parse_period
# ---------------------------------------------------------------------------

# Regex patterns for period parsing.
_Q_FY_RE = re.compile(r"^Q([1-4])FY(\d{2,4})$", re.IGNORECASE)
_FY_RE = re.compile(r"^FY(\d{2,4})$", re.IGNORECASE)
_CY_RE = re.compile(r"^CY(\d{4})$", re.IGNORECASE)
_YEAR_RE = re.compile(r"^(\d{4})$")
_AS_OF_RE = re.compile(r"^as\s+of\s+(.+)$", re.IGNORECASE)


def _fy_to_year(fy_str: str) -> int:
    """Convert a 2- or 4-digit FY year string to the 4-digit fiscal end year.

    "24" → 2024, "2024" → 2024.
    """
    n = int(fy_str)
    if n < 100:
        # Assume 2000s; "24" → 2024
        n += 2000
    return n


def _quarter_dates(quarter: int, fy_end_year: int) -> tuple[str, str]:
    """Return (start, end) ISO date strings for the given Indian FY quarter.

    Indian FY runs Apr–Mar:
      Q1: Apr–Jun   (FY start year)
      Q2: Jul–Sep   (FY start year)
      Q3: Oct–Dec   (FY start year)
      Q4: Jan–Mar   (FY end year)
    """
    fy_start_year = fy_end_year - 1
    quarters = {
        1: (f"{fy_start_year}-04-01", f"{fy_start_year}-06-30"),
        2: (f"{fy_start_year}-07-01", f"{fy_start_year}-09-30"),
        3: (f"{fy_start_year}-10-01", f"{fy_start_year}-12-31"),
        4: (f"{fy_end_year}-01-01", f"{fy_end_year}-03-31"),
    }
    return quarters[quarter]


def parse_period(text: str) -> tuple[str | None, str | None]:
    """Parse a temporal context string into an (ISO start, ISO end) pair.

    Recognised patterns (case-insensitive):
    - ``FY24`` / ``FY2024`` — Indian fiscal year (Apr–Mar); FY24 → 2023-04-01..2024-03-31
    - ``Q4FY24`` — quarter within an Indian FY
    - ``CY2024`` — calendar year
    - ``2024``   — bare year treated as CY
    - ``as of YYYY-MM-DD`` — point-in-time date

    Returns (None, None) for unrecognised patterns.

    Args:
        text: A temporal context string from a Fact.

    Returns:
        A tuple (period_start, period_end) of ISO-8601 date strings, or
        (None, None) if the pattern is not recognised.
    """
    if not text or not text.strip():
        return None, None

    stripped = text.strip()

    # Q<n>FY<yy>
    m = _Q_FY_RE.match(stripped)
    if m:
        quarter = int(m.group(1))
        fy_end = _fy_to_year(m.group(2))
        return _quarter_dates(quarter, fy_end)

    # FY<yy>
    m = _FY_RE.match(stripped)
    if m:
        fy_end = _fy_to_year(m.group(1))
        fy_start = fy_end - 1
        return f"{fy_start}-04-01", f"{fy_end}-03-31"

    # CY<yyyy>
    m = _CY_RE.match(stripped)
    if m:
        year = int(m.group(1))
        return f"{year}-01-01", f"{year}-12-31"

    # Bare 4-digit year
    m = _YEAR_RE.match(stripped)
    if m:
        year = int(m.group(1))
        return f"{year}-01-01", f"{year}-12-31"

    # "as of <date>"
    m = _AS_OF_RE.match(stripped)
    if m:
        date_str = m.group(1).strip()
        # Attempt to parse ISO date; if it looks valid, use it as a point period.
        iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        if iso_re.match(date_str):
            return date_str, date_str
        # Try common formats
        for fmt, pat in [
            (r"(\d{2})/(\d{2})/(\d{4})", "{2}-{0}-{1}"),
            (r"(\d{4})/(\d{2})/(\d{2})", "{0}-{1}-{2}"),
        ]:
            fm = re.match(fmt, date_str)
            if fm:
                # Build ISO date from named groups
                parts = [fm.group(i + 1) for i in range(len(fm.groups()))]
                try:
                    iso = pat.format(*parts)
                    return iso, iso
                except (IndexError, ValueError):
                    pass

    return None, None


# ---------------------------------------------------------------------------
# canonical_unit
# ---------------------------------------------------------------------------


def canonical_unit(unit_str: str) -> str:
    """Return a canonical string for the given unit.

    For units recognised by pint (kg, %, m, etc.) we return the pint
    canonical name (or the input if pint cannot parse it). For custom
    scale words (crore, lakh, million, etc.) we return the scale word
    itself (lower-case), since they represent multiplicative modifiers
    rather than true units.

    Args:
        unit_str: Raw unit string from an LLM-extracted fact.

    Returns:
        A canonical unit string (never empty; falls back to the original).
    """
    if not unit_str or not unit_str.strip():
        return ""

    stripped = unit_str.strip()

    # Check for custom scale words first (they are not pint units).
    low = stripped.lower()
    for word in _SCALE_MAP:
        if low == word:
            return word

    # Keep percentage symbol as-is — widely understood and no conversion needed.
    if stripped in ("%", "percent", "percentage"):
        return "%"

    # Try pint for physical / standard units (use module-level singleton _UREG).
    if _UREG is not None:
        try:
            parsed = _UREG.parse_expression(stripped)
            # Return pint's canonical unit string.
            return str(parsed.units)
        except Exception:  # noqa: BLE001, S110 — unknown unit: fall through to pass-through
            pass
    # Unknown unit — pass through unchanged.
    return stripped


# ---------------------------------------------------------------------------
# _resolve_scale
# ---------------------------------------------------------------------------


def _resolve_scale(unit_str: str) -> tuple[float, str]:
    """Return (multiplier, remainder_unit) for a unit string.

    Searches for any custom scale word as a whole token within the unit
    string. Scale detection is case-insensitive, but the remainder unit is
    built from the ORIGINAL-case tokens (e.g. "INR" stays "INR", not "inr").
    If found, returns (scale, unit_without_scale_word). Otherwise returns
    (1.0, unit_str).

    Examples::

        _resolve_scale("INR Cr")       → (1e7, "INR")
        _resolve_scale("million USD")  → (1e6, "USD")
        _resolve_scale("kg")           → (1.0, "kg")
    """
    stripped = unit_str.strip()
    # Split while preserving original-case tokens for the remainder.
    original_tokens = re.split(r"[\s/]+", stripped)
    # Lowercase copies for scale detection only.
    lower_tokens = [t.lower() for t in original_tokens]

    for i, lower_token in enumerate(lower_tokens):
        if lower_token in _SCALE_MAP:
            # "m" is ambiguous — only treat as million when the FULL unit is "m"
            # or when there are other tokens confirming it's a scale word.
            if lower_token == "m" and len(lower_tokens) == 1:
                # Single token "m" — treat as metre, not million.
                break
            scale = _SCALE_MAP[lower_token]
            # Rebuild remaining unit tokens from ORIGINAL-case tokens (excluding
            # the scale word), so that e.g. "INR" stays "INR".
            remaining_tokens = [t for j, t in enumerate(original_tokens) if j != i]
            remainder = " ".join(remaining_tokens).strip()
            return scale, remainder

    return 1.0, stripped


# ---------------------------------------------------------------------------
# canonical_subject
# ---------------------------------------------------------------------------


def _canonical_subject(subject: str) -> str:
    """Lowercase, trim, and strip punctuation from a subject string."""
    lowered = subject.strip().lower()
    # Remove punctuation except hyphens within words (they may be meaningful)
    cleaned = lowered.translate(str.maketrans("", "", string.punctuation.replace("-", "")))
    # Collapse multiple spaces left after punctuation removal
    collapsed = re.sub(r"\s+", " ", cleaned).strip()
    # Strip any leading/trailing hyphens
    return collapsed.strip("-")


# ---------------------------------------------------------------------------
# normalize_fact
# ---------------------------------------------------------------------------


def normalize_fact(fact: Fact) -> Fact:
    """Return a new Fact with normalized_value, normalized_unit, period_start,
    period_end, and canonical_subject filled in.

    Normalization logic:
    - ``normalized_value``: parse the numeric value; if the unit contains a
      custom scale word (crore, lakh, million, etc.) multiply accordingly;
      for pint-recognisable physical units keep the raw number.
    - ``normalized_unit``: the remainder unit after stripping the scale word,
      or the pint canonical for physical units, or the original unit if
      unknown.
    - ``period_start`` / ``period_end``: parsed from ``temporal_context``.
    - ``canonical_subject``: lowercased, trimmed, punctuation-stripped.

    Unknown / unparseable values are left as None (for numeric) or passed
    through (for unit) — no data is silently invented.

    Args:
        fact: An extracted Fact object.

    Returns:
        A new Fact with the normalized_* fields populated.
    """
    # --- numeric value + scale ---
    raw_number = parse_number(fact.value)
    scale, remainder_unit = _resolve_scale(fact.unit)

    normalized_value: float | None = None
    if raw_number is not None:
        normalized_value = raw_number * scale

    # --- normalized_unit ---
    normalized_unit: str | None = None
    if fact.unit:
        if scale != 1.0:
            # The unit had a scale word; remainder is the "real" unit.
            normalized_unit = remainder_unit if remainder_unit else fact.unit
        else:
            # No custom scale — try pint canonical.
            normalized_unit = canonical_unit(fact.unit)

    # --- period ---
    period_start, period_end = parse_period(fact.temporal_context)

    # --- canonical subject ---
    canon_subject = _canonical_subject(fact.subject)

    return dataclasses.replace(
        fact,
        normalized_value=normalized_value,
        normalized_unit=normalized_unit,
        period_start=period_start,
        period_end=period_end,
        canonical_subject=canon_subject,
    )
