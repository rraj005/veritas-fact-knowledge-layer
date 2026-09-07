"""Domain-neutral adjudication prompt builder for relationship classification."""

from __future__ import annotations

from veritas.models import Fact

_SYSTEM_PROMPT = """\
You are a precise analytical engine that compares pairs of factual claims and \
classifies their logical relationship.

Given two grounded claims (Claim A and Claim B), each with full context \
(value, unit, temporal context, scope qualifiers, normalized value, period \
dates, and verbatim evidence), output a single JSON object (no markdown, no \
prose outside the JSON) with exactly these fields:

  "relation": one of "corroborate" | "contradict" | "reconcilable" | "unrelated"
  "reasoning": a concise explanation (1-3 sentences) of why you chose that relation
  "reconciling_dimension": one of "time" | "scope" | "unit" | "none"
  "confidence": a float in [0.0, 1.0]

Definitions:
- corroborate: the claims assert compatible facts about the same thing, \
  strengthening each other.
- contradict: the claims make incompatible assertions about the same thing \
  that cannot both be true.
- reconcilable: the claims appear to conflict but the conflict is resolved by \
  a differing dimension (time window, scope/population, or unit of measurement).
- unrelated: the claims concern clearly different subjects or attributes; no \
  meaningful comparison is possible.

For "reconciling_dimension":
- "time": the difference is explained by different time periods.
- "scope": the difference is explained by different populations, geographies, \
  or aggregation levels.
- "unit": the difference is explained by different units of measurement.
- "none": not applicable (use for corroborate, contradict, and unrelated).

Output ONLY the JSON object. No other text.
"""


def build_adjudication_prompt(fact_a: Fact, fact_b: Fact) -> tuple[str, str]:
    """Return (system, user) prompt pair for adjudicating two factual claims.

    The prompt is domain-neutral: it never references domain-specific concepts
    and relies entirely on the structured fields of each Fact.
    """

    def _describe(label: str, f: Fact) -> str:
        lines = [
            f"--- {label} ---",
            f"Claim text     : {f.claim_text}",
            f"Subject        : {f.subject}",
            f"Attribute      : {f.attribute}",
            f"Reported value : {f.value} {f.unit}".strip(),
        ]
        if f.normalized_value is not None:
            lines.append(f"Normalized     : {f.normalized_value} {f.normalized_unit or ''}".strip())
        lines.append(f"Temporal ctx   : {f.temporal_context or '(none)'}")
        if f.period_start or f.period_end:
            lines.append(f"Period         : {f.period_start or '?'} → {f.period_end or '?'}")
        if f.scope_qualifiers:
            lines.append(f"Scope          : {', '.join(f.scope_qualifiers)}")
        lines.append(f'Evidence span  : "{f.evidence_span}"')
        return "\n".join(lines)

    user_msg = (
        "Compare the following two claims and classify their relationship.\n\n"
        + _describe("Claim A", fact_a)
        + "\n\n"
        + _describe("Claim B", fact_b)
        + "\n\nReturn your answer as a JSON object only."
    )

    return _SYSTEM_PROMPT, user_msg
