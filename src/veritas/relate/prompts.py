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

PRECISION RULE (apply this first, before any other definition):
corroborate, contradict, and reconcilable apply ONLY when Claim A and Claim B
describe the SAME attribute/measurement/quantity of the SAME subject (or the
same specific event or entity state).  If the two claims describe DIFFERENT
quantities, DIFFERENT attributes, or DIFFERENT subjects — even if they are
thematically or topically related — the relation MUST be "unrelated".

Definitions:
- corroborate: SAME subject AND SAME attribute/measurement; the reported
  values agree (even if worded or rounded differently), strengthening each
  other.
- contradict: SAME subject AND SAME attribute/measurement AND comparable
  context; the reported values are mutually incompatible and cannot both
  be true.
- reconcilable: SAME subject AND SAME attribute/measurement; the values
  appear to conflict but the conflict is fully explained by a difference
  in time period, scope/population, or unit of measurement — state which
  in reconciling_dimension.
- unrelated: anything else, including claims that are merely thematically or
  topically related but measure DIFFERENT quantities or describe DIFFERENT
  subjects.

Illustrative examples of the "loosely related → unrelated" rule:
  Example 1 — Claim A: "Indicator X rose by 0.3 units globally over the past
  decade."  Claim B: "Metric Y declined by 15% at location Z last winter."
  Although both claims are about the same broad phenomenon, they measure
  DIFFERENT quantities (X vs Y) on DIFFERENT scopes and time windows.
  → relation: "unrelated"

  Example 2 — Claim A: "The average rate of process P increased by 0.2/yr
  since 1980."  Claim B: "Event Q caused a temporary reversal of P at site S
  in 2010."  A describes a long-run global trend; B describes a short-term
  local event — different attribute, different scope.
  → relation: "unrelated"

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
