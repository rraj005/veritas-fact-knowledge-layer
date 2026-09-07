"""Four-cases builder and knowledge-layer export utilities (Task 15)."""

from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from veritas.store.db import Store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fact_detail(fact) -> dict:
    """Return a serialisable dict for a Fact with the fields the UI needs."""
    return {
        "id": fact.id,
        "doc_id": fact.doc_id,
        "subject": fact.subject,
        "attribute": fact.attribute,
        "value": fact.value,
        "unit": fact.unit,
        "temporal_context": fact.temporal_context,
        "scope_qualifiers": fact.scope_qualifiers,
        "evidence_span": fact.evidence_span,
        "page": fact.page,
        "claim_text": fact.claim_text,
        "fact_kind": fact.fact_kind,
        "confidence": fact.confidence,
        "normalized_value": fact.normalized_value,
        "normalized_unit": fact.normalized_unit,
        "period_start": fact.period_start,
        "period_end": fact.period_end,
        "canonical_subject": fact.canonical_subject,
    }


def _edge_detail(edge) -> dict:
    """Return a serialisable dict for an Edge."""
    return {
        "id": edge.id,
        "fact_a_id": edge.fact_a_id,
        "fact_b_id": edge.fact_b_id,
        "relation": edge.relation,
        "reasoning": edge.reasoning,
        "reconciling_dimension": edge.reconciling_dimension,
        "confidence": edge.confidence,
    }


def _build_case_slot(edge, store: Store) -> dict | None:
    """Build a single case slot from an edge record.

    Returns None if either endpoint fact cannot be resolved.
    """
    fact_a = store.get_fact(edge.fact_a_id)
    fact_b = store.get_fact(edge.fact_b_id)
    if fact_a is None or fact_b is None:
        return None
    return {
        "edge": _edge_detail(edge),
        "fact_a": _fact_detail(fact_a),
        "fact_b": _fact_detail(fact_b),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_cases(store: Store) -> dict:
    """Return up to one example each of corroborate, contradict, reconcilable,
    plus the lowest-confidence fact as the *failure* slot.

    Structure::

        {
            "corroborate":  { "edge": {...}, "fact_a": {...}, "fact_b": {...} } | None,
            "contradict":   { "edge": {...}, "fact_a": {...}, "fact_b": {...} } | None,
            "reconcilable": { "edge": {...}, "fact_a": {...}, "fact_b": {...} } | None,
            "failure":      { <fact fields> } | None,
        }
    """
    result: dict = {
        "corroborate": None,
        "contradict": None,
        "reconcilable": None,
        "failure": None,
    }

    for relation in ("corroborate", "contradict", "reconcilable"):
        edges = store.list_edges(relation=relation)
        for edge in edges:
            slot = _build_case_slot(edge, store)
            if slot is not None:
                result[relation] = slot
                break  # take first valid example

    # Failure slot — lowest-confidence fact (or None if no facts)
    all_facts = store.all_facts()
    if all_facts:
        lowest = min(all_facts, key=lambda f: f.confidence)
        result["failure"] = _fact_detail(lowest)

    return result


def export_layer(store: Store) -> dict:
    """Return the entire knowledge layer as a JSON-serialisable dict.

    Structure::

        {
            "facts": [ {...}, ... ],
            "edges": [ {...}, ... ],
        }
    """
    facts = [_fact_detail(f) for f in store.all_facts()]
    edges = [_edge_detail(e) for e in store.list_edges()]
    return {"facts": facts, "edges": edges}


def export_csv(store: Store, kind: str = "facts") -> str:
    """Serialise the knowledge layer to CSV.

    Args:
        store: The SQLite store to export from.
        kind:  ``"facts"`` (default) or ``"edges"``.

    Returns:
        A CSV string with a header row and one data row per record.
    """
    buf = io.StringIO()

    if kind == "edges":
        edges = store.list_edges()
        if not edges:
            # Return an empty CSV with headers only.
            writer = csv.DictWriter(
                buf,
                fieldnames=[
                    "id",
                    "fact_a_id",
                    "fact_b_id",
                    "relation",
                    "reasoning",
                    "reconciling_dimension",
                    "confidence",
                ],
            )
            writer.writeheader()
            return buf.getvalue()

        fieldnames = list(_edge_detail(edges[0]).keys())
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for edge in edges:
            writer.writerow(_edge_detail(edge))
    else:
        # Default: facts
        facts = store.all_facts()
        if not facts:
            writer = csv.DictWriter(
                buf,
                fieldnames=[
                    "id",
                    "doc_id",
                    "subject",
                    "attribute",
                    "value",
                    "unit",
                    "temporal_context",
                    "scope_qualifiers",
                    "evidence_span",
                    "page",
                    "claim_text",
                    "fact_kind",
                    "confidence",
                    "normalized_value",
                    "normalized_unit",
                    "period_start",
                    "period_end",
                    "canonical_subject",
                ],
            )
            writer.writeheader()
            return buf.getvalue()

        fieldnames = list(_fact_detail(facts[0]).keys())
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for fact in facts:
            row = _fact_detail(fact)
            # Flatten list fields for CSV
            row["scope_qualifiers"] = "|".join(row["scope_qualifiers"])
            writer.writerow(row)

    return buf.getvalue()
