"""Evaluation harness for the Veritas fact knowledge layer.

``evaluate(store, gold) -> dict``
    Pure function — takes a Store and a gold dict; returns a metrics dict.
    No LLM, no real PDFs needed; fully unit-testable with synthetic data.

``format_report(metrics) -> str``
    Formats the metrics dict as a human-readable text report.

Usage::

    import json
    from pathlib import Path
    from veritas.store.db import Store
    from eval.evaluate import evaluate, format_report

    store = Store("data/demo/veritas.db")
    store.init_schema()
    gold = json.loads(Path("eval/gold.json").read_text())
    metrics = evaluate(store, gold)
    print(format_report(metrics))
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from veritas.store.db import Store


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_RECONCILABLE_OR_CORROBORATE_RELATIONS = {"reconcilable", "corroborate"}


def _matches_gold_relation(edge_relation: str, gold_relation: str) -> bool:
    """Return True if *edge_relation* satisfies *gold_relation*.

    ``"reconcilable_or_corroborate"`` is satisfied by either ``"reconcilable"``
    or ``"corroborate"``; any other gold label requires an exact match.
    """
    if gold_relation == "reconcilable_or_corroborate":
        return edge_relation in _RECONCILABLE_OR_CORROBORATE_RELATIONS
    return edge_relation == gold_relation


def _fact_ids_containing(facts: list, substring: str) -> list[str]:
    """Return ids of facts whose claim_text contains *substring* (case-insensitive)."""
    sub_lower = substring.lower()
    return [f.id for f in facts if sub_lower in f.claim_text.lower()]


def _has_matching_edge(
    a_ids: list[str],
    b_ids: list[str],
    edges: list,
    gold_relation: str,
) -> bool:
    """Return True if any edge between any (a_id, b_id) pair satisfies gold_relation.

    Collects ALL relations per pair (a frozenset -> list[str]) so that when
    multiple edges exist between the same pair, no relation is silently dropped.
    """
    # Build a map: frozenset({fact_a_id, fact_b_id}) -> list of relations.
    edge_pairs: dict[frozenset, list[str]] = {}
    for e in edges:
        key = frozenset({e.fact_a_id, e.fact_b_id})
        edge_pairs.setdefault(key, []).append(e.relation)

    for a_id in a_ids:
        for b_id in b_ids:
            if a_id == b_id:
                continue
            key = frozenset({a_id, b_id})
            for relation in edge_pairs.get(key, []):
                if _matches_gold_relation(relation, gold_relation):
                    return True
    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(store: Store, gold: dict) -> dict:
    """Compute extraction and relationship metrics.

    Args:
        store: A ``veritas.store.db.Store`` (may be empty).
        gold:  A dict with key ``"expected_relations": list[dict]``.  Each item
               must have ``"a_contains"``, ``"b_contains"``, and ``"relation"``.

    Returns:
        A dict with the following keys:

        Extraction sanity:
          - ``total_facts``:               int
          - ``facts_with_evidence_span``:  int   (non-empty evidence_span)
          - ``pct_with_evidence_span``:    float (0–100)
          - ``facts_with_normalized_value``: int
          - ``pct_with_normalized_value``:   float (0–100)

        Relationship accuracy vs gold:
          - ``gold_total``:     int
          - ``gold_matched``:   int
          - ``gold_match_pct``: float (0–100)
          - ``per_item``:       list[dict] — one entry per gold expected_relation

    This function is **pure**: it reads from *store* and *gold* only; it never
    calls the LLM, writes files, or has any side effects.
    """
    all_facts = store.all_facts()
    all_edges = store.list_edges()

    # ------------------------------------------------------------------
    # Extraction sanity metrics
    # ------------------------------------------------------------------
    total_facts = len(all_facts)

    facts_with_evidence = sum(1 for f in all_facts if f.evidence_span and f.evidence_span.strip())
    facts_with_norm_val = sum(1 for f in all_facts if f.normalized_value is not None)

    if total_facts > 0:
        pct_evidence = (facts_with_evidence / total_facts) * 100.0
        pct_norm_val = (facts_with_norm_val / total_facts) * 100.0
    else:
        pct_evidence = 0.0
        pct_norm_val = 0.0

    # ------------------------------------------------------------------
    # Relationship metrics vs gold
    # ------------------------------------------------------------------
    expected_relations: list[dict] = gold.get("expected_relations", [])
    gold_total = len(expected_relations)
    gold_matched = 0
    per_item: list[dict] = []

    for item in expected_relations:
        a_contains: str = item.get("a_contains", "")
        b_contains: str = item.get("b_contains", "")
        gold_relation: str = item.get("relation", "")
        note: str = item.get("note", "")

        # Find candidate fact ids by substring search on claim_text.
        a_candidates = _fact_ids_containing(all_facts, a_contains)
        b_candidates = _fact_ids_containing(all_facts, b_contains)

        matched = bool(
            a_candidates
            and b_candidates
            and _has_matching_edge(a_candidates, b_candidates, all_edges, gold_relation)
        )

        if matched:
            gold_matched += 1

        per_item.append(
            {
                "a_contains": a_contains,
                "b_contains": b_contains,
                "relation": gold_relation,
                "note": note,
                "a_candidates_found": len(a_candidates),
                "b_candidates_found": len(b_candidates),
                "matched": matched,
            }
        )

    gold_match_pct = (gold_matched / gold_total * 100.0) if gold_total > 0 else 0.0

    return {
        # extraction sanity
        "total_facts": total_facts,
        "facts_with_evidence_span": facts_with_evidence,
        "pct_with_evidence_span": pct_evidence,
        "facts_with_normalized_value": facts_with_norm_val,
        "pct_with_normalized_value": pct_norm_val,
        # relationship accuracy
        "gold_total": gold_total,
        "gold_matched": gold_matched,
        "gold_match_pct": gold_match_pct,
        "per_item": per_item,
    }


def format_report(metrics: dict) -> str:
    """Format a metrics dict (from ``evaluate()``) as a human-readable string."""
    lines: list[str] = [
        "=" * 60,
        "Veritas Evaluation Report",
        "=" * 60,
        "",
        "Extraction Sanity",
        "-" * 40,
        f"  Total facts          : {metrics['total_facts']}",
        (
            f"  With evidence span   : {metrics['facts_with_evidence_span']}"
            f"  ({metrics['pct_with_evidence_span']:.1f}%)"
        ),
        (
            f"  With normalized value: {metrics['facts_with_normalized_value']}"
            f"  ({metrics['pct_with_normalized_value']:.1f}%)"
        ),
        "",
        "Relationship Accuracy vs Gold Set",
        "-" * 40,
        f"  Gold expected        : {metrics['gold_total']}",
        f"  Gold matched         : {metrics['gold_matched']}",
        f"  Match %              : {metrics['gold_match_pct']:.1f}%",
        "",
        "Per-item breakdown:",
    ]

    for i, item in enumerate(metrics.get("per_item", []), start=1):
        status = "MATCHED" if item["matched"] else "UNMATCHED"
        lines.append(
            f"  [{i}] {status}  "
            f'"{item["a_contains"]}" <-> "{item["b_contains"]}"  '
            f"relation={item['relation']}  "
            f"(a_cands={item['a_candidates_found']}, "
            f"b_cands={item['b_candidates_found']})"
        )
        if item.get("note"):
            lines.append(f"       note: {item['note']}")

    lines += ["", "=" * 60]
    return "\n".join(lines)
