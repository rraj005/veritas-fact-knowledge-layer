"""Relationship engine: adjudication of fact pairs and RAG retrieval."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from veritas.models import Edge, Fact, new_id
from veritas.relate.prompts import build_adjudication_prompt

if TYPE_CHECKING:
    from veritas.llm.client import LLMClient
    from veritas.store.db import Store
    from veritas.store.vectors import VectorIndex

logger = logging.getLogger(__name__)

_VALID_RELATIONS = frozenset({"corroborate", "contradict", "reconcilable", "unrelated"})
_VALID_DIMENSIONS = frozenset({"time", "scope", "unit", "none"})


def _extract_json_object(text: str) -> dict | None:
    """Robustly extract the first JSON object from *text*.

    Tries direct parsing first; falls back to extracting the first ``{...}``
    block via regex so that prose wrapping the object is tolerated.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find the outermost {...} block (greedy so brace chars inside
    # string values do not truncate the match prematurely).
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def adjudicate(fact_a: Fact, fact_b: Fact, llm: LLMClient) -> Edge | None:
    """Ask the LLM to classify the relationship between *fact_a* and *fact_b*.

    Returns an :class:`~veritas.models.Edge` for any valid relation (including
    ``"unrelated"``; the caller is responsible for filtering).  Returns
    ``None`` on parse failure (the LLM response was not parseable JSON).
    """
    system, user = build_adjudication_prompt(fact_a, fact_b)
    raw = llm.complete(system, user)

    parsed = _extract_json_object(raw)
    if parsed is None:
        logger.warning(
            "adjudicate: failed to parse LLM response for pair (%s, %s): %r",
            fact_a.id,
            fact_b.id,
            raw[:200],
        )
        return None

    relation = parsed.get("relation", "")
    if relation not in _VALID_RELATIONS:
        logger.warning(
            "adjudicate: invalid relation %r for pair (%s, %s); discarding",
            relation,
            fact_a.id,
            fact_b.id,
        )
        return None

    reconciling_dimension = parsed.get("reconciling_dimension", "none")
    if reconciling_dimension not in _VALID_DIMENSIONS:
        logger.warning(
            "adjudicate: invalid reconciling_dimension %r for pair (%s, %s); coercing to 'none'",
            reconciling_dimension,
            fact_a.id,
            fact_b.id,
        )
        reconciling_dimension = "none"

    confidence = float(parsed.get("confidence", 0.0))
    reasoning = str(parsed.get("reasoning", ""))

    return Edge(
        id=new_id("edge"),
        fact_a_id=fact_a.id,
        fact_b_id=fact_b.id,
        relation=relation,
        reasoning=reasoning,
        reconciling_dimension=reconciling_dimension,
        confidence=confidence,
    )


def find_relationships(
    fact: Fact,
    store: Store,
    index: VectorIndex,
    llm: LLMClient,
    top_k: int = 8,
) -> list[Edge]:
    """Retrieve candidate facts from other documents and adjudicate each pair.

    Strategy:
    1. Query *index* for the *top_k* most similar facts that belong to a
       **different** document (``where={"doc_id": {"$ne": fact.doc_id}}``).
    2. Load each candidate from *store* (skip if not found).
    3. Adjudicate the pair; skip ``unrelated`` results.
    4. Deduplicate by unordered fact-pair (sorted id tuple) so the same pair
       is never adjudicated more than once within this call.

    Returns a list of edges whose relation is not ``"unrelated"``.
    """
    where: dict = {"doc_id": {"$ne": fact.doc_id}}
    candidates = index.query(fact.claim_text, top_k, where=where)

    edges: list[Edge] = []
    seen_pairs: set[tuple[str, str]] = set()

    for cand_id, _distance, _meta in candidates:
        # Skip self (should not happen given the $ne filter, but be safe)
        if cand_id == fact.id:
            continue

        # Dedup by unordered pair
        pair_key = tuple(sorted((fact.id, cand_id)))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        candidate_fact = store.get_fact(cand_id)
        if candidate_fact is None:
            logger.warning("find_relationships: fact %r not found in store; skipping", cand_id)
            continue

        edge = adjudicate(fact, candidate_fact, llm)
        if edge is None:
            continue

        if edge.relation == "unrelated":
            continue

        edges.append(edge)

    return edges
