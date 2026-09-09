"""Near-duplicate fact suppression within a single document.

Facts extracted from overlapping chunks often carry almost identical
evidence spans.  This module provides ``dedupe_facts`` which collapses
near-duplicates before they reach the store, keeping only the most
confident representative from each cluster.

Two facts are considered near-duplicates when BOTH conditions hold:

1. **Same normalised subject** – both facts' ``subject`` fields reduce to
   the same lowercased, stripped string.
2. **Highly overlapping evidence spans** – the whitespace-collapsed,
   lowercased token sets of their ``evidence_span`` fields satisfy at
   least one of:
   - Jaccard similarity ≥ 0.8, OR
   - one span is a substring of the other (after normalisation).

Among a group of near-duplicates only the fact with the highest
``confidence`` is kept; ties are broken by insertion order (first wins).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from veritas.models import Fact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_span(span: str) -> str:
    """Return the whitespace-collapsed, lowercased evidence span."""
    return re.sub(r"\s+", " ", span.strip().lower())


def _token_set(normalised_span: str) -> set[str]:
    """Split a normalised span into a set of non-empty tokens."""
    return set(normalised_span.split())


def _jaccard(a: set[str], b: set[str]) -> float:
    """Return the Jaccard similarity of two token sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _normalise_subject(subject: str) -> str:
    """Return a lowercased, stripped subject string for comparison."""
    return subject.strip().lower()


def _spans_overlap(norm_a: str, norm_b: str, tokens_a: set[str], tokens_b: set[str]) -> bool:
    """Return True when two spans are considered near-duplicates.

    Two spans overlap when:
    - Jaccard similarity of their token sets is >= 0.8, OR
    - one normalised span is a substring of the other.
    """
    if _jaccard(tokens_a, tokens_b) >= 0.8:
        return True
    return bool(norm_a in norm_b or norm_b in norm_a)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def dedupe_facts(facts: list[Fact]) -> list[Fact]:
    """Remove near-duplicate facts that share the same document.

    The input list is assumed to contain facts from a **single document**
    (the pipeline calls this per-document).  Facts from different documents
    are never merged.

    Algorithm:
    - Group by normalised subject.
    - Within each subject group, greedily cluster: for each fact, if it is
      near-duplicate of any already-seen fact in a cluster add it to that
      cluster; otherwise start a new cluster.
    - From each cluster keep the fact with the highest confidence (ties:
      keep the first one encountered, i.e. the lower list-index).

    Args:
        facts: A list of :class:`~veritas.models.Fact` objects, all from
               the same document.

    Returns:
        A deduplicated list in the same relative order as the input
        (survivors appear in their original position order).
    """
    if not facts:
        return []

    # Index facts by their position so we can restore order later.
    indexed: list[tuple[int, Fact]] = list(enumerate(facts))

    # Group by normalised subject.
    from collections import defaultdict

    subject_groups: dict[str, list[tuple[int, Fact]]] = defaultdict(list)
    for idx, fact in indexed:
        key = _normalise_subject(fact.subject)
        subject_groups[key].append((idx, fact))

    survivors: list[tuple[int, Fact]] = []

    for group in subject_groups.values():
        # Pre-compute normalised spans + token sets for this group.
        norm_spans = [_normalise_span(f.evidence_span) for _, f in group]
        token_sets = [_token_set(ns) for ns in norm_spans]

        # Greedy clustering: assign each fact to the first cluster whose
        # representative span overlaps with it.
        clusters: list[list[int]] = []  # list of indices into `group`
        cluster_of: list[int] = [-1] * len(group)  # group-index → cluster-id

        for i in range(len(group)):
            placed = False
            for cid, cluster in enumerate(clusters):
                rep = cluster[0]  # representative is the first member
                if _spans_overlap(norm_spans[i], norm_spans[rep], token_sets[i], token_sets[rep]):
                    cluster.append(i)
                    cluster_of[i] = cid
                    placed = True
                    break
            if not placed:
                cluster_of[i] = len(clusters)
                clusters.append([i])

        # From each cluster pick the highest-confidence fact (ties: first).
        for cluster in clusters:
            best_gi = cluster[0]
            best_conf = group[best_gi][1].confidence
            for gi in cluster[1:]:
                conf = group[gi][1].confidence
                if conf > best_conf:
                    best_conf = conf
                    best_gi = gi
            survivors.append(group[best_gi])

    removed = len(facts) - len(survivors)
    if removed > 0:
        logger.info("dedupe_facts: removed %d near-duplicate fact(s) (kept %d)", removed, len(survivors))

    # Restore original insertion order.
    survivors.sort(key=lambda t: t[0])
    return [fact for _, fact in survivors]
