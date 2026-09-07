"""Fact canonicalization: cluster corroborating facts via union-find."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from veritas.models import Edge

if TYPE_CHECKING:
    from veritas.models import Fact


@runtime_checkable
class _HasId(Protocol):
    """Minimal protocol: any object with an `.id` attribute."""

    id: str


def canonical_clusters(
    facts: list[_HasId],
    edges: list[Edge],
) -> list[list[str]]:
    """Cluster fact ids using union-find over ``corroborate`` edges.

    Every fact id appears in exactly one cluster.  Facts not connected by any
    ``corroborate`` edge form singleton clusters.

    Parameters
    ----------
    facts:
        Any objects that have an ``.id`` attribute (e.g. :class:`~veritas.models.Fact`
        instances or lightweight stubs used in tests).
    edges:
        Edges to consider; only those with ``relation == "corroborate"`` are used.

    Returns
    -------
    list[list[str]]
        A partition of all fact ids into clusters.
    """
    # Initialise union-find: each id is its own parent
    parent: dict[str, str] = {f.id: f.id for f in facts}
    rank: dict[str, int] = {f.id: 0 for f in facts}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path compression (two-step)
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        # Union by rank
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    # Process only corroborate edges
    for edge in edges:
        if edge.relation != "corroborate":
            continue
        # Only union ids that are known (guard against stale edges)
        if edge.fact_a_id in parent and edge.fact_b_id in parent:
            union(edge.fact_a_id, edge.fact_b_id)

    # Group ids by root
    groups: dict[str, list[str]] = {}
    for f in facts:
        root = find(f.id)
        groups.setdefault(root, []).append(f.id)

    return list(groups.values())


def canonical_label(cluster_facts: list[Fact]) -> str:
    """Return the ``claim_text`` of the highest-confidence fact in *cluster_facts*.

    Parameters
    ----------
    cluster_facts:
        A list of :class:`~veritas.models.Fact` objects (must have
        ``confidence`` and ``claim_text`` attributes).

    Returns
    -------
    str
        The ``claim_text`` of the fact with the highest ``confidence``.
    """
    best = max(cluster_facts, key=lambda f: f.confidence)
    return best.claim_text
