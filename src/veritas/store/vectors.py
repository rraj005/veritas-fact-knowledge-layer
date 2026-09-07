"""Chroma-backed vector index with pluggable embedder."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from veritas.store.embeddings import Embedder

_COLLECTION_NAME = "veritas_facts"


class VectorIndex:
    """Persistent vector index backed by ChromaDB.

    The embedder is called here (not by Chroma) so that Chroma receives
    pre-computed embeddings and does not need its own embedding model.
    """

    def __init__(self, chroma_dir: str | Path, embedder: Embedder) -> None:
        import chromadb  # lazy import keeps module importable without the lib

        self._embedder = embedder
        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        self._col = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            # Use pre-computed embeddings; disable Chroma's own embedding function.
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        """Embed *texts* and upsert into the collection."""
        embeddings = self._embedder.embed(texts)
        self._col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        top_k: int,
        where: dict | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Return (id, distance, metadata) tuples for the nearest neighbours.

        *where* is passed directly to Chroma's metadata filter, supporting both
        equality shorthand ``{"field": "value"}`` and operator dicts
        ``{"field": {"$ne": "value"}}``.
        """
        embedding = self._embedder.embed([text])[0]

        kwargs: dict = {
            "query_embeddings": [embedding],
            "n_results": min(top_k, self._col.count() or 1),
            "include": ["distances", "metadatas"],
        }
        if where is not None:
            kwargs["where"] = where

        results = self._col.query(**kwargs)

        ids: list[str] = results["ids"][0]
        distances: list[float] = results["distances"][0]
        metadatas: list[dict] = results["metadatas"][0]

        return list(zip(ids, distances, metadatas))

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def count(self) -> int:
        return self._col.count()
