"""Embedder Protocol, SentenceTransformerEmbedder, FakeEmbedder."""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Wraps sentence-transformers; lazy import so the module is importable without the lib."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vecs]


class FakeEmbedder:
    """Deterministic hash-based embedder for tests.

    Same text → same vector every time; different texts → different vectors.
    Vectors are normalised to unit length so cosine distance is meaningful.
    """

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        # Build a float vector from the digest bytes (repeat if needed).
        raw: list[float] = []
        for i in range(self._dim):
            byte_val = digest[i % len(digest)]
            raw.append(float(byte_val) / 255.0 * 2.0 - 1.0)  # map to [-1, 1]

        # Normalize to unit length.
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]
