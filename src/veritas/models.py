"""Core data models for the Veritas fact knowledge layer."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from uuid import uuid4


def new_id(prefix: str) -> str:
    """Return a short prefixed unique identifier."""
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass
class PageBlock:
    """A block of text or table extracted from a single PDF page."""

    doc_id: str
    page: int
    text: str
    char_start: int
    char_end: int
    kind: str = "text"  # "text" | "table"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> PageBlock:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Chunk:
    """A text chunk derived from one or more PageBlocks, bounded to a single page."""

    doc_id: str
    page: int
    text: str
    char_start: int
    char_end: int
    section_hint: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Fact:
    """An atomic factual claim extracted from a document chunk."""

    id: str
    doc_id: str
    subject: str
    attribute: str
    value: str
    unit: str
    temporal_context: str
    scope_qualifiers: list[str]
    evidence_span: str
    page: int
    claim_text: str
    fact_kind: str  # "numerical" | "semantic"
    confidence: float
    normalized_value: float | None = None
    normalized_unit: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    canonical_subject: str | None = None

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        # JSON-encode list field for serialization safety
        d["scope_qualifiers"] = json.dumps(d["scope_qualifiers"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Fact:
        known = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        # Decode scope_qualifiers if it was JSON-encoded
        sq = filtered.get("scope_qualifiers", "[]")
        if isinstance(sq, str):
            filtered["scope_qualifiers"] = json.loads(sq)
        return cls(**filtered)


@dataclass
class Edge:
    """A relationship between two facts across documents."""

    id: str
    fact_a_id: str
    fact_b_id: str
    relation: str  # "corroborate" | "contradict" | "reconcilable" | "unrelated"
    reasoning: str
    reconciling_dimension: str  # "time" | "scope" | "unit" | "none"
    confidence: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Edge:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Document:
    """A PDF document that has been ingested into the knowledge layer."""

    id: str
    filename: str
    content_hash: str
    num_pages: int
    status: str  # "parsing" | "extracting" | "relating" | "done" | "error"
    created_at: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Document:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Job:
    """An async ingestion job tracking document processing progress."""

    id: str
    doc_id: str
    status: str  # "queued" | "parsing" | "extracting" | "relating" | "done" | "error"
    progress: float  # 0.0–1.0
    message: str
    created_at: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Job:
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
