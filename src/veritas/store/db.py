"""SQLite-backed store for documents, facts, edges, jobs, and attribute vocab."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from veritas.models import Document, Edge, Fact, Job

_ALLOWED_JOB_FIELDS = frozenset({"status", "progress", "message", "doc_id"})


class Store:
    """Persistent SQLite store for the Veritas fact knowledge layer.

    All writes are idempotent: ``upsert_document``, ``add_facts``, and
    ``add_edges`` use ``INSERT OR REPLACE`` keyed on the row ``id`` so
    re-ingesting the same data is a no-op.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self._db_path), check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_schema(self) -> None:
        """Create tables if they do not already exist (idempotent)."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id           TEXT PRIMARY KEY,
                filename     TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE,
                num_pages    INTEGER NOT NULL DEFAULT 0,
                status       TEXT NOT NULL DEFAULT 'done',
                created_at   TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS facts (
                id                TEXT PRIMARY KEY,
                doc_id            TEXT NOT NULL,
                subject           TEXT NOT NULL,
                attribute         TEXT NOT NULL,
                value             TEXT NOT NULL,
                unit              TEXT NOT NULL DEFAULT '',
                temporal_context  TEXT NOT NULL DEFAULT '',
                scope_qualifiers  TEXT NOT NULL DEFAULT '[]',
                evidence_span     TEXT NOT NULL,
                page              INTEGER NOT NULL DEFAULT 0,
                claim_text        TEXT NOT NULL DEFAULT '',
                fact_kind         TEXT NOT NULL DEFAULT 'numerical',
                confidence        REAL NOT NULL DEFAULT 0.0,
                normalized_value  REAL,
                normalized_unit   TEXT,
                period_start      TEXT,
                period_end        TEXT,
                canonical_subject TEXT
            );

            CREATE TABLE IF NOT EXISTS edges (
                id                    TEXT PRIMARY KEY,
                fact_a_id             TEXT NOT NULL,
                fact_b_id             TEXT NOT NULL,
                relation              TEXT NOT NULL,
                reasoning             TEXT NOT NULL DEFAULT '',
                reconciling_dimension TEXT NOT NULL DEFAULT 'none',
                confidence            REAL NOT NULL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS jobs (
                id         TEXT PRIMARY KEY,
                doc_id     TEXT NOT NULL,
                status     TEXT NOT NULL DEFAULT 'queued',
                progress   REAL NOT NULL DEFAULT 0.0,
                message    TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attribute_vocab (
                attribute TEXT PRIMARY KEY,
                count     INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Documents
    # ------------------------------------------------------------------

    def upsert_document(self, doc: Document) -> None:
        """Insert or replace a document (keyed on id and content_hash)."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO documents
                (id, filename, content_hash, num_pages, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                doc.id,
                doc.filename,
                doc.content_hash,
                doc.num_pages,
                doc.status,
                doc.created_at,
            ),
        )
        self._conn.commit()

    def get_document_by_hash(self, content_hash: str) -> Document | None:
        """Return the document with the given content hash, or None."""
        row = self._conn.execute(
            "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(self) -> list[Document]:
        """Return all documents ordered by creation time."""
        rows = self._conn.execute("SELECT * FROM documents ORDER BY created_at").fetchall()
        return [self._row_to_document(r) for r in rows]

    # ------------------------------------------------------------------
    # Facts
    # ------------------------------------------------------------------

    def add_facts(self, facts: list[Fact]) -> None:
        """Insert or replace a batch of facts."""
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO facts (
                id, doc_id, subject, attribute, value, unit, temporal_context,
                scope_qualifiers, evidence_span, page, claim_text, fact_kind,
                confidence, normalized_value, normalized_unit,
                period_start, period_end, canonical_subject
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    f.id,
                    f.doc_id,
                    f.subject,
                    f.attribute,
                    f.value,
                    f.unit,
                    f.temporal_context,
                    json.dumps(f.scope_qualifiers),
                    f.evidence_span,
                    f.page,
                    f.claim_text,
                    f.fact_kind,
                    f.confidence,
                    f.normalized_value,
                    f.normalized_unit,
                    f.period_start,
                    f.period_end,
                    f.canonical_subject,
                )
                for f in facts
            ],
        )
        self._conn.commit()

    def get_fact(self, fact_id: str) -> Fact | None:
        """Return a single fact by id, or None."""
        row = self._conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return self._row_to_fact(row) if row else None

    def list_facts(
        self,
        doc_id: str | None = None,
        subject: str | None = None,
        kind: str | None = None,
        min_conf: float = 0.0,
        limit: int = 200,
        offset: int = 0,
    ) -> list[Fact]:
        """Return facts with optional filters."""
        clauses: list[str] = ["confidence >= ?"]
        params: list[Any] = [min_conf]

        if doc_id is not None:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        if subject is not None:
            clauses.append("subject = ?")
            params.append(subject)
        if kind is not None:
            clauses.append("fact_kind = ?")
            params.append(kind)

        where_sql = " AND ".join(clauses)
        params.extend([limit, offset])

        rows = self._conn.execute(
            f"SELECT * FROM facts WHERE {where_sql} LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def all_facts(self) -> list[Fact]:
        """Return every fact in the store."""
        rows = self._conn.execute("SELECT * FROM facts").fetchall()
        return [self._row_to_fact(r) for r in rows]

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    def add_edges(self, edges: list[Edge]) -> None:
        """Insert or replace a batch of edges."""
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO edges
                (id, fact_a_id, fact_b_id, relation, reasoning,
                 reconciling_dimension, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    e.id,
                    e.fact_a_id,
                    e.fact_b_id,
                    e.relation,
                    e.reasoning,
                    e.reconciling_dimension,
                    e.confidence,
                )
                for e in edges
            ],
        )
        self._conn.commit()

    def get_edge(self, edge_id: str) -> Edge | None:
        """Return a single edge by id, or None."""
        row = self._conn.execute("SELECT * FROM edges WHERE id = ?", (edge_id,)).fetchone()
        return self._row_to_edge(row) if row else None

    def list_edges(
        self,
        relation: str | None = None,
        doc_id: str | None = None,
    ) -> list[Edge]:
        """Return edges with optional filters.

        ``doc_id`` matches edges where *either* endpoint fact belongs to
        that document (joined through the facts table).
        """
        if doc_id is not None:
            # Join through facts to filter by doc on either side.
            sql = """
                SELECT DISTINCT e.*
                FROM edges e
                JOIN facts fa ON fa.id = e.fact_a_id
                JOIN facts fb ON fb.id = e.fact_b_id
                WHERE (fa.doc_id = ? OR fb.doc_id = ?)
            """
            params: list[Any] = [doc_id, doc_id]
            if relation is not None:
                sql += " AND e.relation = ?"
                params.append(relation)
        else:
            sql = "SELECT * FROM edges"
            params = []
            if relation is not None:
                sql += " WHERE relation = ?"
                params.append(relation)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_edge(r) for r in rows]

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def create_job(self, job: Job) -> None:
        """Insert a new job record."""
        self._conn.execute(
            """
            INSERT OR REPLACE INTO jobs (id, doc_id, status, progress, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (job.id, job.doc_id, job.status, job.progress, job.message, job.created_at),
        )
        self._conn.commit()

    def update_job(self, job_id: str, **fields: Any) -> None:
        """Update allowed fields on a job row.

        Only the columns listed in ``_ALLOWED_JOB_FIELDS`` may be supplied;
        any unknown key raises ``ValueError`` to prevent SQL injection via
        caller-controlled column names.
        """
        if not fields:
            return
        bad = set(fields) - _ALLOWED_JOB_FIELDS
        if bad:
            raise ValueError(f"Unknown job fields: {bad}")
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        params = list(fields.values()) + [job_id]
        self._conn.execute(f"UPDATE jobs SET {set_clause} WHERE id = ?", params)
        self._conn.commit()

    def get_job(self, job_id: str) -> Job | None:
        """Return a single job by id, or None."""
        row = self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row else None

    # ------------------------------------------------------------------
    # Attribute vocab (dynamic schema surfacing)
    # ------------------------------------------------------------------

    def record_attributes(self, attrs: list[str]) -> None:
        """Increment occurrence counts for each attribute name."""
        for attr in attrs:
            self._conn.execute(
                """
                INSERT INTO attribute_vocab (attribute, count)
                VALUES (?, 1)
                ON CONFLICT(attribute) DO UPDATE SET count = count + 1
                """,
                (attr,),
            )
        self._conn.commit()

    def attribute_vocab(self) -> list[tuple[str, int]]:
        """Return ``(attribute, count)`` pairs sorted by descending count."""
        rows = self._conn.execute(
            "SELECT attribute, count FROM attribute_vocab ORDER BY count DESC"
        ).fetchall()
        return [(r["attribute"], r["count"]) for r in rows]

    # ------------------------------------------------------------------
    # Aggregate counts
    # ------------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        """Return aggregate row counts for docs, facts, and edges.

        Uses three explicit fixed-string queries (no table-name interpolation)
        to avoid any possibility of SQL injection through the table argument.
        """
        docs = self._conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        facts = self._conn.execute("SELECT COUNT(*) AS n FROM facts").fetchone()["n"]
        edges = self._conn.execute("SELECT COUNT(*) AS n FROM edges").fetchone()["n"]
        return {"docs": docs, "facts": facts, "edges": edges}

    # ------------------------------------------------------------------
    # Internal row → dataclass converters
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> Document:
        return Document.from_dict(dict(row))

    @staticmethod
    def _row_to_fact(row: sqlite3.Row) -> Fact:
        d = dict(row)
        # scope_qualifiers is stored as a JSON text column
        sq = d.get("scope_qualifiers", "[]")
        if isinstance(sq, str):
            d["scope_qualifiers"] = json.loads(sq)
        return Fact.from_dict(d)

    @staticmethod
    def _row_to_edge(row: sqlite3.Row) -> Edge:
        return Edge.from_dict(dict(row))

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job.from_dict(dict(row))
