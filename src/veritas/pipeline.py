"""Incremental, idempotent ingestion pipeline for the Veritas fact knowledge layer."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from veritas.models import Document, new_id

if TYPE_CHECKING:
    from veritas.config import Settings
    from veritas.llm.client import LLMClient
    from veritas.store.db import Store
    from veritas.store.vectors import VectorIndex

logger = logging.getLogger(__name__)

# Progress callback type: (stage: str, pct: float, msg: str) -> None
ProgressCallback = Callable[[str, float, str], None]


def content_hash(path: str | Path) -> str:
    """Return the SHA-256 hex digest of the file at *path*."""
    data = Path(path).read_bytes()
    return hashlib.sha256(data).hexdigest()


class Pipeline:
    """Orchestrates the full ingestion pipeline for a single document.

    Steps (in order):
    1. Compute content_hash; if already ingested → return existing Document (no-op).
    2. Create Document with status="parsing", persist to store.
    3. Parse PDF → list[PageBlock].
    4. Chunk blocks → list[Chunk].
    5. Extract facts (with LLM concurrency) → list[Fact].
    6. Normalize each extracted fact.
    7. Persist facts to store + record attribute vocab.
    8. For EACH new fact, find_relationships against the index (which contains
       only PRIOR documents' embeddings at this point).
    9. Persist discovered edges to store.
    10. Add new doc's fact embeddings to the index (AFTER relationship search,
        so the doc never relates to itself).
    11. Update document status to "done".

    Progress is emitted via an optional ``progress_cb(stage, pct, msg)`` and
    job rows are updated when a ``job_id`` is provided.
    """

    def __init__(
        self,
        store: Store,
        index: VectorIndex,
        llm: LLMClient,
        settings: Settings,
    ) -> None:
        self._store = store
        self._index = index
        self._llm = llm
        self._settings = settings
        self.settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        path: str | Path,
        filename: str,
        job_id: str | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> Document:
        """Ingest a PDF document into the knowledge layer.

        Args:
            path:        Filesystem path to the PDF file.
            filename:    Human-readable filename to store on the Document record.
            job_id:      Optional Job id; if provided, job status is updated at
                         each stage.
            progress_cb: Optional callback ``(stage, pct, msg) -> None`` called
                         at each pipeline stage.

        Returns:
            The Document record (existing if idempotent, new otherwise).

        Raises:
            Any exception raised by a pipeline stage — after recording the
            error on both the Document and Job rows.
        """
        path = Path(path)

        def _emit(stage: str, pct: float, msg: str) -> None:
            logger.info("pipeline [%s] %.0f%% — %s", stage, pct * 100, msg)
            if progress_cb is not None:
                progress_cb(stage, pct, msg)
            if job_id is not None:
                try:
                    self._store.update_job(job_id, status=stage, progress=pct, message=msg)
                except Exception:
                    logger.debug("Failed to update job %s", job_id, exc_info=True)

        # ------------------------------------------------------------------
        # Step 1: idempotency check
        # ------------------------------------------------------------------
        file_hash = content_hash(path)
        existing = self._store.get_document_by_hash(file_hash)
        if existing is not None:
            logger.info(
                "pipeline: document %r already ingested (id=%s); skipping",
                filename,
                existing.id,
            )
            if job_id is not None:
                try:
                    self._store.update_job(
                        job_id,
                        doc_id=existing.id,
                        status="done",
                        progress=1.0,
                        message="Already ingested (duplicate document)",
                    )
                except Exception:
                    logger.debug("Failed to update job %s on duplicate path", job_id, exc_info=True)
            return existing

        # ------------------------------------------------------------------
        # Step 2: create Document record
        # ------------------------------------------------------------------
        doc_id = new_id("doc")
        now = datetime.now(tz=UTC).isoformat()
        doc = Document(
            id=doc_id,
            filename=filename,
            content_hash=file_hash,
            num_pages=0,
            status="parsing",
            created_at=now,
        )
        self._store.upsert_document(doc)

        # If there's a job, link it to the new doc id and mark it starting.
        if job_id is not None:
            try:
                self._store.update_job(
                    job_id,
                    doc_id=doc_id,
                    status="parsing",
                    progress=0.0,
                    message="Starting ingestion",
                )
            except Exception:
                logger.debug("Failed to init job %s", job_id, exc_info=True)

        try:
            # ------------------------------------------------------------------
            # Step 3: parse
            # ------------------------------------------------------------------
            _emit("parsing", 0.1, f"Parsing {filename}")

            from veritas.parsing.parser import count_pages, parse

            blocks = parse(path, doc_id)
            num_pages = count_pages(path)

            _emit("parsing", 0.2, f"Parsed {len(blocks)} blocks across {num_pages} pages")

            # ------------------------------------------------------------------
            # Step 4: chunk
            # ------------------------------------------------------------------
            _emit("extracting", 0.25, "Chunking text")

            from veritas.chunking.chunker import chunk

            chunks = chunk(blocks)

            _emit("extracting", 0.3, f"Created {len(chunks)} chunks")

            # ------------------------------------------------------------------
            # Step 5: extract facts
            # ------------------------------------------------------------------
            _emit("extracting", 0.35, "Extracting facts with LLM")

            from veritas.extraction.extractor import extract_facts

            raw_facts = extract_facts(
                chunks, self._llm, concurrency=self._settings.extract_concurrency
            )

            if len(raw_facts) == 0 and len(chunks) > 0:
                logger.warning(
                    "pipeline: extracted 0 facts from %d chunks — "
                    "the model calls may be timing out or returning nothing; "
                    "check the provider/model.",
                    len(chunks),
                )
                _emit(
                    "extracting",
                    0.55,
                    "0 facts — the model calls may be timing out or returning nothing; "
                    "check the provider/model",
                )
            else:
                _emit("extracting", 0.55, f"Extracted {len(raw_facts)} raw facts")

            # ------------------------------------------------------------------
            # Step 6: normalize each fact
            # ------------------------------------------------------------------
            from veritas.normalization.normalize import normalize_fact

            facts = [normalize_fact(f) for f in raw_facts]

            _emit("extracting", 0.6, f"Normalized {len(facts)} facts")

            # ------------------------------------------------------------------
            # Step 7: persist facts + attribute vocab
            # ------------------------------------------------------------------
            if facts:
                self._store.add_facts(facts)
                self._store.record_attributes([f.attribute for f in facts])

            _emit("relating", 0.65, f"Persisted {len(facts)} facts; finding relationships")

            # Update document num_pages now that we have it.
            doc = Document(
                id=doc.id,
                filename=doc.filename,
                content_hash=doc.content_hash,
                num_pages=num_pages,
                status="relating",
                created_at=doc.created_at,
            )
            self._store.upsert_document(doc)

            # ------------------------------------------------------------------
            # Step 8: find relationships for EACH new fact against prior docs
            # (index does NOT yet contain this doc's embeddings)
            # ------------------------------------------------------------------
            from veritas.relate.relationships import find_relationships

            all_edges = []
            for i, fact in enumerate(facts):
                edges = find_relationships(
                    fact,
                    self._store,
                    self._index,
                    self._llm,
                    top_k=self._settings.retrieve_top_k,
                )
                all_edges.extend(edges)

                _emit(
                    "relating",
                    0.65 + 0.25 * (i + 1) / len(facts),
                    f"Related fact {i + 1}/{len(facts)} → {len(edges)} edges",
                )

            # ------------------------------------------------------------------
            # Step 9: persist edges
            # ------------------------------------------------------------------
            if all_edges:
                self._store.add_edges(all_edges)

            _emit("relating", 0.9, f"Persisted {len(all_edges)} relationship edges")

            # ------------------------------------------------------------------
            # Step 10: add this doc's embeddings to the index (AFTER relate)
            # ------------------------------------------------------------------
            if facts:
                ids = [f.id for f in facts]
                texts = [f.claim_text for f in facts]
                metadatas = [{"doc_id": f.doc_id, "fact_id": f.id} for f in facts]
                self._index.add(ids, texts, metadatas)

            _emit("relating", 0.95, "Embeddings indexed")

            # ------------------------------------------------------------------
            # Step 11: mark document done
            # ------------------------------------------------------------------
            doc = Document(
                id=doc.id,
                filename=doc.filename,
                content_hash=doc.content_hash,
                num_pages=num_pages,
                status="done",
                created_at=doc.created_at,
            )
            self._store.upsert_document(doc)

            _emit("done", 1.0, f"Ingestion complete: {len(facts)} facts, {len(all_edges)} edges")

            return doc

        except Exception as exc:
            error_msg = f"Pipeline error during ingestion of {filename}: {exc}"
            logger.exception(error_msg)

            # Update document status to error.
            error_doc = Document(
                id=doc.id,
                filename=doc.filename,
                content_hash=doc.content_hash,
                num_pages=doc.num_pages,
                status="error",
                created_at=doc.created_at,
            )
            self._store.upsert_document(error_doc)

            # Update job status to error.
            if job_id is not None:
                try:
                    self._store.update_job(job_id, status="error", progress=0.0, message=error_msg)
                except Exception:
                    logger.debug("Failed to record error on job %s", job_id, exc_info=True)

            raise
