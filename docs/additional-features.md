# Veritas — Additional Features Roadmap

This document lists concrete features that could be added beyond the current phase-1 prototype. Each entry includes a rationale, a rough effort estimate (S = days, M = week or two, L = month+), and what the feature unlocks for users.

---

## 1. OCR Support for Scanned PDFs

**Rationale:** The current parser (PyMuPDF + pdfplumber) extracts machine-readable text only. Scanned or image-based PDFs produce empty text blocks, yielding zero facts. Many real-world documents — government filings, legal contracts, older annual reports — exist only as scanned images.

**Implementation sketch:** Integrate `pytesseract` or `easyocr` as a fallback when PyMuPDF extracts fewer than a threshold number of characters per page. Run OCR on a rendered page image (`pymupdf` can render pages to PIL images); replace the empty text block with the OCR output and carry the same page/char-offset grounding metadata.

**Effort:** M

**Unlocks:** Full coverage of the document corpus; without OCR, a significant fraction of real-world PDFs produce no knowledge at all.

---

## 2. Active-Learning Loop for Extraction Improvement

**Rationale:** Extraction quality degrades on unusual document layouts or specialized vocabulary. The current failure slot surfaces the lowest-confidence fact, but there is no mechanism to feed failures back into the system to improve future extractions.

**Implementation sketch:** Expose a human-review endpoint (`POST /facts/{fact_id}/feedback`) that accepts `correct: bool` and an optional corrected `claim_text`. Store feedback in a `feedback` table. Periodically run a batch job that mines low-rated or low-confidence facts and uses them as few-shot negative examples in the extraction prompt. Track prompt versions alongside facts for reproducibility.

**Effort:** L

**Unlocks:** Self-improving extraction quality over time; turns the evaluation harness from a read-only metric into a training signal.

---

## 3. Contradiction Severity and Likelihood Scoring

**Rationale:** Not all contradictions are equally serious. A 0.1% discrepancy in a GDP estimate is different from a 50% discrepancy. The current `Edge.confidence` reflects adjudication confidence, not the magnitude of the disagreement.

**Implementation sketch:** For `contradict` edges where both facts have `normalized_value` set, compute a disagreement magnitude (`|a - b| / max(|a|, |b|)`). Add `severity: float` to the `Edge` model and surface it in the API and UI. Propagate severity into the cases view and export.

**Effort:** S

**Unlocks:** Prioritized contradiction review; analysts can sort contradictions by severity to focus on material discrepancies first.

---

## 4. Interactive Knowledge-Graph Visualization

**Rationale:** The current web UI lists facts and edges in tables. For dense corpora, a graph view makes structural patterns (clusters of corroborating facts, chains of contradictions) immediately visible.

**Implementation sketch:** Add a `/graph` endpoint that returns a graph-ready JSON structure (`{nodes: [...], edges: [...]}`) from the SQLite store. Render using a JavaScript graph library (e.g. Cytoscape.js or vis.js) in a new `web/graph.html` view. Node color encodes fact kind; edge color encodes relation type.

**Effort:** M

**Unlocks:** Visual exploration of the knowledge layer; useful for presentations and for spotting unexpected cross-document connections.

---

## 5. Multi-Hop and Transitive Reasoning Across Facts

**Rationale:** The current system detects direct pairwise relationships. Some inferences require chaining: if Fact A corroborates Fact B, and Fact B contradicts Fact C, the system has no way to reason that A and C are indirectly related.

**Implementation sketch:** Add a graph traversal step (BFS/DFS over the `edges` table) that identifies transitive chains up to a configurable depth. Present multi-hop paths in a `POST /reason` endpoint. The LLM is given the full chain of facts and edges and asked to summarize the net relationship.

**Effort:** L

**Unlocks:** Richer reasoning across large corpora; surfaces non-obvious structural tensions that direct pairwise comparison misses.

---

## 6. Temporal Fact-Versioning Timelines

**Rationale:** Facts about the same entity evolve over time (e.g. a company's revenue across fiscal years, a country's inflation rate across quarters). The current system stores each fact independently with a `temporal_context` field but does not link them into a coherent timeline.

**Implementation sketch:** Group facts by `canonical_subject` + `attribute` and sort by `period_start`. Expose a `GET /timeline/{canonical_subject}/{attribute}` endpoint returning the ordered series. Detect trend reversals or sudden jumps as candidates for contradiction review. Render as a sparkline in the web UI.

**Effort:** M

**Unlocks:** Longitudinal analysis; analysts can see how a metric evolved across documents rather than inspecting individual point-in-time facts.

---

## 7. Streaming and Async Ingestion Queue

**Rationale:** The current background ingestion (FastAPI `BackgroundTasks`) blocks a thread for the duration of the pipeline and does not survive a server restart. For large corpora or production use, a proper task queue is needed.

**Implementation sketch:** Replace `BackgroundTasks` with a Celery or RQ worker pool backed by Redis or a file-based broker. The `/documents` endpoint enqueues a task and returns `job_id` as before; the worker consumes the queue independently. Job status is still written to the SQLite `jobs` table. Add a Dockerfile and `docker-compose.yml` for the worker.

**Effort:** M

**Unlocks:** Production-grade ingestion; survives server restarts; enables horizontal scaling of the extraction/relating stages.

---

## 8. Distributed Vector Store

**Rationale:** ChromaDB's `PersistentClient` is single-node and single-process. For corpora with hundreds of thousands of facts, a distributed vector store with approximate nearest-neighbor (ANN) indexing is needed.

**Implementation sketch:** Implement an alternative `VectorIndex` backend using `pgvector` (PostgreSQL extension) or Qdrant. The `Embedder` protocol remains unchanged; only the storage backend swaps. Add a `VECTOR_BACKEND=chroma|pgvector|qdrant` env var to `Settings`. Run migrations for the pgvector schema via `alembic`.

**Effort:** M

**Unlocks:** Horizontal scalability; millions of facts; concurrent ingestion from multiple workers; production-grade ANN performance.

---

## 9. Authentication and Multi-Tenant Knowledge Layers

**Rationale:** The current API has no authentication. All users share a single knowledge layer. In a real deployment, different teams or projects should have isolated layers with access control.

**Implementation sketch:** Add JWT-based authentication (FastAPI-Users or a simple custom middleware). Introduce a `tenant_id` column on `documents`, `facts`, and `edges`. All queries filter by `tenant_id` derived from the JWT claims. Chroma collections are namespaced per tenant. Rate-limit ingestion per tenant.

**Effort:** L

**Unlocks:** SaaS deployment; concurrent multi-team usage; compliance with data-isolation requirements.

---

## 10. Human-in-the-Loop Review and Curation UI

**Rationale:** The LLM makes mistakes. Low-confidence facts and disputed relationships benefit from human review before being treated as ground truth.

**Implementation sketch:** Add a review queue endpoint (`GET /review/queue`) that returns facts below a confidence threshold and edges flagged as contradictions. A review UI (new web view) shows each item with the source evidence and lets a reviewer approve, reject, or correct. Approved items get `human_verified: bool = True` in the schema. Rejections delete the fact/edge; corrections create a revised record.

**Effort:** M

**Unlocks:** Human-verified knowledge layer; dramatically higher precision for downstream consumers; feeds the active-learning loop (feature 2).

---

## 11. Fact Provenance Diff Across Document Versions

**Rationale:** When a document is revised (e.g. a quarterly report updated with final figures), users need to see which facts changed, were added, or were removed relative to the previous version.

**Implementation sketch:** When a new upload shares the same filename as a prior document, detect it as a potential revision (rather than a duplicate — the content hash differs). Run a fact-level diff: match facts by `(canonical_subject, attribute, temporal_context)` and report `added`, `removed`, and `changed` sets. Expose via `GET /documents/{doc_id}/diff/{prior_doc_id}`.

**Effort:** M

**Unlocks:** Regulatory and audit use cases where tracking changes between document versions is a compliance requirement.

---

## 12. Configurable Embedding and LLM Model Registry

**Rationale:** The system supports two LLM providers (OpenAI and Ollama) and one embedding model (`all-MiniLM-L6-v2`). As new providers or models are released, upgrading currently requires code changes rather than configuration changes.

**Implementation sketch:** Add a `models.yaml` registry file listing supported LLM providers/models and embedding models with their dimension, max-token limits, and cost-per-token estimates. The `get_client` factory reads from this registry. The settings UI (or env vars) allow switching models per-deployment without code changes. Add a `POST /admin/model` endpoint for runtime model switching (requires auth).

**Effort:** S

**Unlocks:** Easy model upgrades; cost/quality trade-off tuning per deployment; A/B testing of extraction quality across model versions.
