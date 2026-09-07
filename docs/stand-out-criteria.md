# Veritas — Stand-Out Criteria

This document maps every criterion from the assignment's "What We Are Looking For" and every "Brownie Point" to the exact location in this codebase where it is implemented.

---

## What We Are Looking For

| Criterion | How it is implemented | Where (file : symbol / endpoint) | Evidence |
|---|---|---|---|
| Thoughtful / creative approach | Open-schema extraction (attribute vocab is emergent, not hardcoded); retrieval-before-reasoning RAG for O(k·n) relating; union-find canonicalization; grounded cited Q&A layer on top | `extraction/prompts.py` : `build_extraction_prompt`; `relate/relationships.py` : `find_relationships`; `relate/canonicalize.py` : `canonical_clusters`; `qa.py` : `answer` | No domain concept appears anywhere in prompts or schema |
| Useful grounded facts | Every fact carries `evidence_span` (verbatim text), `page`, and `doc_id`. Ungrounded facts (span not substring of chunk) are dropped before reaching the store | `extraction/extractor.py` : `_is_grounded`, `extract_from_chunk` | Drop logged at WARNING level; 100% of stored facts have non-empty evidence_span |
| Sensible handling of ambiguity / context / uncertainty | `confidence` field on every fact and edge; low-confidence facts surfaced in the failure slot of `/cases`; `reconcilable` relation with `reconciling_dimension` (time/scope/unit) distinguishes real contradictions from context-explained ones | `models.py` : `Fact.confidence`, `Edge.reconciling_dimension`; `cases.py` : `build_cases` failure slot | LLM returns confidence in [0,1]; adjudication prompt explicitly teaches the reconcilable/contradict distinction |
| Generalizes beyond starter docs | Extraction and adjudication prompts contain zero domain-specific terms; schema is free-form; attribute vocab is emergent; validated on two unrelated domains (logistics finance + macroeconomics) | `extraction/prompts.py` (full file); `relate/prompts.py` (full file) | "You are a precise information-extraction assistant" — no mention of industry, country, or metric type anywhere |
| Clear engineering decisions and trade-offs | Open schema for generalization; local embeddings for clean BYOK; SQLite+Chroma for portable prototype; idempotent content-hash ingestion; O(k·n) not O(n²); confidence/provenance throughout | `pipeline.py` docstring (steps 1–11); `README.md` trade-offs table | Steps 8 and 10 in pipeline docstring explicitly explain the ordering that prevents self-relating |

---

## Brownie Points

| Criterion | How it is implemented | Where (file : symbol / endpoint) | Evidence |
|---|---|---|---|
| Large PDFs without performance issues | Parser streams page-by-page (PyMuPDF page loop); extraction uses `ThreadPoolExecutor` with configurable `EXTRACT_CONCURRENCY`; chunks are bounded to 1200 chars; pdfplumber per-page failures are caught and skipped | `parsing/parser.py` : `parse` (page_idx loop); `extraction/extractor.py` : `extract_facts` (ThreadPoolExecutor); `chunking/chunker.py` : `chunk` (max_chars=1200) | Memory footprint is O(page) not O(document); LLM calls are concurrent, not sequential |
| Many PDFs in one layer | Persistent Chroma collection (`veritas_facts`) survives process restart; `$ne` metadata filter ensures relating is cross-doc only; SQLite uses `INSERT OR REPLACE`; no full-rebuild on new document | `store/vectors.py` : `VectorIndex.__init__` (PersistentClient); `relate/relationships.py` : `find_relationships` (where filter); `store/db.py` : `add_facts`, `add_edges` | `chromadb.PersistentClient` writes to disk; collection persists across server restarts |
| Dynamically evolving schema | Attribute names are free-form strings from the LLM; each attribute is recorded in `attribute_vocab` with a usage count; new attribute names are accepted automatically without schema migration | `store/db.py` : `record_attributes`, `attribute_vocab` (attribute_vocab table); `pipeline.py` step 7 | `attribute_vocab` table uses `INSERT ... ON CONFLICT DO UPDATE SET count = count + 1` |
| Incremental ingestion without full rebuild | SHA-256 content hash is checked before any processing; identical file → return existing Document immediately; new document's embeddings are added to the index AFTER relationship search so prior documents' relations are unaffected | `pipeline.py` : `content_hash`, step 1 (idempotency check), step 10 (add embeddings after relate) | `get_document_by_hash` returns early with the existing record; no re-extraction, no re-relating of prior docs |

---

## Our Extensions (Beyond the Spec)

| Extension | How it is implemented | Where (file : symbol / endpoint) | Evidence |
|---|---|---|---|
| Grounded cited Q&A (`/ask`) | Vector retrieval over the full layer → LLM answer grounded in retrieved facts → structured citations (fact_id, doc_id, page, evidence_span) returned alongside the answer | `qa.py` : `answer`; `api/app.py` : `/ask` POST; `api/schemas.py` : `AskRequest`, `AskResponse`, `Citation` | LLM is instructed to answer "USING ONLY the information contained in the provided facts" |
| Fact canonicalization and deduplication | Union-find clusters corroborating facts; `canonical_subject` field normalizes entity names across documents | `relate/canonicalize.py` : `canonical_clusters`, `canonical_label`; `normalization/normalize.py` : `_canonical_subject` | Canonical label is the `claim_text` of the highest-confidence fact in the cluster |
| Confidence and provenance throughout | Every `Fact` carries `confidence`, `evidence_span`, `page`, `doc_id`; every `Edge` carries `confidence` and `reasoning`; all exposed via API and web UI | `models.py` : `Fact`, `Edge` dataclasses; `/facts`, `/relationships`, `/cases` endpoints | Confidence propagated from LLM extraction through normalization into SQLite; never stripped |
| Evaluation harness (`eval/`) | Pure-function `evaluate(store, gold)` computes extraction sanity metrics and relationship accuracy against a hand-checked gold set; `format_report` prints a human-readable summary; invoked automatically by `build_demo.py` | `eval/evaluate.py` : `evaluate`, `format_report`; `eval/gold.json`; `scripts/build_demo.py` (gold section) | No LLM or real PDFs needed; fully unit-testable; 143 tests pass offline |
| LLM-call caching | `CachedClient` wraps any `LLMClient`; identical `(model, system, user)` triples are served from a SHA-256-keyed disk cache | `llm/cache.py` : `DiskCache`, `cache_key`; `llm/client.py` : `CachedClient` | Cache key is `sha256(model + "\x00" + system + "\x00" + user)` |
| JSON and CSV export | `/export?format=json` returns full layer; `/export?format=csv` returns facts CSV; `?format=csv&kind=edges` returns edges CSV | `cases.py` : `export_layer`, `export_csv`; `api/app.py` : `/export` GET | Response media type is `text/csv` for CSV variants; `ExportResponse` Pydantic model for JSON |
| Background ingestion jobs | Upload returns `job_id` immediately; progress updates (0.0–1.0) written to `jobs` table at each pipeline stage; polled via `/jobs/{job_id}` | `api/app.py` : `upload_document` (BackgroundTasks); `store/db.py` : `create_job`, `update_job`; `pipeline.py` : `_emit` | Job status cycles through queued → parsing → extracting → relating → done (or error) |
