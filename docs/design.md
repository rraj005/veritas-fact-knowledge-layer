# Veritas — A Fact Knowledge Layer (Design Spec)

**Date:** 2026-09-07
**Status:** Implemented
**Assignment:** Superjoin VIT 2026 · Engineering Intern — "Build a Fact Knowledge Layer"

---

## 1. Problem & Goal

Important facts are scattered across documents, stated differently, supported by
other evidence, or contradicted elsewhere. Build a system that:

1. **Extracts** meaningful numerical or semantic facts from PDFs.
2. **Grounds** every fact to verbatim evidence in its source document (page + span).
3. **Relates** facts across documents — detecting **corroboration**, **contradiction**,
   or **apparent contradiction reconcilable by context** (time, scope, units).
4. Exposes a **simple API + UI** to upload PDFs and inspect results.

**Hard constraint — generalization.** The system must work on *any* document, in
*any* domain (finance, macroeconomics, legal, scientific, medical, news, policy,
resumes, ...). No hard-coded facts, filenames, schemas, sections, or
document-specific rules. It will be graded with unseen PDFs.

**Non-goal:** perfect extraction or a production system. A smaller, understandable,
well-reasoned prototype is the target.

## 2. Required Demonstrations (the four cases)

The system must surface, with source evidence and its own reasoning for 1–3:

1. A fact **corroborated** across documents, even if expressed differently.
2. A genuine or likely **contradiction**.
3. An **apparent contradiction explained by context** (time / scope / units).
4. An **extraction or reasoning failure** we found, and how we handled or would improve it.

Datasets used to demonstrate: `delhivery/` (finance) and `india-macroeconomy/`
(macro). Generalization is additionally proven on at least one **off-category** PDF.

## 3. Design Principles

- **Domain-neutral by construction.** Prompts and schema never name a domain concept.
- **Everything grounded.** No fact exists without a verbatim evidence span + page.
- **Retrieval before reasoning (RAG).** Find candidate related facts by semantic
  retrieval, then let the LLM reason over a small, relevant set — not the whole corpus.
- **Incremental & idempotent.** Adding a document never triggers a full rebuild;
  re-uploading the same document does not duplicate.
- **Uncertainty is a feature.** Confidence scores on extraction and adjudication;
  low-confidence items are flagged, not hidden.
- **BYOK, no secrets in repo.** The LLM is behind an interface; the user connects
  their own provider + key at runtime via the UI or API. No API key is ever stored
  in the repository.

## 4. Architecture

```
        ┌───────────┐   ┌──────────┐   ┌───────────┐   ┌──────────────┐
 PDF ─► │  Parse    │─► │  Chunk   │─► │  Extract  │─► │  Normalize   │
        │ (pages,   │   │ (page-   │   │  facts    │   │ (units/dates │
        │  tables)  │   │  anchored)│  │  (LLM)    │   │  /entities)  │
        └───────────┘   └──────────┘   └───────────┘   └──────┬───────┘
                                                              │
                       ┌──────────────────────────────────────┘
                       ▼
              ┌───────────────────┐      ┌───────────────────────────┐
              │ Store & Index     │      │  Relate (RAG core)         │
              │ SQLite (facts,    │◄────►│  retrieve similar facts    │
              │ docs, edges)      │      │  from OTHER docs → LLM      │
              │ + Chroma vectors  │      │  adjudicates each pair      │
              └─────────┬─────────┘      └───────────────────────────┘
                        │
          ┌─────────────┴──────────────┐
          ▼                            ▼
   ┌──────────────┐            ┌─────────────────┐
   │ FastAPI      │            │  Web UI (SPA)   │
   │ REST + jobs  │◄───────────│ upload/inspect  │
   └──────────────┘            └─────────────────┘
```

### 4.1 Components (each independently testable)

1. **Parser** (`parsing/`) — PyMuPDF for text with page indices; pdfplumber for
   tables (numerical facts often live in tables). Output: page-anchored text blocks
   with char offsets. Streaming/page-by-page to bound memory on large PDFs.
   *Interface:* `parse(path) -> Iterable[PageBlock]`.

2. **Chunker** (`chunking/`) — structural, bounded-size chunks carrying
   `doc_id, page, section_hint, char_start, char_end`. Grounding metadata is
   preserved end-to-end. *Interface:* `chunk(blocks) -> Iterable[Chunk]`.

3. **Extractor** (`extraction/`) — LLM extracts atomic facts per chunk into an
   **open schema**:
   ```json
   {
     "subject": "entity/topic the fact is about",
     "attribute": "what is being stated (free-form, emergent)",
     "value": "the stated value (number | string | boolean)",
     "unit": "unit if any (₹ Cr, %, persons, ...)",
     "temporal_context": "period the fact refers to (FY24, Q4FY24, 2025, as-of date)",
     "scope_qualifiers": ["consolidated", "standalone", "segment=..."],
     "evidence_span": "verbatim text supporting the fact",
     "page": 34,
     "doc_id": "...",
     "claim_text": "one-sentence natural-language statement of the fact",
     "fact_kind": "numerical | semantic",
     "confidence": 0.0-1.0
   }
   ```
   Attributes/kinds are **discovered**, never enumerated in code. Batched +
   concurrent calls; response-schema validation; malformed rows dropped and logged.

4. **Normalizer** (`normalization/`) — general, not currency-specific: numbers
   (lakh/crore/million/billion), units (via `pint` + a general table), dates/periods
   (FY/Q/CY/as-of), entity aliases & addresses (canonical form). Produces a
   `normalized_value`, `normalized_unit`, `period_start/end`, `canonical_subject`.
   Enables cross-doc matching and unit/time reconciliation.

5. **Store** (`store/`) — SQLite (`documents`, `facts`, `edges`, `jobs`,
   `attribute_vocab`) + Chroma persistent collection over `claim_text` embeddings.
   Local `sentence-transformers` embeddings (no extra API key → clean BYOK).
   Content-hash on documents for idempotent re-ingest. Incremental writes only.

6. **Relationship engine** (`relate/`) — the **RAG core**. For each new fact,
   retrieve top-k semantically similar facts from *other* documents (ANN), cap &
   dedup candidates, then the LLM adjudicates each pair →
   `{corroborate | contradict | reconcilable | unrelated}` with:
   - `reasoning` (why),
   - `reconciling_dimension` (time | scope | unit | none),
   - `confidence`.
   Edges stored with full provenance. O(k·n), never O(n²).

7. **Canonicalization** (`relate/`, stand-out extension) — facts that corroborate
   and are truly the same claim are clustered into a **canonical fact** with member
   evidence from each source (dedup / "same fact expressed differently").

8. **API** (`api/`) — FastAPI. Background job queue for uploads (large PDFs).
   Endpoints in §5.

9. **Frontend** (`web/`) — lightweight, good-looking SPA (vanilla JS + modern CSS):
   upload, fact browser with evidence snippets & page refs, relationship view, a
   **"Four Cases"** showcase, and a grounded **Ask** box (Q&A extension).

### 4.2 LLM abstraction (provider-agnostic BYOK)

`llm/client.py` defines an `LLMClient` protocol. The system has **no default
provider** — the user must connect a provider at runtime before any LLM-dependent
operation is available.

**Supported providers:**

| Provider | Requires | Base URL |
|----------|----------|----------|
| `openai`  | `OPENAI_API_KEY` env var or runtime key input | OpenAI API |
| `ollama`  | No key (local model server) | `OLLAMA_BASE_URL` (default `http://localhost:11434`) |

**Runtime configuration flow (provider-agnostic):**

1. `POST /config/llm/connect` — the client supplies `provider` (+ optional `api_key`
   and `base_url`). The backend calls the provider to list available models and
   returns that list. The API key is held **in memory only** and is never logged or
   persisted.
2. `POST /config/llm/select` — the client selects one model from the returned list.
   The backend is now ready for LLM-dependent operations (ingest, Q&A).
3. `GET /config/llm` — returns current provider, model, and `configured` flag (never
   returns the key).

Response caching (keyed on model + prompt hash) cuts cost and makes runs
reproducible across identical prompts.

## 5. API surface (FastAPI)

| Method | Path | Purpose |
|--------|------|---------|
| GET  | `/health`               | Liveness + provider + aggregate counts. |
| GET  | `/config/llm`           | Current runtime LLM config (no key returned). |
| POST | `/config/llm/connect`   | Connect a provider; returns available model list. |
| POST | `/config/llm/select`    | Select active model from the available list. |
| POST | `/documents`            | Upload a PDF; returns `job_id`; processes in background. |
| GET  | `/documents`            | List ingested documents. |
| GET  | `/jobs/{id}`            | Job status + progress. |
| GET  | `/facts`                | List/filter facts (by doc, subject, kind, confidence). |
| GET  | `/facts/{id}`           | Fact detail + verbatim evidence + page. |
| GET  | `/relationships`        | List edges (filter by type/doc), with reasoning. |
| GET  | `/relationships/{id}`   | Edge detail: both facts, evidence, reasoning. |
| GET  | `/cases`                | The four demonstrated cases, precomputed/queryable. |
| POST | `/ask`                  | Grounded Q&A over the knowledge layer (RAG, cited). |
| GET  | `/export`               | Export facts + edges (JSON/CSV). |

## 6. Brownie points & stand-out coverage (all included)

**Listed brownie points (emphasized):**
- **Large PDFs, no perf issues** — streaming parse, bounded-memory chunking, batched
  + concurrent extraction, background jobs with progress.
- **Many PDFs in one layer** — persistent vector index, retrieval-scoped relating,
  candidate capping/dedup; no O(n²).
- **Dynamically evolving schema** — open attribute vocabulary, persisted & surfaced;
  new fact kinds appear without code change.
- **Incremental ingestion** — new doc embeds only its facts and relates against the
  existing layer; content-hash idempotency; no rebuild.

**Additional stand-out extensions (beyond the list):**
- Grounded **Q&A** endpoint over the knowledge layer (cited answers).
- **Fact canonicalization / dedup** across sources.
- **Temporal fact-versioning** (e.g., director active → resigned over time).
- **Confidence & provenance** throughout; low-confidence flagging.
- **Evaluation harness** — a small hand-checked gold set + metrics (extraction
  precision, relationship accuracy) to make quality legible.
- **LLM-call caching** for cost + reproducibility.
- **Export** (JSON/CSV) of the whole knowledge layer.
- **Runtime model discovery** — provider-agnostic BYOK; no provider hard-coded.

**"What We Are Looking For" mapping** (traced fully in the stand-out document):
thoughtful/creative approach (RAG fact graph), grounded facts (verbatim spans),
handling of ambiguity/uncertainty (confidence + reconciliation), generalization
(domain-neutral prompts/schema, off-category test), clear trade-offs (documented).

## 7. Technology choices & trade-offs

| Concern | Choice | Why / trade-off |
|---|---|---|
| Language | Python 3.12 | Best PDF/RAG/ML ecosystem. |
| Parse | PyMuPDF + pdfplumber | Fast text + page anchors; pdfplumber for tables. |
| LLM | Provider-agnostic (OpenAI / Ollama), runtime BYOK | No hard-coded provider; interface keeps it swappable. |
| Embeddings | sentence-transformers (local) | No 2nd API key; free; good enough for retrieval. |
| Vector store | Chroma (persistent) | Simple, persistent, incremental. Trade-off: not distributed (fine for prototype). |
| DB | SQLite | Zero-config, portable, transactional. |
| API | FastAPI + Uvicorn | Async, background tasks, auto OpenAPI docs. |
| Frontend | Vanilla JS + CSS SPA | Good-looking but light; backend stays the focus. |
| Jobs | FastAPI BackgroundTasks + SQLite job table | Simple durable-enough progress without extra infra. |

## 8. Error handling & uncertainty

- Parse failures (scanned/no-text pages) → recorded per page; OCR is a documented
  next step, not in scope.
- Extraction JSON validation; malformed facts dropped + logged (feeds case #4).
- LLM/network errors → retry with backoff; cached results reused.
- Every fact and edge carries a confidence; the UI/API expose it; low-confidence
  items are flagged rather than silently trusted.

## 9. Testing strategy

- **Unit** per component (parser, chunker, normalizer, store, relate) with fixtures;
  no live LLM in unit tests (LLM calls mocked; a thin contract test hits the real
  client behind a flag).
- **Integration**: ingest a tiny synthetic 2-PDF pair with a known corroboration,
  contradiction, and unit/time reconciliation → assert the edges appear.
- **Generalization test**: run on an off-category fixture; assert non-empty grounded
  facts and no domain-specific code path was required.
- **Evaluation harness**: metrics over the hand-checked gold set.

## 10. Deliverables & documents

Focused artifacts:
1. **This design doc** (`docs/design.md`).
2. **Implementation plan** (`docs/implementation-plan.md`) — phased task breakdown,
   for reference.
3. **Additional-features document** — forward-looking roadmap beyond phase 1.
4. **Stand-out criteria document** — maps every *What We're Looking For* item and
   every *Brownie Point* (+ extensions) to where/how it's implemented.
5. **README.md** — Setup/Run, Video Demo, Approach, Limitations & Next Steps,
   Additional Notes (per assignment). Credentials kept out of the repo.

## 11. Out of scope (YAGNI)

OCR for scanned PDFs, distributed vector DB, multi-tenant auth, real-time
collaboration, fine-tuning. All noted as next steps where relevant.
