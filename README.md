# Veritas — A Fact Knowledge Layer

A domain-agnostic RAG system that ingests arbitrary PDFs, extracts grounded facts, and detects corroboration, contradiction, and context-reconcilable relationships across documents — exposed via a FastAPI backend and a lightweight single-page web UI.

Demo video: <link to be added>

---

## Setup and Run Instructions

**Prerequisites:** Python 3.12, Git.

```bash
# 1. Clone and enter the repository
git clone <repo-url>
cd veritas

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install the package (editable, with dev extras)
pip install -e ".[dev]"

# 4. (Optional) Pre-configure your LLM provider via .env
cp .env.example .env
# Open .env to pre-set a provider.  Alternatively, connect via the web UI
# after starting the server (see step 5 below).
# Example for OpenAI:   LLM_PROVIDER=openai   OPENAI_API_KEY=sk-...
# Example for Ollama:   LLM_PROVIDER=ollama   (no key needed)

# 5. Run the API server
uvicorn veritas.api.app:app --reload
# Open http://127.0.0.1:8000/ in your browser
# Interactive API docs: http://127.0.0.1:8000/docs

# 6. Run the demo (ingests starter PDFs and emits the four cases)
python scripts/build_demo.py

# 7. Run the test suite
pytest -q
```

**LLM provider setup:** The system has no hard-coded provider. After starting the server, open the web UI (or call `POST /config/llm/connect`) to connect either OpenAI (supply your `OPENAI_API_KEY`) or a local Ollama instance (no key required). The backend queries the provider to list available models; select one via `POST /config/llm/select`. The API key is held in memory only and is never stored on disk.

**Embeddings** run locally via `sentence-transformers` (model `all-MiniLM-L6-v2` by default). No additional API key is required for embeddings.

All data is written to the `data/` directory (SQLite database, Chroma vector index, upload cache). This directory is gitignored.

---

## Video Demo

Demo video: <link to be added>

---

## Approach

### Architecture

```
        +-------------+   +----------+   +-----------+   +----------------+
 PDF -> |   Parse     |-> |  Chunk   |-> |  Extract  |-> |  Normalize     |
        | (PyMuPDF +  |   | (1200ch  |   |  facts    |   | (units, dates, |
        |  pdfplumber)|   |  window) |   |  (LLM)    |   |  subjects)     |
        +-------------+   +----------+   +-----------+   +-------+--------+
                                                                 |
                          +--------------------------------------+
                          v
                 +-----------------+       +---------------------------+
                 | Store & Index   |       |  Relate (RAG core)        |
                 | SQLite (facts,  |<----->|  embed new fact ->         |
                 | docs, edges,    |       |  retrieve top-k similar    |
                 | attr_vocab)     |       |  from OTHER docs ->        |
                 | + Chroma vector |       |  LLM adjudicates pair      |
                 +---------+-------+       +---------------------------+
                           |
             +-------------+--------------+
             v                            v
      +--------------+            +-----------------+
      | FastAPI      |            |  Web UI (SPA)   |
      | REST + jobs  |<-----------| upload/inspect  |
      +--------------+            +-----------------+
```

### Pipeline Stages

1. **Parse** (`parsing/parser.py`) — PyMuPDF extracts text blocks page-by-page; pdfplumber extracts tables rendered as pipe-delimited text. Both produce `PageBlock` objects with document-global character offsets. Parsing streams page by page, bounding memory on large PDFs.

2. **Chunk** (`chunking/chunker.py`) — Same-page blocks are concatenated then sliced into overlapping windows (default 1200 chars, 150-char overlap). Chunks never cross page boundaries, preserving grounding metadata end-to-end.

3. **Extract** (`extraction/extractor.py`) — The LLM receives each chunk and returns a JSON array of atomic facts in an **open schema**: `subject`, `attribute`, `value`, `unit`, `temporal_context`, `scope_qualifiers`, `evidence_span`, `claim_text`, `fact_kind`, `confidence`. No domain-specific schema is enforced; the attribute vocabulary emerges from the documents. Facts whose `evidence_span` is not a verbatim substring of the chunk text are dropped (grounding check). Extraction runs concurrently via a `ThreadPoolExecutor` (configurable `EXTRACT_CONCURRENCY`).

4. **Normalize** (`normalization/normalize.py`) — Fills `normalized_value` (numeric value scaled by unit multipliers such as crore/million/lakh), `normalized_unit`, `period_start`/`period_end` (parsed from FY/CY/quarter strings), and `canonical_subject` (lowercased, punctuation-stripped). Uses `pint` for physical unit normalization.

5. **Store** (`store/db.py`) — Facts, edges, documents, jobs, and a dynamic `attribute_vocab` table are persisted in SQLite. All writes use `INSERT OR REPLACE` so re-ingesting is a no-op.

6. **Index** (`store/vectors.py`) — Fact `claim_text` fields are embedded by a local `SentenceTransformerEmbedder` and stored in a persistent ChromaDB collection. Embeddings for the new document are added **after** relationship search, so a document never relates to itself.

7. **Relate** (`relate/relationships.py`) — For each new fact, the vector index is queried for the top-k most semantically similar facts from **other** documents (`$ne` metadata filter). Each candidate pair is presented to the LLM for adjudication: `corroborate`, `contradict`, `reconcilable`, or `unrelated`. The LLM provides `reasoning`, a `reconciling_dimension` (time / scope / unit), and a `confidence` score. `unrelated` pairs are discarded. This is O(k·n) in the number of new facts, not O(n²) over the full corpus.

8. **Canonicalize** (`relate/canonicalize.py`) — Corroborating fact clusters are computed with union-find over `corroborate` edges. The highest-confidence fact in each cluster provides the canonical label.

### Basic RAG Technique

Veritas uses **retrieval-before-reasoning**: rather than sending all known facts to the LLM (which is slow and expensive), it first retrieves a small set of semantically similar candidates using vector similarity, then asks the LLM to reason only over that focused context. This makes relationship detection practical at scale — the LLM sees only the most relevant pairs, not the full corpus.

### Key Decisions and Trade-offs

| Decision | Rationale | Trade-off |
|---|---|---|
| Open/dynamic fact schema | Generalizes to any domain without hard-coded fields | Attribute names vary; normalization is best-effort |
| Local embeddings (sentence-transformers) | Clean BYOK — no embedding API key required | Embedding quality lower than hosted models for niche domains |
| SQLite + Chroma | Portable, zero-infrastructure prototype | Single-node; not suitable for distributed deployments |
| Incremental + idempotent ingestion | Content-hash deduplication; re-uploading is safe | Requires stable content; byte-identical files only |
| O(k·n) relating (not O(n²)) | Retrieval scoped by semantic similarity; scales to many documents | High-k retrieval can miss non-obvious cross-domain connections |
| LLM-call caching (DiskCache) | Repeated identical prompts served from disk | Cache grows unbounded; no TTL |
| Facts indexed AFTER relate | Prevents self-relating within a single document | Edges are only created at ingestion time; new edges to later documents require re-ingestion |

### AI Tools Used

AI coding assistants were used during development.

---

## The Four Required Cases

The system demonstrates all four required cases via the `/cases` endpoint and the "Four Cases" view in the web UI. Run `python scripts/build_demo.py` to populate a demo layer and generate `docs/demo-cases.md`.

1. **Corroboration** — Two facts from different documents asserting compatible information about the same subject, e.g. the same metric reported by two independent sources. The LLM returns `relation: "corroborate"` with supporting reasoning.

2. **Genuine contradiction** — Two facts making numerically or semantically incompatible claims for the same scope and period, e.g. the same fiscal-deficit-to-GDP ratio reported with different values by two publications (preliminary vs. revised estimates). The LLM returns `relation: "contradict"`.

3. **Apparent contradiction reconciled by context** — Two facts that look inconsistent but are explained by a differing time window, population scope, or unit of measurement. The LLM returns `relation: "reconcilable"` with `reconciling_dimension: "time"` (or `"scope"` or `"unit"`).

4. **Extraction/reasoning failure** — Surfaced as the `failure` slot in `/cases`: the lowest-confidence fact in the layer. This may be a fact where the LLM hedged (low confidence score), or where the evidence span is borderline. In the extractor, facts with ungrounded `evidence_span` values are silently dropped before reaching the store; the confidence field flags borderline cases that do make it through.

See `scripts/build_demo.py` and `docs/demo-cases.md` (generated by the script) for real examples, and the `/cases` endpoint for the live JSON representation.

---

## Generalization

Veritas is **domain-neutral by construction**:

- The extraction prompt (`extraction/prompts.py`) never names a domain concept, industry metric, or entity class.
- The adjudication prompt (`relate/prompts.py`) reasons entirely from the structured fields of each fact.
- No filenames, section headings, or domain-specific regex appear anywhere in the pipeline code.
- The attribute vocabulary is **emergent** — it is learned from the documents and stored in the `attribute_vocab` table.
- The system has been validated on the provided `delhivery/` (logistics/finance) and `india-macroeconomy/` (macroeconomics) starter datasets and is intended to work on any uploaded PDF without configuration changes.

---

## API

All endpoints are served by `veritas.api.app` (FastAPI). Interactive docs are at `http://127.0.0.1:8000/docs`.

| Method | Path | Tag | Description |
|--------|------|-----|-------------|
| GET | `/health` | meta | Liveness check; returns provider and aggregate counts |
| GET | `/config/llm` | config | Current runtime LLM config (provider, model, available models — never returns the key) |
| POST | `/config/llm/connect` | config | Connect a provider (OpenAI or Ollama); returns list of available models |
| POST | `/config/llm/select` | config | Select the active model from the available list |
| POST | `/documents` | ingest | Upload a PDF for background ingestion; returns `job_id` |
| GET | `/documents` | ingest | List all ingested documents |
| GET | `/jobs/{job_id}` | ingest | Poll ingestion job status and progress (0.0–1.0) |
| GET | `/facts` | facts | List facts with optional filters (`doc_id`, `subject`, `kind`, `min_conf`, `limit`, `offset`) |
| GET | `/facts/{fact_id}` | facts | Retrieve a single fact with verbatim evidence and page |
| GET | `/relationships` | relationships | List relationship edges, each enriched with both facts' detail |
| GET | `/relationships/{edge_id}` | relationships | Retrieve a single edge |
| GET | `/cases` | cases | One example of each relation type plus the failure slot |
| POST | `/ask` | qa | Grounded Q&A: answer a question with fact citations |
| GET | `/export` | export | Export layer as JSON (`?format=json`) or CSV (`?format=csv&kind=facts\|edges`) |
| GET | `/` | web | Serves the single-page web UI |

---

## Limitations and Next Steps

**Current limitations:**

- No OCR support — scanned PDFs (image-only) produce no text blocks and will yield zero facts.
- Extraction and adjudication quality are bounded by the LLM; low-quality or ambiguous text produces low-confidence or incorrect facts.
- Retrieval recall bounds cross-document linking: pairs that are semantically dissimilar but topically related may be missed at the default `top_k=8`.
- ChromaDB runs single-node; not suitable for distributed or high-throughput deployments.
- No authentication or multi-tenancy — all uploaded documents share a single knowledge layer.
- The demo script and `/cases` require a real API key; the layer is otherwise queryable without one.
- No support for document versioning — re-uploading a modified PDF creates a new document rather than diffing against the old one.

See `docs/additional-features.md` for a forward-looking roadmap.

---

## Additional Notes

**BYOK — credentials never enter the repo.** Copy `.env.example` to `.env` and set your key. The `.env` file is gitignored. Sample outputs (e.g. `docs/demo-cases.md` generated by `build_demo.py`) and a video recording substitute for a shared key during review.

**Test suite:** 143 tests pass (`pytest -q`). Tests are fully offline — the LLM is replaced by `FakeLLMClient` and embeddings use `FakeEmbedder`. No API key is required to run the test suite.

**Design and planning documents:**
- `docs/design.md` — design spec
- `docs/implementation-plan.md` — implementation plan
