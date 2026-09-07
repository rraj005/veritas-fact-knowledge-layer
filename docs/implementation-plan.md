# Fact Knowledge Layer (Veritas) Implementation Plan

**Goal:** Build Veritas — a domain-agnostic RAG "fact knowledge layer" that ingests arbitrary PDFs, extracts grounded facts, and detects corroboration / contradiction / context-reconcilable relationships across documents, exposed via a FastAPI backend and a lightweight web UI.

**Architecture:** A staged pipeline (parse → chunk → LLM extract → normalize → store+embed → RAG relate → canonicalize) writing to SQLite + a persistent Chroma vector index. Relationship detection retrieves semantically similar facts from *other* documents and has the LLM adjudicate each pair. Everything is grounded to verbatim evidence + page. The LLM sits behind a provider-agnostic BYOK interface (OpenAI or Ollama; no default provider). Ingestion is incremental and idempotent.

**Tech Stack:** Python 3.12, PyMuPDF, pdfplumber, openai SDK, httpx (Ollama), sentence-transformers, ChromaDB, SQLite (stdlib sqlite3), FastAPI, Uvicorn, pytest, pint.

**Design doc:** `docs/design.md`

## Global Constraints

- Python **3.12**; all code type-hinted; formatted with `ruff`.
- **Domain-neutral:** no hard-coded facts, filenames, sections, metrics, or domain regex anywhere. Extraction prompts never name a domain concept.
- **BYOK:** no API keys in the repo. LLM provider and model are configured at runtime via `POST /config/llm/connect` and `POST /config/llm/select`. Optional env vars (`LLM_PROVIDER`, `LLM_MODEL`, `OPENAI_API_KEY`, `OLLAMA_BASE_URL`) can pre-seed settings. `.env` is gitignored; `.env.example` is committed.
- **Grounding:** every fact carries `evidence_span` (verbatim) + `page` + `doc_id`. No ungrounded facts.
- **Incremental & idempotent:** ingesting a document never rebuilds the layer; re-ingesting identical content (same `content_hash`) is a no-op.
- **No live LLM in unit tests:** the LLM client is mocked via a `FakeLLMClient`; embeddings use a deterministic fake in unit tests.
- **Package root:** `src/veritas/`. Tests mirror under `tests/`.
- **Commits:** conventional commits. Never `--no-verify`.

---

## File Structure

```
src/veritas/
  __init__.py
  config.py              # env-driven settings (Settings)
  models.py              # dataclasses: PageBlock, Chunk, Fact, Edge, Document, Job
  llm/
    __init__.py
    client.py            # LLMClient protocol, OpenAIClient, OllamaClient, FakeLLMClient, CachedClient, get_client()
    cache.py             # on-disk prompt->response cache
  parsing/parser.py      # parse(path) -> list[PageBlock]
  chunking/chunker.py    # chunk(blocks) -> list[Chunk]
  extraction/extractor.py# extract_facts(chunks, llm) -> list[Fact]
  extraction/prompts.py  # domain-neutral extraction prompt builder
  normalization/normalize.py  # normalize_fact(fact) -> Fact (fills normalized_* fields)
  store/db.py            # SQLite schema + CRUD
  store/vectors.py       # Chroma wrapper (add/query embeddings)
  store/embeddings.py    # Embedder protocol, SentenceTransformerEmbedder, FakeEmbedder
  relate/relationships.py# find_relationships(fact, store, llm) -> list[Edge]
  relate/prompts.py      # domain-neutral adjudication prompt
  relate/canonicalize.py # cluster corroborating facts -> canonical facts
  pipeline.py            # ingest_document(path) orchestration + progress
  qa.py                  # grounded Q&A over the layer
  api/app.py             # FastAPI app + routes
  api/schemas.py         # pydantic request/response models
  web/index.html         # SPA
  web/app.js
  web/styles.css
tests/...                # mirrors src, plus tests/fixtures/
scripts/build_demo.py    # ingest starter datasets, emit the four cases
eval/gold.json           # hand-checked gold set
eval/evaluate.py         # metrics harness
```

---

## Task 1: Project scaffolding & config

**Files:**
- Create: `pyproject.toml`, `.env.example`, `src/veritas/__init__.py`, `src/veritas/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `veritas.config.Settings` dataclass with fields `llm_provider: str`, `llm_model: str`, `openai_api_key: str | None`, `ollama_base_url: str`, `data_dir: Path`, `db_path: Path`, `chroma_dir: Path`, `cache_dir: Path`, `upload_dir: Path`, `embedding_model: str`, `extract_concurrency: int`, `retrieve_top_k: int`. Factory `get_settings() -> Settings` reads env; `llm_provider` defaults to empty string (provider has no default — must be set explicitly or connected at runtime).

- [ ] **Step 1: Write the failing test**
```python
# tests/test_config.py
import os
from veritas.config import get_settings

def test_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("VERITAS_DATA_DIR", str(tmp_path))
    s = get_settings()
    assert s.llm_provider == ""  # no default provider
    assert s.retrieve_top_k == 8
    assert s.db_path == tmp_path / "veritas.db"

def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("VERITAS_DATA_DIR", str(tmp_path))
    s = get_settings()
    assert s.llm_provider == "openai"
    assert s.llm_model == "gpt-4o"
```

- [ ] **Step 2: Run test to verify it fails** — `pytest tests/test_config.py -v` → FAIL (module not found).

- [ ] **Step 3: Implement `pyproject.toml`** with deps: `pymupdf`, `pdfplumber`, `openai`, `httpx`, `sentence-transformers`, `chromadb`, `fastapi`, `uvicorn[standard]`, `pint`, `python-dotenv`, `pydantic`; dev: `pytest`, `ruff`. Configure `[tool.pytest.ini_options] pythonpath = ["src"]`.

- [ ] **Step 4: Implement `config.py`**
```python
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    openai_api_key: str | None
    ollama_base_url: str
    data_dir: Path
    db_path: Path
    chroma_dir: Path
    cache_dir: Path
    upload_dir: Path
    embedding_model: str
    extract_concurrency: int
    retrieve_top_k: int

def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "")  # no default — must be set explicitly
    data_dir = Path(os.getenv("VERITAS_DATA_DIR", "data"))
    for sub in ("", "chroma", "cache", "uploads"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)
    return Settings(
        llm_provider=provider,
        llm_model=os.getenv("LLM_MODEL", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        data_dir=data_dir,
        db_path=data_dir / "veritas.db",
        chroma_dir=data_dir / "chroma",
        cache_dir=data_dir / "cache",
        upload_dir=data_dir / "uploads",
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        extract_concurrency=int(os.getenv("EXTRACT_CONCURRENCY", "4")),
        retrieve_top_k=int(os.getenv("RETRIEVE_TOP_K", "8")),
    )
```
Write `.env.example` with commented `LLM_PROVIDER`, `LLM_MODEL`, `OPENAI_API_KEY=`, `OLLAMA_BASE_URL=`, etc.

- [ ] **Step 5: Run tests** — `pytest tests/test_config.py -v` → PASS.
- [ ] **Step 6: Commit** — `feat: project scaffolding and env-driven settings`.

---

## Task 2: Core data models

**Files:**
- Create: `src/veritas/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces dataclasses (all with `to_dict()`/`from_dict()`):
  - `PageBlock(doc_id: str, page: int, text: str, char_start: int, char_end: int, kind: str = "text")` (`kind` ∈ `text|table`).
  - `Chunk(doc_id: str, page: int, text: str, char_start: int, char_end: int, section_hint: str = "")`.
  - `Fact(id, doc_id, subject, attribute, value, unit, temporal_context, scope_qualifiers: list[str], evidence_span, page, claim_text, fact_kind, confidence, normalized_value: float | None = None, normalized_unit: str | None = None, period_start: str | None = None, period_end: str | None = None, canonical_subject: str | None = None)`.
  - `Edge(id, fact_a_id, fact_b_id, relation, reasoning, reconciling_dimension, confidence)` — `relation` ∈ `corroborate|contradict|reconcilable|unrelated`; `reconciling_dimension` ∈ `time|scope|unit|none`.
  - `Document(id, filename, content_hash, num_pages, status, created_at)`.
  - `Job(id, doc_id, status, progress, message, created_at)` — `status` ∈ `queued|parsing|extracting|relating|done|error`.
  - Helper `new_id(prefix: str) -> str` returning `f"{prefix}_{uuid4().hex[:12]}"`.

- [ ] **Step 1: Write failing test**
```python
# tests/test_models.py
from veritas.models import Fact, Edge, new_id

def test_fact_roundtrip():
    f = Fact(id=new_id("fact"), doc_id="d1", subject="Company", attribute="revenue",
             value="8142", unit="INR Cr", temporal_context="FY24", scope_qualifiers=["consolidated"],
             evidence_span="Revenue was Rs 8,142 Cr", page=12, claim_text="Revenue FY24 was 8142 INR Cr",
             fact_kind="numerical", confidence=0.9)
    assert Fact.from_dict(f.to_dict()) == f

def test_new_id_prefix():
    assert new_id("edge").startswith("edge_")
```

- [ ] **Step 2: Run** → FAIL. 
- [ ] **Step 3: Implement** dataclasses with `to_dict`/`from_dict` (JSON-encode `scope_qualifiers`). Use `@dataclass` and `dataclasses.asdict`; `from_dict` filters unknown keys.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `feat: core data models for facts, edges, documents, jobs`.

---

## Task 3: PDF parser

**Files:**
- Create: `src/veritas/parsing/__init__.py`, `src/veritas/parsing/parser.py`
- Test: `tests/parsing/test_parser.py`, fixture `tests/fixtures/make_pdf.py`

**Interfaces:**
- Consumes: `PageBlock` from Task 2.
- Produces: `parse(path: str | Path, doc_id: str) -> list[PageBlock]`. Yields text blocks per page (PyMuPDF `page.get_text("blocks")`) with running global `char_start/char_end`; appends table blocks (pdfplumber `extract_tables`) rendered as pipe-delimited text with `kind="table"`. Pages with no extractable text produce no block but are counted. Also `count_pages(path) -> int`.

- [ ] **Step 1: Fixture helper** `make_pdf.py`: `make_pdf(path, pages: list[str])` using PyMuPDF to write text pages (for deterministic tests).
- [ ] **Step 2: Write failing test**
```python
# tests/parsing/test_parser.py
from veritas.parsing.parser import parse, count_pages
from tests.fixtures.make_pdf import make_pdf

def test_parse_pages(tmp_path):
    p = tmp_path / "doc.pdf"
    make_pdf(p, ["Revenue was 100 crore in FY24.", "Profit grew 20 percent."])
    blocks = parse(p, "d1")
    assert count_pages(p) == 2
    assert any("Revenue" in b.text for b in blocks)
    assert all(b.doc_id == "d1" for b in blocks)
    assert blocks[0].char_start == 0 and blocks[0].char_end == len(blocks[0].text)
```
- [ ] **Step 3: Run** → FAIL.
- [ ] **Step 4: Implement** `parse` with PyMuPDF for text + pdfplumber for tables; maintain a global offset counter so `char_start/char_end` are document-global (needed for evidence anchoring). Guard pdfplumber failures per page (log, continue).
- [ ] **Step 5: Run** → PASS.
- [ ] **Step 6: Commit** — `feat: page-anchored PDF parser with table extraction`.

---

## Task 4: Chunker

**Files:**
- Create: `src/veritas/chunking/__init__.py`, `src/veritas/chunking/chunker.py`
- Test: `tests/chunking/test_chunker.py`

**Interfaces:**
- Consumes: `PageBlock`, `Chunk`.
- Produces: `chunk(blocks: list[PageBlock], max_chars: int = 1200, overlap: int = 150) -> list[Chunk]`. Concatenates consecutive blocks *within the same page* into ≤`max_chars` chunks with `overlap`, preserving `char_start/char_end` and `page`. Never merges across pages (keeps grounding page-accurate). `section_hint` = first line of the block if it looks like a heading (short, title-ish), else "".

- [ ] **Step 1: Failing test**
```python
def test_chunk_respects_page_and_size():
    from veritas.models import PageBlock
    from veritas.chunking.chunker import chunk
    blocks = [PageBlock("d1", 1, "A"*1000, 0, 1000), PageBlock("d1", 1, "B"*1000, 1000, 2000),
              PageBlock("d1", 2, "C"*500, 2000, 2500)]
    chunks = chunk(blocks, max_chars=1200, overlap=100)
    assert all(len(c.text) <= 1200 for c in chunks)
    assert all(c.page in (1, 2) for c in chunks)
    assert {c.page for c in chunks} == {1, 2}
```
- [ ] **Step 2: Run** → FAIL. 
- [ ] **Step 3: Implement** page-grouped sliding window.
- [ ] **Step 4: Run** → PASS. 
- [ ] **Step 5: Commit** — `feat: page-safe chunker with overlap`.

---

## Task 5: LLM client abstraction (BYOK) + cache

**Files:**
- Create: `src/veritas/llm/__init__.py`, `src/veritas/llm/client.py`, `src/veritas/llm/cache.py`
- Test: `tests/llm/test_client.py`, `tests/llm/test_cache.py`

**Interfaces:**
- Produces:
  - `class LLMClient(Protocol): def complete(self, system: str, user: str, *, max_tokens: int = 2048, temperature: float = 0.0) -> str: ...`
  - `OpenAIClient(api_key, model)` and `OllamaClient(model, base_url)` implementing the protocol (lazy imports of `openai` / `httpx` inside `__init__`).
  - `FakeLLMClient(responses: list[str] | Callable[[str,str], str])` for tests — returns queued/computed responses.
  - `get_client(settings) -> LLMClient` selecting by `settings.llm_provider` (`openai` or `ollama`); raises `RuntimeError` with a clear BYOK message if the provider is unset or unsupported, or if OpenAI key is missing.
  - `llm/models.py`: `list_models(provider, api_key, base_url) -> list[str]` — queries the provider to discover available models at runtime.
  - `cache.py`: `class DiskCache: get(key)->str|None; set(key,val)`; `cache_key(model, system, user)->str` (sha256). `CachedClient(inner, cache, model)` wraps any `LLMClient`.

- [ ] **Step 1: Failing tests**
```python
# tests/llm/test_client.py
from veritas.llm.client import FakeLLMClient, CachedClient
from veritas.llm.cache import DiskCache

def test_fake_returns_queued():
    c = FakeLLMClient(["hello"])
    assert c.complete("s", "u") == "hello"

def test_cache_avoids_second_call(tmp_path):
    calls = {"n": 0}
    def gen(s, u):
        calls["n"] += 1; return "R"
    inner = FakeLLMClient(gen)
    cached = CachedClient(inner, DiskCache(tmp_path), model="m")
    assert cached.complete("s", "u") == "R"
    assert cached.complete("s", "u") == "R"
    assert calls["n"] == 1
```
- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** `cache.py` then `client.py`. `OpenAIClient.complete` uses `openai.OpenAI(api_key=...).chat.completions.create(...)`; `OllamaClient.complete` posts to the Ollama local HTTP API (`/api/chat`). `get_client` maps provider to class, validates key for OpenAI, skips key check for Ollama. Add `llm/models.py` with `list_models()` for runtime model discovery.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `feat: provider-agnostic BYOK LLM client (OpenAI + Ollama) with model discovery and disk cache`.

---

## Task 6: Embeddings + vector store

**Files:**
- Create: `src/veritas/store/__init__.py`, `src/veritas/store/embeddings.py`, `src/veritas/store/vectors.py`
- Test: `tests/store/test_vectors.py`

**Interfaces:**
- Produces:
  - `class Embedder(Protocol): def embed(self, texts: list[str]) -> list[list[float]]: ...`
  - `SentenceTransformerEmbedder(model_name)` (lazy import); `FakeEmbedder(dim=16)` — deterministic hash-based vectors for tests.
  - `class VectorIndex: __init__(self, chroma_dir, embedder); add(self, ids: list[str], texts: list[str], metadatas: list[dict]); query(self, text: str, top_k: int, where: dict | None = None) -> list[tuple[str, float, dict]]` (returns `(id, distance, metadata)`), `count() -> int`.

- [ ] **Step 1: Failing test**
```python
# tests/store/test_vectors.py
from veritas.store.vectors import VectorIndex
from veritas.store.embeddings import FakeEmbedder

def test_add_and_query_excludes_via_where(tmp_path):
    idx = VectorIndex(tmp_path, FakeEmbedder())
    idx.add(["f1","f2"], ["revenue was 100 cr","gdp grew 6 percent"],
            [{"doc_id":"a"},{"doc_id":"b"}])
    res = idx.query("revenue 100 crore", top_k=1, where={"doc_id":"b"})
    assert res and res[0][2]["doc_id"] == "b"
```
- [ ] **Step 2: Run** → FAIL. 
- [ ] **Step 3: Implement** using `chromadb.PersistentClient`. Pass precomputed embeddings (call `embedder.embed`) so Chroma doesn't need its own model. `where` maps to Chroma metadata filter (used later to exclude same-doc facts via `{"doc_id": {"$ne": ...}}`).
- [ ] **Step 4: Run** → PASS. 
- [ ] **Step 5: Commit** — `feat: pluggable embeddings and persistent Chroma vector index`.

---

## Task 7: SQLite store

**Files:**
- Create: `src/veritas/store/db.py`
- Test: `tests/store/test_db.py`

**Interfaces:**
- Consumes: `Document, Fact, Edge, Job`.
- Produces `class Store:` with methods:
  - `__init__(db_path)`, `init_schema()`.
  - `upsert_document(doc) -> None`, `get_document_by_hash(hash) -> Document | None`, `list_documents() -> list[Document]`.
  - `add_facts(facts: list[Fact])`, `get_fact(id) -> Fact | None`, `list_facts(doc_id=None, subject=None, kind=None, min_conf=0.0, limit=200, offset=0) -> list[Fact]`, `all_facts() -> list[Fact]`.
  - `add_edges(edges: list[Edge])`, `list_edges(relation=None, doc_id=None) -> list[Edge]`, `get_edge(id)`.
  - `create_job(job)`, `update_job(id, **fields)`, `get_job(id)`.
  - `record_attributes(attrs: list[str])`, `attribute_vocab() -> list[tuple[str,int]]` (dynamic schema surfacing).
  - `counts() -> dict` (docs/facts/edges).

- [ ] **Step 1: Failing test**
```python
# tests/store/test_db.py
from veritas.store.db import Store
from veritas.models import Document, Fact, new_id

def _fact(doc="d1", attr="revenue"):
    return Fact(new_id("fact"), doc, "Co", attr, "100", "INR Cr", "FY24", [],
                "Revenue 100 Cr", 3, "Revenue FY24 100 INR Cr", "numerical", 0.9)

def test_idempotent_document(tmp_path):
    s = Store(tmp_path/"t.db"); s.init_schema()
    d = Document(new_id("doc"), "a.pdf", "hash123", 10, "done", "2026-01-01")
    s.upsert_document(d)
    assert s.get_document_by_hash("hash123").filename == "a.pdf"

def test_facts_and_vocab(tmp_path):
    s = Store(tmp_path/"t.db"); s.init_schema()
    s.add_facts([_fact(attr="revenue"), _fact(attr="profit")])
    s.record_attributes(["revenue","profit","revenue"])
    assert len(s.list_facts()) == 2
    vocab = dict(s.attribute_vocab())
    assert vocab["revenue"] >= 1
```
- [ ] **Step 2: Run** → FAIL. 
- [ ] **Step 3: Implement** with stdlib `sqlite3` (row factory → dataclass). Schema: tables `documents, facts, edges, jobs, attribute_vocab`. `scope_qualifiers` stored as JSON text. Use `INSERT OR REPLACE` keyed on id/hash for idempotency.
- [ ] **Step 4: Run** → PASS. 
- [ ] **Step 5: Commit** — `feat: SQLite store with idempotent docs, facts, edges, jobs, dynamic vocab`.

---

## Task 8: Fact extraction (LLM, domain-neutral)

**Files:**
- Create: `src/veritas/extraction/__init__.py`, `src/veritas/extraction/prompts.py`, `src/veritas/extraction/extractor.py`
- Test: `tests/extraction/test_extractor.py`

**Interfaces:**
- Consumes: `Chunk`, `Fact`, `LLMClient`.
- Produces:
  - `prompts.build_extraction_prompt(chunk_text: str) -> tuple[str, str]` (system, user). System instructs: extract atomic factual claims (numerical OR semantic), each with a verbatim `evidence_span` copied from the text, and the open-schema JSON fields; return a JSON array; extract nothing if no factual claim. **No domain words.**
  - `extractor.extract_from_chunk(chunk, llm) -> list[Fact]` — calls LLM, parses JSON, validates each row (must have non-empty `evidence_span` that is a substring of the chunk text — else drop and log), attaches `doc_id/page` from the chunk, assigns `id`, sets `confidence`.
  - `extractor.extract_facts(chunks, llm, concurrency=4) -> list[Fact]` — thread-pool over chunks.

- [ ] **Step 1: Failing test (mocked LLM)**
```python
# tests/extraction/test_extractor.py
import json
from veritas.models import Chunk
from veritas.llm.client import FakeLLMClient
from veritas.extraction.extractor import extract_from_chunk

def test_extracts_and_grounds():
    text = "Revenue was Rs 8,142 crore in FY24. The sky is blue."
    resp = json.dumps([
      {"subject":"Company","attribute":"revenue","value":"8142","unit":"INR Cr",
       "temporal_context":"FY24","scope_qualifiers":[],"evidence_span":"Revenue was Rs 8,142 crore in FY24",
       "claim_text":"Revenue in FY24 was 8142 INR Cr","fact_kind":"numerical","confidence":0.95}])
    c = Chunk("d1", 5, text, 0, len(text))
    facts = extract_from_chunk(c, FakeLLMClient([resp]))
    assert len(facts) == 1
    assert facts[0].page == 5 and facts[0].doc_id == "d1"
    assert facts[0].evidence_span in text  # grounding enforced

def test_drops_ungrounded():
    text = "Revenue was 100 crore."
    resp = json.dumps([{"subject":"X","attribute":"y","value":"z","unit":"",
       "temporal_context":"","scope_qualifiers":[],"evidence_span":"NOT IN TEXT",
       "claim_text":"c","fact_kind":"semantic","confidence":0.5}])
    facts = extract_from_chunk(Chunk("d1",1,text,0,len(text)), FakeLLMClient([resp]))
    assert facts == []
```
- [ ] **Step 2: Run** → FAIL. 
- [ ] **Step 3: Implement** prompt + extractor. Grounding check: normalize whitespace before substring test; allow fuzzy match (strip/collapse spaces). Robust JSON parse (extract first `[...]`).
- [ ] **Step 4: Run** → PASS. 
- [ ] **Step 5: Commit** — `feat: domain-neutral LLM fact extraction with evidence grounding`.

---

## Task 9: Normalization

**Files:**
- Create: `src/veritas/normalization/__init__.py`, `src/veritas/normalization/normalize.py`
- Test: `tests/normalization/test_normalize.py`

**Interfaces:**
- Consumes: `Fact`.
- Produces: `normalize_fact(fact: Fact) -> Fact` filling `normalized_value` (float), `normalized_unit` (canonical, e.g. Indian `crore/lakh` → absolute count; `%` kept), `period_start`/`period_end` (ISO dates from `FY24`, `Q4FY24`, `CY2024`, `as of <date>`), `canonical_subject` (lowercased, trimmed, punctuation-stripped). Helpers: `parse_number(str)->float|None`, `parse_period(str)->tuple[str|None,str|None]`, `canonical_unit(str)->str`. Uses `pint` for physical units; custom map for crore/lakh/mn/bn. Indian FY = Apr–Mar.

- [ ] **Step 1: Failing test**
```python
def test_number_and_period():
    from veritas.normalization.normalize import parse_number, parse_period
    assert parse_number("8,142") == 8142.0
    assert parse_number("1.2") == 1.2
    assert parse_period("FY24") == ("2023-04-01","2024-03-31")
    assert parse_period("Q4FY24") == ("2024-01-01","2024-03-31")

def test_normalize_crore():
    from veritas.models import Fact
    from veritas.normalization.normalize import normalize_fact
    f = Fact("i","d","Co","revenue","8,142","INR Cr","FY24",[], "ev",1,"c","numerical",0.9)
    out = normalize_fact(f)
    assert out.normalized_value == 8142 * 10**7  # crore -> absolute
    assert out.period_start == "2023-04-01"
```
- [ ] **Step 2: Run** → FAIL. 
- [ ] **Step 3: Implement** parsers + `normalize_fact`. Keep general (works on kg, %, persons, currency); unknown units pass through unchanged with `normalized_value` = parsed number.
- [ ] **Step 4: Run** → PASS. 
- [ ] **Step 5: Commit** — `feat: general numeric/date/unit/entity normalization`.

---

## Task 10: Relationship engine (RAG core)

**Files:**
- Create: `src/veritas/relate/__init__.py`, `src/veritas/relate/prompts.py`, `src/veritas/relate/relationships.py`
- Test: `tests/relate/test_relationships.py`

**Interfaces:**
- Consumes: `Fact`, `Edge`, `Store`, `VectorIndex`, `LLMClient`.
- Produces:
  - `prompts.build_adjudication_prompt(fact_a: Fact, fact_b: Fact) -> tuple[str,str]` — domain-neutral: given two grounded claims with their contexts, classify relation ∈ `corroborate|contradict|reconcilable|unrelated`, give `reasoning`, and `reconciling_dimension` ∈ `time|scope|unit|none`. Return JSON object.
  - `relationships.adjudicate(fact_a, fact_b, llm) -> Edge | None`.
  - `relationships.find_relationships(fact: Fact, store, index, llm, top_k=8) -> list[Edge]` — retrieve top_k similar facts from OTHER docs (`where={"doc_id":{"$ne":fact.doc_id}}`), adjudicate each, keep non-`unrelated` edges (dedup by fact-pair).

- [ ] **Step 1: Failing test**
```python
# tests/relate/test_relationships.py
import json
from veritas.models import Fact, new_id
from veritas.llm.client import FakeLLMClient
from veritas.relate.relationships import adjudicate

def _f(doc, val, period): 
    return Fact(new_id("f"), doc, "Co", "revenue", val, "INR Cr", period, [], f"rev {val}", 1,
                f"Revenue {period} {val}", "numerical", 0.9)

def test_adjudicate_contradiction():
    resp = json.dumps({"relation":"contradict","reasoning":"Same period, different value",
                       "reconciling_dimension":"none","confidence":0.8})
    e = adjudicate(_f("a","100","FY24"), _f("b","200","FY24"), FakeLLMClient([resp]))
    assert e.relation == "contradict"

def test_adjudicate_reconcilable():
    resp = json.dumps({"relation":"reconcilable","reasoning":"Different periods",
                       "reconciling_dimension":"time","confidence":0.85})
    e = adjudicate(_f("a","100","FY23"), _f("b","200","FY24"), FakeLLMClient([resp]))
    assert e.relation == "reconcilable" and e.reconciling_dimension == "time"
```
- [ ] **Step 2: Run** → FAIL. 
- [ ] **Step 3: Implement** prompts + `adjudicate` + `find_relationships`. Include normalized values/periods in the prompt to help the model. Skip pairs already adjudicated (pair key = sorted ids).
- [ ] **Step 4: Run** → PASS. 
- [ ] **Step 5: Commit** — `feat: RAG relationship engine (corroborate/contradict/reconcilable)`.

---

## Task 11: Canonicalization (dedup extension)

**Files:**
- Create: `src/veritas/relate/canonicalize.py`
- Test: `tests/relate/test_canonicalize.py`

**Interfaces:**
- Consumes: `Fact`, `Edge`.
- Produces: `canonical_clusters(facts: list[Fact], edges: list[Edge]) -> list[list[str]]` — union-find over `corroborate` edges; returns clusters of fact ids (each singleton fact is its own cluster). `canonical_label(cluster_facts) -> str` picks the highest-confidence `claim_text`.

- [ ] **Step 1: Failing test**
```python
def test_union_find_clusters():
    from veritas.models import Edge, new_id
    from veritas.relate.canonicalize import canonical_clusters
    facts_ids = ["a","b","c"]
    class F:  # minimal stand-in
        def __init__(self,i): self.id=i
    facts=[F("a"),F("b"),F("c")]
    edges=[Edge(new_id("e"),"a","b","corroborate","same","none",0.9)]
    clusters = canonical_clusters(facts, edges)
    assert sorted(sorted(c) for c in clusters) == [["a","b"],["c"]]
```
- [ ] **Step 2: Run** → FAIL. 
- [ ] **Step 3: Implement** union-find.
- [ ] **Step 4: Run** → PASS. 
- [ ] **Step 5: Commit** — `feat: fact canonicalization via corroboration clustering`.

---

## Task 12: Ingestion pipeline + jobs (incremental, idempotent)

**Files:**
- Create: `src/veritas/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `content_hash(path) -> str` (sha256 of bytes).
  - `class Pipeline: __init__(self, store, index, llm, settings)`.
  - `ingest_document(self, path, filename, job_id=None, progress_cb=None) -> Document` — steps: hash → if existing doc with hash return it (idempotent no-op) → create Document(status=parsing) → parse → chunk → extract (concurrency) → normalize → persist facts + embeddings + attribute vocab → for each new fact `find_relationships` against existing index (which already contains prior docs) → persist edges → status=done. Emits progress via `progress_cb(stage, pct, msg)` and job updates. **Embeddings for the new doc are added AFTER relationship search so a doc never relates to itself.**

- [ ] **Step 1: Failing integration test (fakes)**
```python
# tests/test_pipeline.py — uses FakeLLMClient + FakeEmbedder, two tiny PDFs
```
Build two PDFs: doc A "Revenue was 100 crore in FY24." doc B "Revenue was 200 crore in FY24." Program `FakeLLMClient` to return one extraction per chunk and a `contradict` adjudication. Assert: ingest A then B → 2 facts, ≥1 contradict edge, re-ingest B (same bytes) is a no-op (counts unchanged).
- [ ] **Step 2: Run** → FAIL. 
- [ ] **Step 3: Implement** `Pipeline`. Order matters (relate before adding new doc's vectors). Wrap per-stage errors → job status `error` with message.
- [ ] **Step 4: Run** → PASS. 
- [ ] **Step 5: Commit** — `feat: incremental idempotent ingestion pipeline with progress`.

---

## Task 13: Grounded Q&A (extension)

**Files:**
- Create: `src/veritas/qa.py`
- Test: `tests/test_qa.py`

**Interfaces:**
- Produces: `answer(question: str, store, index, llm, top_k=8) -> dict` with keys `answer`, `citations: list[{fact_id, doc_id, page, evidence_span}]`. Retrieves relevant facts across ALL docs, builds a prompt that MUST cite fact ids, returns parsed answer + resolved citations. Refuses (answer="Not enough grounded evidence.") when retrieval is empty.

- [ ] **Step 1: Failing test** (fake retrieval + fake LLM returning an answer citing `f1`). Assert citation resolves to the stored fact's page/evidence.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement. Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `feat: grounded, cited Q&A over the knowledge layer`.

---

## Task 14: FastAPI app + schemas

**Files:**
- Create: `src/veritas/api/__init__.py`, `src/veritas/api/schemas.py`, `src/veritas/api/app.py`
- Test: `tests/api/test_app.py`

**Interfaces:**
- Produces a FastAPI `app` with routes from spec §5. Uses a module-level `Pipeline`/`Store`/`VectorIndex` built from `get_settings()`, but allow dependency-injection override in tests (a `build_app(store, index, pipeline)` factory). `POST /documents` saves upload, creates a Job, runs `ingest_document` in `BackgroundTasks`. Serves `web/` static files at `/`.

- [ ] **Step 1: Failing test** with `TestClient` + injected fakes:
```python
def test_upload_and_list_facts(client_with_fakes, tiny_pdf_bytes):
    r = client_with_fakes.post("/documents", files={"file":("a.pdf", tiny_pdf_bytes, "application/pdf")})
    assert r.status_code == 200 and "job_id" in r.json()
    # background task runs inline under TestClient
    facts = client_with_fakes.get("/facts").json()
    assert isinstance(facts, list)
def test_health(client_with_fakes):
    assert client_with_fakes.get("/health").json()["status"] == "ok"
```
- [ ] **Step 2: Run** → FAIL. 
- [ ] **Step 3: Implement** schemas (pydantic) + routes: `/config/llm`(GET), `/config/llm/connect`(POST), `/config/llm/select`(POST), `/documents`(POST,GET), `/jobs/{id}`, `/facts`(GET, filters), `/facts/{id}`, `/relationships`(GET), `/relationships/{id}`, `/cases`, `/ask`(POST), `/export`, `/health`. Runtime LLM config stored in `app.state` (never persisted). CORS enabled. Static mount for UI.
- [ ] **Step 4: Run** → PASS. 
- [ ] **Step 5: Commit** — `feat: FastAPI backend exposing facts, relationships, cases, ask, export`.

---

## Task 15: `/cases` + `/export` logic

**Files:**
- Modify: `src/veritas/api/app.py`; Create: `src/veritas/cases.py`
- Test: `tests/test_cases.py`

**Interfaces:**
- Produces `cases.build_cases(store) -> dict` returning up to one example each of `corroborate`, `contradict`, `reconcilable`, plus a `failure` slot (lowest-confidence fact or flagged extraction) — each with both facts' evidence + reasoning. `export_layer(store) -> dict` (facts+edges) and CSV serializer.

- [ ] **Step 1: Failing test** — seed store with one edge of each type; assert `build_cases` returns those ids with evidence populated.
- [ ] **Step 2: Run→FAIL. Step 3: Implement. Step 4: Run→PASS.**
- [ ] **Step 5: Commit** — `feat: four-cases builder and layer export (JSON/CSV)`.

---

## Task 16: Frontend SPA

**Files:**
- Create: `src/veritas/web/index.html`, `src/veritas/web/app.js`, `src/veritas/web/styles.css`
- Test: `tests/web/test_static_served.py` (asserts `/` returns HTML and `/facts` reachable — behavior smoke only)

**Interfaces:**
- Consumes the API. Views: **Upload** (drag-drop, shows job progress via polling `/jobs/{id}`), **Facts** (table with subject/attribute/value/period/confidence + click → evidence panel showing verbatim span & page), **Relationships** (filter by type; each row shows both claims, reasoning, reconciling dimension), **Four Cases** (the showcase), **Ask** (question box → cited answer). Clean modern CSS (system font stack, cards, subtle color per relation: green=corroborate, red=contradict, amber=reconcilable). No framework/build step.

- [ ] **Step 1: Failing test** — `GET /` returns 200 + `text/html`.
- [ ] **Step 2: Run→FAIL. Step 3: Implement HTML/CSS/JS. Step 4: Run→PASS.**
- [ ] **Step 5: Manual check** — screenshot in demo. 
- [ ] **Step 6: Commit** — `feat: lightweight web UI for upload, facts, relationships, cases, ask`.

---

## Task 17: Demo builder + evaluation harness

**Files:**
- Create: `scripts/build_demo.py`, `eval/gold.json`, `eval/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- `scripts/build_demo.py` — ingests the six starter PDFs (both folders) into a fresh layer, prints the four cases with evidence to stdout and writes `docs/demo-cases.md`. Requires a real key (documented); guarded so it errors clearly without one.
- `eval/evaluate.py` — `evaluate(store, gold) -> dict` computing extraction precision (sampled) and relationship accuracy against `eval/gold.json` (a small hand-checked set of expected cross-doc relations, referenced by claim text substrings, not ids). 
- Test uses a fake gold + seeded store to verify metric math.

- [ ] **Step 1: Failing test** for `evaluate()` metric computation with a tiny synthetic store+gold.
- [ ] **Step 2: Run→FAIL. Step 3: Implement. Step 4: Run→PASS.**
- [ ] **Step 5: Commit** — `feat: demo builder and evaluation harness with gold set`.

---

## Task 18: Documentation (README + stand-out + additional-features)

**Files:**
- Create: `README.md`, `docs/stand-out-criteria.md`, `docs/additional-features.md`
- (No test.)

**Interfaces:** N/A — prose deliverables required by the assignment and by the user.

- [ ] **Step 1: `README.md`** with the assignment's required sections: **Setup and Run Instructions** (venv, `pip install -e .`, `.env` BYOK, `uvicorn veritas.api.app:app`, open UI, connect provider in UI), **Video Demo** (placeholder link), **Approach** (architecture, RAG core, provider-agnostic LLM setup, key decisions & trade-offs), **Limitations and Next Steps** (OCR, distributed store, recall bounds), **Additional Notes**. Include an architecture diagram (ASCII) and the four cases summary.
- [ ] **Step 2: `docs/stand-out-criteria.md`** — table mapping every *What We Are Looking For* item AND every *Brownie Point* (+ extensions) → the component/task/test that implements it, with evidence.
- [ ] **Step 3: `docs/additional-features.md`** — future features beyond phase 1 (OCR, active-learning on failures, contradiction severity scoring, graph visualization, multi-hop reasoning, streaming ingestion, auth/multi-tenant), each with rationale + rough effort.
- [ ] **Step 4: Commit** — `docs: README, stand-out criteria, and additional-features roadmap`.

---

## Self-Review

**Spec coverage:** parse (T3), chunk (T4), extract+ground (T8), normalize (T9), store+vectors (T6,T7), RAG relate (T10), canonicalize (T11), incremental/idempotent pipeline (T12), Q&A (T13), API (T14,T15), UI (T16), brownie points — large (T3 streaming/T8 concurrency), many (T6/T10 retrieval-scoped), dynamic schema (T7 vocab/T8 open schema), incremental (T12) — all covered. Uncertainty/confidence threaded through T8/T10 and surfaced T14/T16. Four cases (T15/T17). Docs incl. stand-out + additional-features (T18). Generalization proven by off-category test in T12/T17 datasets note.

**Placeholder scan:** Frontend (T16) and demo (T17) intentionally describe behavior + smoke tests rather than full JS/HTML source (UI code is verbose and low-risk); every backend task carries real test + implementation code. Acceptable per plan granularity.

**Type consistency:** `Fact`/`Edge` field names consistent across T2→T18; `find_relationships`, `adjudicate`, `ingest_document`, `VectorIndex.query`, `Store.*` signatures match across consumers.

---

## Implementation Order

Independent early tasks (T3 parser, T4 chunker, T5 LLM client + model discovery, T6 vectors, T7 store) have no cross-dependencies and can be worked on in any order. T8–T12 depend on the shared `models.py` contract established in T2. T13–T17 depend on the completed pipeline. T18 (docs) is last.
