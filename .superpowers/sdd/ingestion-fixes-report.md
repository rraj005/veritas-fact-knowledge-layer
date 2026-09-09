# Ingestion Fixes Report

## Root Causes

1. **Concurrent SQLite access** — `src/veritas/store/db.py` used a single shared `sqlite3.Connection` with `check_same_thread=False` but had no locking. Multiple threads (API reads + background ingestion workers) could corrupt the connection state or hit "database is locked" errors.

2. **Overlapping ingestion workers** — `POST /documents` scheduled `Pipeline.ingest_document` via FastAPI's `BackgroundTasks`, which runs tasks in the anyio thread pool. Two concurrent uploads would run two ingestion pipelines simultaneously, racing on the shared SQLite connection and the Chroma vector index. The second job typically got stuck at "queued" or silently failed.

3. **No Documents list view** — The UI had no way to browse previously ingested PDFs; there was no "Documents" nav entry or page.

4. **Single-file upload only** — The `<input type="file">` had no `multiple` attribute; users could only select one PDF at a time and had no per-file progress visibility.

5. **No live refresh** — After ingestion completed, the user had to manually navigate away and back to Facts/Relationships/Cases to see results.

6. **Missing `fact_count` on documents** — `GET /documents` returned document metadata without the number of facts extracted, making it impossible to know if ingestion was productive.

---

## Fixes Applied

### Backend

**1. Store thread safety** (`src/veritas/store/db.py`)
Already had `threading.RLock` and `with self._lock` guards on all public methods — confirmed complete. No changes needed.

**2. Serialized ingestion queue** (`src/veritas/api/app.py`)
- Removed `BackgroundTasks` usage from `POST /documents`.
- At `build_app(...)` time: create `app.state.ingest_queue = queue.Queue()` and start one daemon worker thread (`_ingest_worker`) that loops, pulling `(upload_path, filename, job_id, llm)` tuples, building a per-job `Pipeline`, calling `ingest_document`, catching and logging any uncaught exception, and calling `queue.task_done()`.
- `POST /documents` now: saves file, creates Job(status="queued"), puts the work item on the queue, returns immediately.
- Only one ingestion runs at a time; queued uploads are processed in order with no concurrent shared-resource access.

**3. `fact_count` in documents endpoint** (`src/veritas/api/schemas.py`, `src/veritas/api/app.py`)
- Added `fact_count: int = 0` to `DocumentResponse`.
- `GET /documents` handler now calls `store.fact_count(d.id)` for each doc and includes it in the response.

### Frontend

**4. Multiple-file upload** (`src/veritas/web/index.html`, `app.js`, `styles.css`)
- Added `multiple` attribute to `<input type="file">`.
- `initUpload()` now maintains an in-memory queue of `{file, row}` items and drains it sequentially: upload one file → poll job to done/error → upload next.
- Each file gets its own status row in `#upload-list` with filename, progress bar, percentage, and stage text.
- Drag-and-drop accepts a `FileList` and enqueues all PDF files.

**5. Documents view** (`src/veritas/web/index.html`, `app.js`, `styles.css`)
- Added "Documents" sidebar nav button and `#view-documents` section.
- `loadDocuments()` calls `GET /documents` and renders a table: filename, pages, colored status badge, fact count (with a zero-fact note for done docs with 0 facts), ingestion date.
- Friendly empty state when no documents exist.

**6. Live refresh after ingestion** (`app.js`)
- `reloadActiveDataView()` helper switches on `currentView` and calls `loadFacts()`, `loadRelationships()`, or `loadCases()` as appropriate.
- `refreshDocumentsIfVisible()` calls `loadDocuments()` only when the Documents view is active.
- Both helpers are called inside `processFile()` when a job reaches "done".
- `refreshHealth()` is also called so top-bar counts update immediately.

**7. Zero-fact clarity** (`app.js`, styles.css)
- In the upload list: when a doc completes with `fact_count === 0`, the stage message reads "Processed — 0 facts extracted (model returned no structured facts; try another model)" in amber.
- In the Documents table: a zero-fact done doc shows "0 — no facts extracted (try another model)" in amber instead of a plain "0".

**XSS safety** preserved throughout: all user/server text is written via `textContent`, `createTextNode`, or `txt()`. No `innerHTML` of server data.

---

## New Tests

File: `tests/api/test_app.py`

- `test_two_sequential_uploads_both_ingest`: Uploads two PDFs with distinct content hashes sequentially, waits for each job to complete via `ingest_queue.join()`, then asserts both jobs are "done", `GET /documents` lists both docs as "done", and `GET /facts` returns facts from BOTH doc_ids. Regression test for the stuck-second-upload bug.

- `test_documents_reports_fact_count`: Uploads a PDF, waits for ingestion, then asserts `GET /documents` returns at least one document with `fact_count >= 1`.

Helper `_wait_ingest(client)` added (calls `client.app.state.ingest_queue.join()`). Existing tests that upload and then immediately check facts (`test_upload_and_list_facts`, `test_export_csv`) updated to call `_wait_ingest()` before assertions.

---

## Test Results

```
186 passed, 1 warning in ~14s
(184 pre-existing + 2 new)
```

## Ruff Results

```
All checks passed!
```

## Commit Hashes

See git log — commits created after this report.

---

## Fix Round 1

### Findings addressed

**Finding 1 (Important) — duplicate upload job stuck at "queued"**
`src/veritas/pipeline.py`: In `ingest_document()`, the idempotency early-return path now updates the job row to `status="done"`, `progress=1.0`, `doc_id=existing.id` before returning, so duplicate uploads no longer leave the poller spinning forever.

**Finding 2 (Minor) — test consistency**
`tests/api/test_app.py`: Replaced the two direct `client.app.state.ingest_queue.join()` calls in `test_two_sequential_uploads_both_ingest` with the existing `_wait_ingest(client)` helper.

**Finding 3 (Minor) — worker import**
`src/veritas/api/app.py`: Moved `from veritas.pipeline import Pipeline` out of the `while True` body of `_ingest_worker` to function scope above the loop; import now happens once per worker thread instead of once per job.

### New test added
`tests/api/test_app.py::test_duplicate_upload_job_reaches_done`: Uploads the same PDF bytes twice (same content hash). Asserts the second job reaches `status="done"` and `progress=1.0`, and that no extra document or fact records are created.

### Test results

```
187 passed, 1 warning in 15.48s
(186 pre-existing + 1 new)
```

### Ruff results

```
All checks passed!
```
