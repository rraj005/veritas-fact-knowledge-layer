/**
 * Veritas — Fact Knowledge Layer SPA
 * Vanilla JS, no framework, no build step.
 * Communicates with the FastAPI backend via fetch().
 */

"use strict";

// ---------------------------------------------------------------------------
// Escape helper — never use innerHTML with untrusted text
// ---------------------------------------------------------------------------
function esc(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

// ---------------------------------------------------------------------------
// Text-to-DOM: safely set text content, return the element
// ---------------------------------------------------------------------------
function txt(el, text) {
  el.textContent = String(text ?? "");
  return el;
}

// ---------------------------------------------------------------------------
// createElement helper
// ---------------------------------------------------------------------------
function el(tag, { cls = "", attrs = {}, children = [] } = {}) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const c of children) {
    if (typeof c === "string") node.appendChild(document.createTextNode(c));
    else if (c) node.appendChild(c);
  }
  return node;
}

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------
// Backend base URL. Defaults to same-origin (empty string) so the app works
// when the FastAPI backend serves this UI directly. For a split deployment
// (e.g. frontend on Netlify, backend on a Hugging Face Space), set
// window.VERITAS_API_BASE in config.js to the backend URL.
const API_BASE = (window.VERITAS_API_BASE || "").replace(/\/+$/, "");

const API = {
  async get(path) {
    const r = await fetch(API_BASE + path);
    if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${path}`);
    return r.json();
  },

  async post(path, body) {
    const opts = { method: "POST" };
    if (body instanceof FormData) {
      opts.body = body;
    } else {
      opts.headers = { "Content-Type": "application/json" };
      opts.body = JSON.stringify(body);
    }
    const r = await fetch(API_BASE + path, opts);
    if (!r.ok) {
      const msg = await r.text().catch(() => r.statusText);
      throw new Error(`${r.status} — ${msg}`);
    }
    return r.json();
  },
};

// ---------------------------------------------------------------------------
// Health bar + live counts
// ---------------------------------------------------------------------------
async function refreshHealth() {
  try {
    const h = await API.get("/health");
    const counts = h.counts || {};
    setCount("count-docs", counts.docs ?? 0);
    setCount("count-facts", counts.facts ?? 0);
    setCount("count-edges", counts.edges ?? 0);
  } catch {
    // silently ignore if backend not reachable yet
  }
}

function setCount(id, n) {
  const el = document.getElementById(id);
  if (el) el.textContent = n;
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
const views = {};       // id → { el, onEnter }
let currentView = null;

function registerView(id, onEnter) {
  const elem = document.getElementById(id);
  if (!elem) return;
  views[id] = { el: elem, onEnter: onEnter || (() => {}) };
}

function showView(id) {
  if (currentView === id) return;
  if (currentView) {
    const prev = views[currentView];
    if (prev) prev.el.classList.remove("active");
  }
  const nav = views[id];
  if (!nav) return;
  currentView = id;
  nav.el.classList.add("active");
  nav.onEnter();

  // Update nav buttons
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.view === id);
  });
}

// ---------------------------------------------------------------------------
// Relation badge helper
// ---------------------------------------------------------------------------
function relationBadge(relation) {
  const map = {
    corroborate:  ["badge-corroborate",  "✓ Corroborate"],
    contradict:   ["badge-contradict",   "✗ Contradict"],
    reconcilable: ["badge-reconcilable", "⟳ Reconcilable"],
    unrelated:    ["badge-unrelated",    "· Unrelated"],
  };
  const [cls, label] = map[relation] || ["badge-unrelated", relation];
  const span = el("span", { cls: `badge ${cls}` });
  span.textContent = label;
  return span;
}

// ---------------------------------------------------------------------------
// Confidence display helpers
// ---------------------------------------------------------------------------
function confClass(c) {
  if (c >= 0.8) return "conf-high";
  if (c >= 0.5) return "conf-mid";
  return "conf-low";
}

function confBar(conf) {
  const pct = Math.round((conf ?? 0) * 100);
  const wrap = el("div", { cls: "conf-bar-wrap" });
  const track = el("div", { cls: "conf-bar" });
  const fill = el("div", { cls: "conf-bar-fill" });
  fill.style.width = pct + "%";
  const color = conf >= 0.8 ? "var(--green)" : conf >= 0.5 ? "var(--amber)" : "var(--red)";
  fill.style.background = color;
  track.appendChild(fill);
  const label = el("span", { cls: "cell-mono" });
  label.textContent = pct + "%";
  wrap.append(track, label);
  return wrap;
}

// ---------------------------------------------------------------------------
// Period display
// ---------------------------------------------------------------------------
function formatPeriod(fact) {
  if (fact.temporal_context) return fact.temporal_context;
  if (fact.period_start) return fact.period_start.slice(0, 7);
  return "—";
}

// ---------------------------------------------------------------------------
// VIEW: Upload — multiple-file support, sequential processing
// ---------------------------------------------------------------------------
function initUpload() {
  const zone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const uploadList = document.getElementById("upload-list");

  // Queue of pending File objects; only one upload+poll runs at a time.
  let _uploadQueue = [];
  let _uploading = false;

  // Create a status row for a single file and return its controller.
  function createUploadRow(filename) {
    const item = el("div", { cls: "upload-item" });

    const header = el("div", { cls: "upload-item-header" });
    const nameEl = el("span", { cls: "upload-item-filename" });
    nameEl.textContent = filename;
    const pctEl = el("span", { cls: "upload-item-pct" });
    pctEl.textContent = "0%";
    header.append(nameEl, pctEl);

    const track = el("div", { cls: "upload-item-bar-track" });
    const fill = el("div", { cls: "upload-item-bar-fill" });
    track.appendChild(fill);

    const stageEl = el("span", { cls: "upload-item-stage" });
    stageEl.textContent = "Queued";

    item.append(header, track, stageEl);
    uploadList.appendChild(item);

    return {
      setProgress(pct, stageTxt, cls) {
        fill.style.width = pct + "%";
        pctEl.textContent = pct + "%";
        stageEl.textContent = stageTxt;
        stageEl.className = "upload-item-stage" + (cls ? " " + cls : "");
      },
    };
  }

  // Process one file: upload → poll job to completion → call cb.
  async function processFile(file, row) {
    if (file.type !== "application/pdf" && !file.name.endsWith(".pdf")) {
      row.setProgress(0, "Skipped — not a PDF", "error");
      return;
    }
    const fd = new FormData();
    fd.append("file", file);

    row.setProgress(0, "Uploading…", "");

    let jobId;
    try {
      const res = await API.post("/documents", fd);
      if (!res.job_id) throw new Error("No job_id returned");
      jobId = res.job_id;
    } catch (err) {
      row.setProgress(0, "Upload failed: " + err.message, "error");
      return;
    }

    // Poll until done or error.
    await new Promise((resolve) => {
      const timer = setInterval(async () => {
        try {
          const job = await API.get(`/jobs/${jobId}`);
          const pct = Math.round((job.progress ?? 0) * 100);

          if (job.status === "done") {
            clearInterval(timer);
            const factCount = job.doc_id
              ? await getFactCount(job.doc_id)
              : null;
            if (factCount === 0) {
              row.setProgress(
                100,
                "Processed — 0 facts extracted (model returned no structured facts; try another model)",
                "warn"
              );
            } else {
              const factTxt = factCount != null ? ` (${factCount} facts)` : "";
              row.setProgress(100, "Ingestion complete" + factTxt + " ✓", "done");
            }
            // Live refresh: counts + documents list + active data view.
            refreshHealth();
            refreshDocumentsIfVisible();
            reloadActiveDataView();
            resolve();
          } else if (job.status === "error") {
            clearInterval(timer);
            row.setProgress(pct, "Error: " + (job.message || "unknown"), "error");
            resolve();
          } else {
            row.setProgress(pct, job.message || job.status, "");
          }
        } catch (err) {
          clearInterval(timer);
          row.setProgress(0, "Poll error: " + err.message, "error");
          resolve();
        }
      }, 1500);
    });
  }

  // Helper: fetch fact count for a doc_id.
  async function getFactCount(docId) {
    try {
      const docs = await API.get("/documents");
      const doc = docs.find((d) => d.id === docId);
      return doc ? (doc.fact_count ?? null) : null;
    } catch {
      return null;
    }
  }

  // Drain the upload queue one file at a time.
  async function drainQueue() {
    if (_uploading) return;
    _uploading = true;
    while (_uploadQueue.length) {
      const { file, row } = _uploadQueue.shift();
      await processFile(file, row);
    }
    _uploading = false;
  }

  // Enqueue a list of files (FileList or array).
  function enqueueFiles(files) {
    for (const file of files) {
      const row = createUploadRow(file.name);
      _uploadQueue.push({ file, row });
    }
    drainQueue();
  }

  // Drag-and-drop
  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    const files = Array.from(e.dataTransfer.files).filter(
      (f) => f.type === "application/pdf" || f.name.endsWith(".pdf")
    );
    if (files.length) enqueueFiles(files);
  });

  // File picker (multiple)
  fileInput.addEventListener("change", () => {
    const files = Array.from(fileInput.files);
    if (files.length) enqueueFiles(files);
    fileInput.value = "";
  });

  // Click to open picker (exclude the input itself)
  zone.addEventListener("click", (e) => {
    if (e.target === fileInput) return;
    fileInput.click();
  });

  // Keyboard activation: Enter or Space triggers the file picker
  zone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });
}

// ---------------------------------------------------------------------------
// VIEW: Documents
// ---------------------------------------------------------------------------
async function loadDocuments() {
  const container = document.getElementById("docs-table-wrap");
  container.innerHTML = "";
  const loading = el("div", { cls: "state-loading" });
  loading.textContent = "Loading documents…";
  container.appendChild(loading);

  try {
    const docs = await API.get("/documents");
    renderDocumentsTable(container, docs);
  } catch (err) {
    container.innerHTML = "";
    const e = el("div", { cls: "state-error" });
    e.textContent = "Failed to load documents: " + err.message;
    container.appendChild(e);
  }
}

function renderDocumentsTable(container, docs) {
  container.innerHTML = "";

  if (!docs.length) {
    const empty = el("div", { cls: "state-empty" });
    empty.textContent = "No documents ingested yet. Upload a PDF to get started.";
    container.appendChild(empty);
    return;
  }

  const wrap = el("div", { cls: "table-wrap" });
  const table = el("table");
  const thead = el("thead");
  const headerRow = el("tr");

  for (const h of ["Filename", "Pages", "Status", "Facts", "Ingested"]) {
    const th = el("th");
    th.textContent = h;
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = el("tbody");
  for (const doc of docs) {
    const tr = el("tr");

    // Filename
    const tdName = el("td");
    const nameSpan = el("span", { cls: "truncate" });
    nameSpan.textContent = doc.filename;
    tdName.appendChild(nameSpan);
    tr.appendChild(tdName);

    // Pages
    const tdPages = el("td", { cls: "cell-mono" });
    tdPages.textContent = doc.num_pages != null ? String(doc.num_pages) : "—";
    tr.appendChild(tdPages);

    // Status badge
    const tdStatus = el("td");
    const badge = el("span", { cls: "status-badge status-" + (doc.status || "queued") });
    badge.textContent = doc.status || "queued";
    tdStatus.appendChild(badge);
    tr.appendChild(tdStatus);

    // Fact count
    const tdFacts = el("td");
    if (doc.status === "done" && doc.fact_count === 0) {
      const zeroNote = el("span");
      zeroNote.style.color = "var(--amber)";
      zeroNote.style.fontSize = "var(--text-xs)";
      zeroNote.textContent = "0 — no facts extracted (try another model)";
      tdFacts.appendChild(zeroNote);
    } else {
      tdFacts.textContent = doc.fact_count != null ? String(doc.fact_count) : "—";
    }
    tr.appendChild(tdFacts);

    // Created at
    const tdDate = el("td", { cls: "cell-mono" });
    const dateStr = doc.created_at ? doc.created_at.slice(0, 19).replace("T", " ") : "—";
    tdDate.textContent = dateStr;
    tr.appendChild(tdDate);

    tbody.appendChild(tr);
  }

  table.appendChild(tbody);
  wrap.appendChild(table);
  container.appendChild(wrap);
}

// Refresh the documents view only if it's currently visible.
function refreshDocumentsIfVisible() {
  if (currentView === "view-documents") {
    loadDocuments();
  }
}

// ---------------------------------------------------------------------------
// Live-refresh helper: reload the active data view after ingestion completes.
// ---------------------------------------------------------------------------
function reloadActiveDataView() {
  switch (currentView) {
    case "view-facts":         loadFacts();         break;
    case "view-relationships": loadRelationships(); break;
    case "view-cases":         loadCases();         break;
    default:                                        break;
  }
}

// ---------------------------------------------------------------------------
// VIEW: Facts
// ---------------------------------------------------------------------------
let factsData = [];
let activeFactId = null;

async function loadFacts() {
  const container = document.getElementById("facts-table-wrap");
  container.innerHTML = "";
  const loading = el("div", { cls: "state-loading" });
  loading.textContent = "Loading facts…";
  container.appendChild(loading);

  // Read filters
  const docFilter  = document.getElementById("fact-filter-doc").value.trim();
  const kindFilter = document.getElementById("fact-filter-kind").value;
  const confFilter = parseFloat(document.getElementById("fact-filter-conf").value) || 0;

  const params = new URLSearchParams();
  if (docFilter)  params.set("doc_id", docFilter);
  if (kindFilter) params.set("kind", kindFilter);
  if (confFilter) params.set("min_conf", confFilter);
  params.set("limit", "500");

  try {
    factsData = await API.get(`/facts?${params}`);
    renderFactsTable(container);
  } catch (err) {
    container.innerHTML = "";
    const e = el("div", { cls: "state-error" });
    e.textContent = "Failed to load facts: " + err.message;
    container.appendChild(e);
  }
}

function renderFactsTable(container) {
  container.innerHTML = "";
  if (!factsData.length) {
    const empty = el("div", { cls: "state-empty" });
    empty.textContent = "No facts found. Upload a document to get started.";
    container.appendChild(empty);
    return;
  }

  const wrap = el("div", { cls: "table-wrap" });
  const table = el("table");
  const thead = el("thead");
  const headerRow = el("tr");

  for (const h of ["Subject", "Attribute", "Value / Unit", "Period", "Kind", "Confidence"]) {
    const th = el("th");
    th.textContent = h;
    headerRow.appendChild(th);
  }
  thead.appendChild(headerRow);
  table.appendChild(thead);

  const tbody = el("tbody");
  for (const fact of factsData) {
    const tr = el("tr");
    tr.dataset.factId = fact.id;

    const cells = [
      () => {
        const td = el("td");
        const span = el("span", { cls: "truncate" });
        span.textContent = fact.subject || "—";
        td.appendChild(span);
        return td;
      },
      () => {
        const td = el("td");
        td.textContent = fact.attribute || "—";
        return td;
      },
      () => {
        const td = el("td");
        const v = el("span");
        v.textContent = fact.value || "—";
        if (fact.unit) {
          const u = el("span", { cls: "text-muted" });
          u.textContent = " " + fact.unit;
          td.append(v, u);
        } else {
          td.appendChild(v);
        }
        return td;
      },
      () => {
        const td = el("td", { cls: "cell-mono" });
        td.textContent = formatPeriod(fact);
        return td;
      },
      () => {
        const td = el("td");
        const badge = el("span", { cls: "badge" });
        badge.style.background = fact.fact_kind === "numerical"
          ? "rgba(129,140,248,0.12)" : "rgba(88,166,255,0.1)";
        badge.style.color = fact.fact_kind === "numerical"
          ? "var(--accent-bright)" : "var(--blue)";
        badge.style.border = "1px solid transparent";
        badge.textContent = fact.fact_kind || "—";
        td.appendChild(badge);
        return td;
      },
      () => {
        const td = el("td");
        td.appendChild(confBar(fact.confidence));
        return td;
      },
    ];

    for (const cellFn of cells) tr.appendChild(cellFn());

    tr.addEventListener("click", () => openFactDetail(fact));
    tbody.appendChild(tr);
  }

  table.appendChild(tbody);
  wrap.appendChild(table);
  container.appendChild(wrap);
}

// Track the element that had focus before opening the panel
let _panelPreviousFocus = null;

function openFactDetail(fact) {
  activeFactId = fact.id;
  const panel = document.getElementById("fact-detail-panel");

  // Title
  const titleEl = document.getElementById("detail-title");
  titleEl.textContent = (fact.subject || "Fact") + " — " + (fact.attribute || "");

  // Fields
  setDetailField("detail-subject",    fact.subject);
  setDetailField("detail-attribute",  fact.attribute);
  setDetailField("detail-value",      `${fact.value}${fact.unit ? " " + fact.unit : ""}`);
  setDetailField("detail-period",     formatPeriod(fact));
  setDetailField("detail-kind",       fact.fact_kind);
  setDetailField("detail-confidence", `${Math.round((fact.confidence ?? 0) * 100)}%`);
  setDetailField("detail-doc",        fact.doc_id);
  setDetailField("detail-claim",      fact.claim_text);

  // Verbatim evidence
  const evBlock = document.getElementById("detail-evidence");
  evBlock.textContent = fact.evidence_span || "(no verbatim span)";

  const pageEl = document.getElementById("detail-page");
  pageEl.textContent = fact.page != null ? `Page ${fact.page}` : "";

  // a11y: update aria-hidden and move focus to the close button
  _panelPreviousFocus = document.activeElement;
  panel.setAttribute("aria-hidden", "false");
  panel.classList.add("open");
  const closeBtn = document.getElementById("fact-detail-close");
  if (closeBtn) closeBtn.focus();
}

function setDetailField(id, value) {
  const el = document.getElementById(id);
  if (el) el.textContent = value ?? "—";
}

function closeFactDetail() {
  const panel = document.getElementById("fact-detail-panel");
  panel.setAttribute("aria-hidden", "true");
  panel.classList.remove("open");
  activeFactId = null;
  // Restore focus to the element that opened the panel
  if (_panelPreviousFocus && typeof _panelPreviousFocus.focus === "function") {
    _panelPreviousFocus.focus();
  }
  _panelPreviousFocus = null;
}

// ---------------------------------------------------------------------------
// VIEW: Relationships
// ---------------------------------------------------------------------------
async function loadRelationships() {
  const container = document.getElementById("rels-container");
  container.innerHTML = "";
  const loading = el("div", { cls: "state-loading" });
  loading.textContent = "Loading relationships…";
  container.appendChild(loading);

  const filterVal = document.getElementById("rel-filter-type").value;
  const params = new URLSearchParams();
  if (filterVal) params.set("relation", filterVal);

  try {
    const edges = await API.get(`/relationships?${params}`);
    renderRelationships(container, edges);
  } catch (err) {
    container.innerHTML = "";
    const e = el("div", { cls: "state-error" });
    e.textContent = "Failed to load relationships: " + err.message;
    container.appendChild(e);
  }
}

function renderRelationships(container, edges) {
  container.innerHTML = "";
  if (!edges.length) {
    const empty = el("div", { cls: "state-empty" });
    empty.textContent = "No relationships detected yet. Ingest multiple documents to find cross-document relationships.";
    container.appendChild(empty);
    return;
  }

  for (const edge of edges) {
    container.appendChild(buildRelCard(edge));
  }
}

function buildRelCard(edge) {
  const card = el("div", { cls: "rel-card" });

  // Header
  const header = el("div", { cls: `rel-card-header ${esc(edge.relation)}` });
  header.appendChild(relationBadge(edge.relation));
  if (edge.reconciling_dimension && edge.reconciling_dimension !== "none") {
    const dimTag = el("span", { cls: "dim-tag" });
    dimTag.textContent = "by " + edge.reconciling_dimension;
    header.appendChild(dimTag);
  }
  const confSpan = el("span", { cls: `conf-badge ${confClass(edge.confidence)} ml-auto` });
  confSpan.style.marginLeft = "auto";
  confSpan.textContent = Math.round((edge.confidence ?? 0) * 100) + "%";
  header.appendChild(confSpan);
  card.appendChild(header);

  // Body: two facts side by side
  const body = el("div", { cls: "rel-card-body" });

  const factA = edge.fact_a;
  const factB = edge.fact_b;

  body.appendChild(buildFactBlock(factA, "Claim A"));
  body.appendChild(buildFactBlock(factB, "Claim B"));

  // Reasoning (full width)
  if (edge.reasoning) {
    const reasonDiv = el("div", { cls: "rel-reasoning" });
    const lbl = el("strong");
    lbl.textContent = "System reasoning";
    const txt = document.createTextNode(edge.reasoning);
    reasonDiv.append(lbl, txt);
    body.appendChild(reasonDiv);
  }

  card.appendChild(body);
  return card;
}

function buildFactBlock(fact, label) {
  const block = el("div", { cls: "rel-fact-block" });

  const lbl = el("div", { cls: "rel-fact-label" });
  lbl.textContent = label;
  block.appendChild(lbl);

  if (!fact) {
    const missing = el("div", { cls: "rel-evidence" });
    missing.textContent = "(fact not found)";
    block.appendChild(missing);
    return block;
  }

  // Claim text
  const claimDiv = el("div");
  claimDiv.style.fontSize = "var(--text-sm)";
  claimDiv.style.fontWeight = "500";
  claimDiv.style.color = "var(--text-primary)";
  claimDiv.style.marginBottom = "6px";
  claimDiv.textContent = fact.claim_text || `${fact.subject}: ${fact.value} ${fact.unit || ""}`;
  block.appendChild(claimDiv);

  // Verbatim evidence
  const evDiv = el("div", { cls: "rel-evidence" });
  evDiv.textContent = fact.evidence_span || "(no evidence span)";
  block.appendChild(evDiv);

  // Page + doc
  const meta = el("div", { cls: "evidence-meta" });
  if (fact.page != null) {
    const pg = el("span", { cls: "evidence-meta-item" });
    const pgStrong = el("strong");
    pgStrong.textContent = "p. " + fact.page;
    pg.append("Page ", pgStrong);
    meta.appendChild(pg);
  }
  if (fact.doc_id) {
    const doc = el("span", { cls: "evidence-meta-item" });
    const docSpan = el("span", { cls: "font-mono" });
    docSpan.textContent = fact.doc_id.slice(0, 16) + (fact.doc_id.length > 16 ? "…" : "");
    doc.append("Doc: ", docSpan);
    meta.appendChild(doc);
  }
  if (meta.children.length) block.appendChild(meta);

  return block;
}

// ---------------------------------------------------------------------------
// VIEW: Four Cases
// ---------------------------------------------------------------------------
async function loadCases() {
  const container = document.getElementById("cases-grid");
  container.innerHTML = "";
  const loading = el("div", { cls: "state-loading" });
  loading.textContent = "Loading four cases…";
  container.appendChild(loading);

  try {
    const data = await API.get("/cases");
    container.innerHTML = "";
    const cases = [
      { key: "corroborate",  label: "Corroboration",       icon: "✓" },
      { key: "contradict",   label: "Contradiction",        icon: "✗" },
      { key: "reconcilable", label: "Context-Reconcilable", icon: "⟳" },
      { key: "failure",      label: "Low-Confidence / Failure", icon: "⚠" },
    ];
    for (const { key, label, icon } of cases) {
      container.appendChild(buildCasePanel(key, label, icon, data[key]));
    }
  } catch (err) {
    container.innerHTML = "";
    const e = el("div", { cls: "state-error" });
    e.textContent = "Failed to load cases: " + err.message;
    container.appendChild(e);
  }
}

function buildCasePanel(key, label, icon, caseData) {
  const panel = el("div", { cls: "case-panel" });

  // Header
  const header = el("div", { cls: `case-panel-header ${key}` });
  const iconSpan = el("span");
  iconSpan.style.fontSize = "18px";
  iconSpan.textContent = icon;
  const labelEl = el("span", { cls: `case-panel-label ${key}` });
  labelEl.textContent = label;
  header.append(iconSpan, labelEl);

  // Badge
  if (key !== "failure") {
    header.appendChild(relationBadge(key));
  } else {
    const b = el("span", { cls: "badge badge-failure" });
    b.textContent = "⚠ Failure";
    header.appendChild(b);
  }

  panel.appendChild(header);

  // Body
  const body = el("div", { cls: "case-panel-body" });

  if (!caseData || (!caseData.edge && !caseData.fact_a)) {
    const empty = el("div", { cls: "case-empty" });
    empty.textContent = key === "failure"
      ? "No low-confidence facts recorded yet."
      : `No ${label.toLowerCase()} case detected yet.`;
    body.appendChild(empty);
    panel.appendChild(body);
    return panel;
  }

  const edge = caseData.edge;
  const factA = caseData.fact_a;
  const factB = caseData.fact_b;

  if (factA) body.appendChild(buildCaseFact(factA, "Claim A"));
  if (factB) body.appendChild(buildCaseFact(factB, "Claim B"));

  if (edge && edge.reasoning) {
    const reasonDiv = el("div", { cls: "case-reasoning" });
    const lbl = el("span", { cls: "label" });
    lbl.textContent = "System reasoning";
    const reasonTxt = document.createTextNode(edge.reasoning);
    reasonDiv.append(lbl, reasonTxt);
    body.appendChild(reasonDiv);
  }

  if (edge && edge.confidence != null) {
    const confWrap = el("div", { cls: "case-conf" });
    confWrap.appendChild(confBar(edge.confidence));
    body.appendChild(confWrap);
  }

  panel.appendChild(body);
  return panel;
}

function buildCaseFact(fact, label) {
  const wrap = el("div", { cls: "case-fact" });

  const numEl = el("div", { cls: "case-fact-num" });
  numEl.textContent = label;
  wrap.appendChild(numEl);

  const claim = el("div", { cls: "case-fact-claim" });
  claim.textContent = fact.claim_text || `${fact.subject}: ${fact.value} ${fact.unit || ""}`;
  wrap.appendChild(claim);

  const ev = el("div", { cls: "case-fact-evidence" });
  ev.textContent = fact.evidence_span || "(no verbatim span)";
  wrap.appendChild(ev);

  // Page pill
  if (fact.page != null) {
    const pg = el("span", { cls: "page-pill mt-4" });
    pg.style.display = "inline-flex";
    pg.style.marginTop = "8px";
    pg.textContent = `📄 Page ${fact.page}`;
    wrap.appendChild(pg);
  }

  return wrap;
}

// ---------------------------------------------------------------------------
// VIEW: Ask
// ---------------------------------------------------------------------------
function initAsk() {
  const form = document.getElementById("ask-form");
  const questionInput = document.getElementById("ask-question");
  const submitBtn = document.getElementById("ask-submit");
  const answerDiv = document.getElementById("ask-answer");
  const answerText = document.getElementById("answer-text");
  const citationsWrap = document.getElementById("citations-wrap");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    submitBtn.disabled = true;
    const spinner = el("span", { cls: "spinner" });
    submitBtn.prepend(spinner);

    answerDiv.classList.remove("visible");

    try {
      const result = await API.post("/ask", { question });
      txt(answerText, result.answer || "No answer returned.");

      citationsWrap.innerHTML = "";
      const citations = result.citations || [];
      if (citations.length) {
        const title = el("div", { cls: "citations-title" });
        title.textContent = `${citations.length} Citation${citations.length !== 1 ? "s" : ""}`;
        citationsWrap.appendChild(title);

        for (const cite of citations) {
          citationsWrap.appendChild(buildCitationCard(cite));
        }
      } else {
        const none = el("div", { cls: "text-muted text-sm" });
        none.textContent = "No citations provided.";
        citationsWrap.appendChild(none);
      }

      answerDiv.classList.add("visible");
    } catch (err) {
      txt(answerText, "Error: " + err.message);
      citationsWrap.innerHTML = "";
      answerDiv.classList.add("visible");
    } finally {
      submitBtn.disabled = false;
      const s = submitBtn.querySelector(".spinner");
      if (s) s.remove();
    }
  });
}

function buildCitationCard(cite) {
  const card = el("div", { cls: "citation-item" });

  const meta = el("div", { cls: "citation-meta" });
  if (cite.page != null) {
    const pg = el("span");
    const pgStrong = el("strong");
    pgStrong.textContent = cite.page;
    pg.append("Page ", pgStrong);
    meta.appendChild(pg);
  }
  if (cite.doc_id) {
    const doc = el("span");
    const docMono = el("span", { cls: "font-mono" });
    docMono.textContent = cite.doc_id.slice(0, 20);
    doc.append("Doc: ", docMono);
    meta.appendChild(doc);
  }
  if (cite.fact_id) {
    const fid = el("span");
    const fidMono = el("span", { cls: "font-mono" });
    fidMono.textContent = cite.fact_id.slice(0, 16) + "…";
    fid.append("Fact: ", fidMono);
    meta.appendChild(fid);
  }
  card.appendChild(meta);

  if (cite.evidence_span) {
    const ev = el("div", { cls: "citation-evidence" });
    ev.textContent = cite.evidence_span;
    card.appendChild(ev);
  }

  return card;
}

// ---------------------------------------------------------------------------
// Export downloads
// ---------------------------------------------------------------------------
function triggerDownload(url, filename) {
  const a = document.createElement("a");
  a.href = API_BASE + url;
  a.download = filename;
  a.rel = "noopener noreferrer";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function exportFacts(format) {
  if (format === "json") {
    triggerDownload("/export?format=json", "veritas-export.json");
  } else if (format === "csv-facts") {
    triggerDownload("/export?format=csv&kind=facts", "veritas-facts.csv");
  } else if (format === "csv-edges") {
    triggerDownload("/export?format=csv&kind=edges", "veritas-edges.csv");
  }
}

function buildExportMenu() {
  const wrap = el("div", { cls: "export-wrap" });

  const btn = el("button", {
    cls: "btn btn-ghost export-btn",
    attrs: {
      type: "button",
      "aria-haspopup": "true",
      "aria-expanded": "false",
      "aria-label": "Export knowledge layer",
      id: "export-btn",
    },
  });
  btn.textContent = "↓ Export";

  const menu = el("div", {
    cls: "export-menu",
    attrs: {
      role: "menu",
      "aria-labelledby": "export-btn",
      hidden: "",
    },
  });

  const options = [
    { label: "Facts + Edges (JSON)", value: "json" },
    { label: "Facts (CSV)",          value: "csv-facts" },
    { label: "Edges (CSV)",          value: "csv-edges" },
  ];

  for (const opt of options) {
    const item = el("button", {
      cls: "export-menu-item",
      attrs: { type: "button", role: "menuitem" },
    });
    item.textContent = opt.label;
    item.addEventListener("click", () => {
      exportFacts(opt.value);
      closeExportMenu(btn, menu);
    });
    menu.appendChild(item);
  }

  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    const isOpen = !menu.hidden;
    if (isOpen) {
      closeExportMenu(btn, menu);
    } else {
      openExportMenu(btn, menu);
    }
  });

  // Close on outside click
  document.addEventListener("click", () => {
    if (!menu.hidden) closeExportMenu(btn, menu);
  });

  // Close on Escape
  menu.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeExportMenu(btn, menu);
      btn.focus();
    }
  });

  wrap.appendChild(btn);
  wrap.appendChild(menu);
  return wrap;
}

function openExportMenu(btn, menu) {
  menu.hidden = false;
  btn.setAttribute("aria-expanded", "true");
  // Focus first item
  const first = menu.querySelector(".export-menu-item");
  if (first) first.focus();
}

function closeExportMenu(btn, menu) {
  menu.hidden = true;
  btn.setAttribute("aria-expanded", "false");
}

// ---------------------------------------------------------------------------
// LLM status state (module-level, updated by setup flow)
// ---------------------------------------------------------------------------
let _llmConfigured = false;

/**
 * Update the top-bar LLM status indicator and Upload/Ask banners.
 *
 * Three states:
 *  - No provider: provider=null, model=null  → "LLM: not configured"
 *  - Provider connected, no model: provider set, model=null → "LLM: <provider> connected - select a model"
 *  - Ready: provider and model both set → "LLM: <provider> / <model>"
 *
 * _llmConfigured (ready) is true only when a model has been selected.
 * Banners are shown whenever _llmConfigured is false.
 */
function setLlmStatus(configured, provider, model) {
  _llmConfigured = !!configured;

  const dot  = document.getElementById("llm-status-dot");
  const text = document.getElementById("llm-status-text");

  if (!dot || !text) return;

  if (_llmConfigured && provider && model) {
    // State 3: fully ready
    dot.className    = "llm-status-dot llm-status-dot--on";
    text.textContent = "LLM: " + provider + " / " + model;
  } else if (provider) {
    // State 2: provider connected but no model selected yet
    dot.className    = "llm-status-dot llm-status-dot--mid";
    text.textContent = "LLM: " + provider + " connected - select a model";
  } else {
    // State 1: nothing configured
    dot.className    = "llm-status-dot llm-status-dot--off";
    text.textContent = "LLM: not configured";
  }

  // Show/hide banners in Upload and Ask views.
  // Banners stay visible until a model is selected (state 3).
  // When provider is connected but no model yet (state 2), swap banner message.
  const uploadBanner        = document.getElementById("upload-llm-banner");
  const askBanner           = document.getElementById("ask-llm-banner");
  const uploadBannerDefault = document.getElementById("upload-llm-banner-default");
  const uploadBannerModel   = document.getElementById("upload-llm-banner-model");
  const askBannerDefault    = document.getElementById("ask-llm-banner-default");
  const askBannerModel      = document.getElementById("ask-llm-banner-model");

  // In state 2 show the "select a model" variant; otherwise show the default.
  const needsModel = !!(provider && !model && !_llmConfigured);

  if (uploadBanner) {
    uploadBanner.hidden = _llmConfigured;
    if (uploadBannerDefault) uploadBannerDefault.hidden = needsModel;
    if (uploadBannerModel)   uploadBannerModel.hidden   = !needsModel;
  }
  if (askBanner) {
    askBanner.hidden = _llmConfigured;
    if (askBannerDefault) askBannerDefault.hidden = needsModel;
    if (askBannerModel)   askBannerModel.hidden   = !needsModel;
  }
}

// ---------------------------------------------------------------------------
// API helper with 400-aware error extraction (returns detail text from body)
// ---------------------------------------------------------------------------
async function apiPostRaw(path, body) {
  const r = await fetch(API_BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  return { ok: r.ok, status: r.status, data };
}

// ---------------------------------------------------------------------------
// VIEW: LLM Setup
// ---------------------------------------------------------------------------

// Module-level provider metadata fetched once from GET /config/llm/providers.
// Each entry: { id, label, needs_key, needs_base_url, default_base_url }
let _providerMeta = [];

// Guard: listeners are bound exactly once, even if the user navigates away
// and back to the LLM Setup view multiple times.
let _llmSetupInit = false;

function initLlmSetup() {
  // Re-entrancy guard — bind listeners only on the first call.
  // Subsequent calls (repeated navigation) may still refresh dynamic state
  // via refreshLlmStatus(), but must NOT re-bind event handlers.
  if (_llmSetupInit) return;
  _llmSetupInit = true;

  const providerSelect  = document.getElementById("llm-provider-select");
  const keyFields       = document.getElementById("llm-key-fields");
  const urlFields       = document.getElementById("llm-url-fields");
  const apiKeyInput     = document.getElementById("llm-api-key");
  const baseUrlInput    = document.getElementById("llm-base-url");
  const connectBtn      = document.getElementById("llm-connect-btn");
  const connectError    = document.getElementById("llm-connect-error");
  const modelCard       = document.getElementById("llm-model-card");
  const modelSelect     = document.getElementById("llm-model-select");
  const selectBtn       = document.getElementById("llm-select-btn");
  const selectError     = document.getElementById("llm-select-error");

  // Fetch provider list from the backend and build the <select> dynamically.
  async function loadProviders() {
    try {
      const providers = await API.get("/config/llm/providers");
      _providerMeta = providers;
      // Clear and rebuild options (XSS-safe via textContent)
      providerSelect.innerHTML = "";
      for (const p of providers) {
        const opt = document.createElement("option");
        opt.value = p.id;
        opt.textContent = p.label;
        providerSelect.appendChild(opt);
      }
      updateProviderFields();
    } catch {
      // If the endpoint is unavailable (old server), keep the select empty.
    }
  }

  // Show/hide key and base-url fields based on the selected provider's metadata.
  function updateProviderFields() {
    const selectedId = providerSelect.value;
    const meta = _providerMeta.find((p) => p.id === selectedId) || {};

    const needsKey     = !!meta.needs_key;
    const needsBaseUrl = !!meta.needs_base_url;

    keyFields.hidden = !needsKey;
    urlFields.hidden = !needsBaseUrl;

    // Pre-fill default_base_url when switching to a provider that has one.
    if (needsBaseUrl && meta.default_base_url) {
      if (!baseUrlInput.value || _lastProviderDefaultUrl === baseUrlInput.value) {
        baseUrlInput.value = meta.default_base_url;
      }
    } else if (!needsBaseUrl) {
      // Clear if switching away from a base-url provider.
      baseUrlInput.value = "";
    }
    _lastProviderDefaultUrl = (needsBaseUrl && meta.default_base_url) ? meta.default_base_url : "";
  }

  // Track the last auto-filled default URL so we don't overwrite user edits.
  let _lastProviderDefaultUrl = "";

  providerSelect.addEventListener("change", updateProviderFields);

  // Kick off provider load immediately (async, non-blocking).
  loadProviders();

  // Show inline error (XSS-safe via textContent)
  function showError(errEl, msg) {
    errEl.hidden = false;
    errEl.textContent = String(msg || "Unknown error");
  }

  function hideError(errEl) {
    errEl.hidden = true;
    errEl.textContent = "";
  }

  // Connect button handler
  connectBtn.addEventListener("click", async () => {
    hideError(connectError);
    const provider = providerSelect.value;
    const meta = _providerMeta.find((p) => p.id === provider) || {};

    const reqBody = { provider };

    if (meta.needs_key) {
      const key = apiKeyInput ? apiKeyInput.value : "";
      if (!key.trim() && !meta.key_optional) {
        showError(connectError, "Please enter an API key.");
        return;
      }
      // Send api_key even if empty (backend accepts empty string for keyless endpoints)
      reqBody.api_key = key;
    }

    if (meta.needs_base_url) {
      const url = baseUrlInput ? baseUrlInput.value.trim() : "";
      if (url) reqBody.base_url = url;
    }

    connectBtn.disabled = true;
    const spinner = el("span", { cls: "spinner" });
    connectBtn.prepend(spinner);

    try {
      const { ok, data } = await apiPostRaw("/config/llm/connect", reqBody);

      // Clear the key from input immediately after the request — never hold it
      if (apiKeyInput) apiKeyInput.value = "";

      if (!ok) {
        showError(connectError, data.detail || "Connection failed.");
        return;
      }

      // Success — populate and reveal model card
      const models = data.available_models || [];
      populateModels(modelSelect, models, null);
      modelCard.hidden = false;
      hideError(selectError);
      // Update top-bar: provider connected but no model selected yet (state 2)
      setLlmStatus(false, provider, null);
    } catch (err) {
      // Clear key even on network error
      if (apiKeyInput) apiKeyInput.value = "";
      showError(connectError, "Network error: " + err.message);
    } finally {
      connectBtn.disabled = false;
      const s = connectBtn.querySelector(".spinner");
      if (s) s.remove();
    }
  });

  // "Use this model" button handler
  selectBtn.addEventListener("click", async () => {
    hideError(selectError);
    const model = modelSelect.value;
    if (!model) {
      showError(selectError, "No model selected.");
      return;
    }

    selectBtn.disabled = true;
    const spinner = el("span", { cls: "spinner" });
    selectBtn.prepend(spinner);

    try {
      const { ok, data } = await apiPostRaw("/config/llm/select", { model });
      if (!ok) {
        showError(selectError, data.detail || "Model selection failed.");
        return;
      }
      // Reflect success in top bar (state 3: fully ready)
      setLlmStatus(true, data.provider, data.model);

      // Show a "connected" status badge inside the model card
      let badge = document.getElementById("llm-connected-badge");
      if (!badge) {
        badge = el("div", { cls: "llm-connected-badge", attrs: { id: "llm-connected-badge" } });
        modelCard.appendChild(badge);
      }
      const providerTxt = document.createTextNode("Connected: ");
      const modelTxt    = document.createTextNode(data.provider + " / " + data.model);
      badge.innerHTML   = "";     // wipe previous (safe — no API text here)
      badge.appendChild(providerTxt);
      const strong = el("strong");
      strong.appendChild(modelTxt);
      badge.appendChild(strong);
    } catch (err) {
      showError(selectError, "Network error: " + err.message);
    } finally {
      selectBtn.disabled = false;
      const s = selectBtn.querySelector(".spinner");
      if (s) s.remove();
    }
  });
}

// ---------------------------------------------------------------------------
// Shared helper: populate a <select> element from a sorted model list.
// `selected` may be null/undefined (nothing pre-selected).
// ---------------------------------------------------------------------------
function populateModels(selectEl, models, selected) {
  selectEl.innerHTML = "";
  const sorted = [...models].sort();
  for (const m of sorted) {
    const opt = document.createElement("option");
    opt.textContent = m;   // safe — textContent only
    opt.value = m;
    if (m === selected) opt.selected = true;
    selectEl.appendChild(opt);
  }
}

// Fetch current LLM config on load and hydrate UI state.
// Handles three backend states:
//   (a) configured=true, provider, model set       → state 3 (ready)
//   (b) has_key=true, provider set, model=null     → state 2 (connected, no model)
//   (c) nothing configured                          → state 1 (not configured)
async function refreshLlmStatus() {
  try {
    const cfg = await API.get("/config/llm");

    const providerSelect = document.getElementById("llm-provider-select");
    const modelCard      = document.getElementById("llm-model-card");
    const modelSelect    = document.getElementById("llm-model-select");

    // State (a): fully configured — provider + key + model all set
    if (cfg.configured && cfg.provider && cfg.model) {
      setLlmStatus(true, cfg.provider, cfg.model);

      // Pre-select provider in the dropdown (read-only hydration)
      if (providerSelect && cfg.provider) {
        providerSelect.value = cfg.provider;
        providerSelect.dispatchEvent(new Event("change"));
      }

      // Populate + reveal model card, pre-select current model
      if (modelCard && modelSelect && cfg.available_models && cfg.available_models.length) {
        populateModels(modelSelect, cfg.available_models, cfg.model);
        modelCard.hidden = false;

        // Show connected badge
        let badge = document.getElementById("llm-connected-badge");
        if (!badge) {
          badge = el("div", { cls: "llm-connected-badge", attrs: { id: "llm-connected-badge" } });
          modelCard.appendChild(badge);
        }
        badge.innerHTML = "";
        badge.appendChild(document.createTextNode("Connected: "));
        const strong = el("strong");
        strong.textContent = cfg.provider + " / " + cfg.model;
        badge.appendChild(strong);
      }

    // State (b): key/base_url held by backend, available_models known, but no model selected yet
    } else if ((cfg.has_key || cfg.provider) && cfg.provider && cfg.available_models && cfg.available_models.length) {
      setLlmStatus(false, cfg.provider, null);

      // Pre-select provider in the dropdown
      if (providerSelect && cfg.provider) {
        providerSelect.value = cfg.provider;
        providerSelect.dispatchEvent(new Event("change"));
      }

      // Reveal and populate the model dropdown so the user can pick without re-entering the key
      if (modelCard && modelSelect) {
        populateModels(modelSelect, cfg.available_models, cfg.model || null);
        modelCard.hidden = false;
      }

    // State (c): nothing configured
    } else {
      setLlmStatus(false, null, null);
    }
  } catch {
    // Backend might not have the endpoint yet — fail silently
    setLlmStatus(false, null, null);
  }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  // Register views
  registerView("view-llm-setup",     initLlmSetup);
  registerView("view-upload",        () => {});
  registerView("view-documents",     loadDocuments);
  registerView("view-facts",         loadFacts);
  registerView("view-relationships", loadRelationships);
  registerView("view-cases",         loadCases);
  registerView("view-ask",          () => {});

  // Wire nav buttons
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => showView(btn.dataset.view));
  });

  // Fact detail panel close (button + Escape key)
  const closeBtn = document.getElementById("fact-detail-close");
  if (closeBtn) closeBtn.addEventListener("click", closeFactDetail);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && activeFactId !== null) {
      closeFactDetail();
    }
  });

  // Fact filters
  const applyFiltersBtn = document.getElementById("facts-apply-filters");
  if (applyFiltersBtn) applyFiltersBtn.addEventListener("click", loadFacts);

  // Export menu — inject into the facts filter bar
  const factsFilterBar = document.querySelector("#view-facts .filter-bar");
  if (factsFilterBar) {
    factsFilterBar.appendChild(buildExportMenu());
  }

  // Relationship filter
  const relFilterBtn = document.getElementById("rels-apply-filter");
  if (relFilterBtn) relFilterBtn.addEventListener("click", loadRelationships);

  // Ask form
  initAsk();

  // Upload wiring
  initUpload();

  // Health counts
  refreshHealth();
  setInterval(refreshHealth, 15000);

  // LLM status (on load, hydrate state from backend)
  refreshLlmStatus();

  // Start on LLM Setup view (first/landing view)
  showView("view-llm-setup");
});
