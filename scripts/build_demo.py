"""Demo builder for the Veritas fact knowledge layer.

Ingests the starter PDFs from both datasets into a FRESH knowledge layer
(data/demo/ directory), builds the four showcase cases, prints them to
stdout, and writes docs/demo-cases.md.

Usage::

    # Requires LLM_PROVIDER and (for openai) OPENAI_API_KEY
    python scripts/build_demo.py                         # ingest all 6 PDFs
    python scripts/build_demo.py --dataset delhivery     # only Delhivery PDFs
    python scripts/build_demo.py --dataset india-macroeconomy  # only macro PDFs
    python scripts/build_demo.py --limit 2               # cap at 2 PDFs per dataset

Provider setup:
  OpenAI:  export LLM_PROVIDER=openai  OPENAI_API_KEY=sk-...  LLM_MODEL=gpt-4o
  Ollama:  export LLM_PROVIDER=ollama  LLM_MODEL=llama3.1

Model selection:
  Set LLM_MODEL to skip discovery, or leave it unset to auto-pick the first
  available model returned by the provider.

BYOK: keys are NEVER committed to the repo.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup -- allow running from repo root without editable install.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Provider guard -- checked BEFORE any heavy import so the message is fast.
# ---------------------------------------------------------------------------
from veritas.config import get_settings

_settings = get_settings()

_PROVIDER = _settings.llm_provider

_SUPPORTED = {"openai", "ollama"}

if _PROVIDER not in _SUPPORTED:
    print(
        "\nERROR: LLM_PROVIDER must be set to one of: "
        + ", ".join(sorted(_SUPPORTED))
        + f"\nCurrently set to: '{_PROVIDER or '(empty)'}'\n"
        "\nQuick start:\n"
        "  export LLM_PROVIDER=openai\n"
        "  export OPENAI_API_KEY=sk-...\n"
        "  export LLM_MODEL=gpt-4o\n"
        "  python scripts/build_demo.py\n"
        "\nOr for Ollama:\n"
        "  export LLM_PROVIDER=ollama\n"
        "  export LLM_MODEL=llama3.1\n"
        "  python scripts/build_demo.py\n"
        "\nOr copy .env.example -> .env and fill in your values.\n",
        file=sys.stderr,
    )
    sys.exit(1)

if _PROVIDER == "openai" and not _settings.openai_api_key:
    print(
        "\nERROR: OPENAI_API_KEY is required when LLM_PROVIDER=openai.\n"
        "\nSet it in your environment or in a .env file (never commit real keys).\n",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Heavy imports (after guard so the missing-config message is instant).
# ---------------------------------------------------------------------------
import json
import shutil
import textwrap
import types
from datetime import UTC, datetime

from veritas.cases import build_cases
from veritas.llm.client import get_client
from veritas.llm.models import list_models
from veritas.pipeline import Pipeline
from veritas.store.db import Store
from veritas.store.embeddings import SentenceTransformerEmbedder
from veritas.store.vectors import VectorIndex


def _resolve_model(provider: str, settings) -> str:
    """Return the model to use: from LLM_MODEL env, or first from discovery."""
    model_env = os.getenv("LLM_MODEL", "").strip()
    if model_env:
        return model_env

    print(f"LLM_MODEL not set -- discovering models for provider '{provider}'...")
    try:
        api_key = settings.openai_api_key if provider == "openai" else None
        base_url = settings.ollama_base_url if provider == "ollama" else None
        models = list_models(provider, api_key=api_key, base_url=base_url)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: Could not list models: {exc}", file=sys.stderr)
        sys.exit(1)

    if not models:
        print(
            "ERROR: No models returned by provider. "
            "Set LLM_MODEL explicitly and try again.",
            file=sys.stderr,
        )
        sys.exit(1)

    chosen = models[0]
    print(f"Auto-selected first available model: '{chosen}'")
    print(f"  (Available: {', '.join(models[:5])}{'...' if len(models) > 5 else ''})")
    return chosen


def _load_evaluate_module() -> types.ModuleType:
    """Load eval/evaluate.py by absolute path via importlib.util.

    This avoids fragile sys.path manipulation and ensures we always load
    the intended file, regardless of any other 'evaluate' module on sys.path.
    """
    eval_py = _REPO_ROOT / "eval" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("veritas_eval.evaluate", eval_py)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load eval module from {eval_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert mod.__file__ == str(eval_py), (
        f"Loaded wrong evaluate module: {mod.__file__!r} != {eval_py!r}"
    )
    return mod


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a fresh Veritas knowledge layer from the starter PDFs."
    )
    p.add_argument(
        "--dataset",
        choices=["delhivery", "india-macroeconomy", "all"],
        default="all",
        help="Which dataset(s) to ingest (default: all).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Cap ingestion at N PDFs per dataset (useful for shorter demo runs).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# PDF discovery helpers
# ---------------------------------------------------------------------------


def _collect_pdfs(
    dataset: str,
    limit: int | None,
    repo_root: Path,
) -> list[Path]:
    """Return sorted list of PDF paths for the requested dataset(s)."""
    datasets_dir = repo_root / "starter-datasets"

    folders: list[Path] = []
    if dataset in ("delhivery", "all"):
        folders.append(datasets_dir / "delhivery")
    if dataset in ("india-macroeconomy", "all"):
        folders.append(datasets_dir / "india-macroeconomy")

    pdfs: list[Path] = []
    for folder in folders:
        folder_pdfs = sorted(folder.glob("*.pdf"))
        if limit is not None:
            folder_pdfs = folder_pdfs[:limit]
        pdfs.extend(folder_pdfs)

    return pdfs


# ---------------------------------------------------------------------------
# Case rendering helpers
# ---------------------------------------------------------------------------


def _render_fact_block(label: str, fact: dict) -> str:
    """Return a multi-line string describing one fact."""
    lines = [
        f"  {label}:",
        f"    Claim     : {fact.get('claim_text', '')}",
        f"    Value     : {fact.get('value', '')} {fact.get('unit', '')}",
        f"    Period    : {fact.get('temporal_context', '')}",
        f"    Page      : {fact.get('page', '')}",
        f"    Document  : {fact.get('doc_id', '')}",
        f"    Evidence  : {fact.get('evidence_span', '')}",
    ]
    return "\n".join(lines)


def _render_edge_block(edge: dict) -> str:
    return (
        f"  Relation  : {edge.get('relation', '').upper()}\n"
        f"  Reasoning : {edge.get('reasoning', '')}\n"
        f"  Reconcile : {edge.get('reconciling_dimension', '')}\n"
        f"  Confidence: {edge.get('confidence', '')}"
    )


def _render_case(label: str, case: dict | None) -> str:
    """Render a single case slot (corroborate / contradict / reconcilable / failure)."""
    sep = "-" * 70
    header = f"\n{'=' * 70}\n CASE: {label.upper()}\n{'=' * 70}"

    if case is None:
        return f"{header}\n  (no example found - ingest more documents or check the log)\n"

    # Failure slot only has a single fact.
    if label == "failure":
        lines = [
            header,
            "  (Lowest-confidence extracted fact - review for quality)\n",
            _render_fact_block("Fact", case),
            "",
        ]
        return "\n".join(lines)

    # Relation slots have edge + two facts.
    edge = case.get("edge", {})
    fact_a = case.get("fact_a", {})
    fact_b = case.get("fact_b", {})

    lines = [
        header,
        _render_edge_block(edge),
        sep,
        _render_fact_block("Fact A", fact_a),
        sep,
        _render_fact_block("Fact B", fact_b),
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown rendering helpers
# ---------------------------------------------------------------------------


def _md_escape(s: str) -> str:
    """Minimal Markdown escaping for plain-text content."""
    return s.replace("|", "\\|").replace("<", "&lt;").replace(">", "&gt;")


def _fact_to_md_table(fact: dict) -> str:
    rows = [
        ("Claim", _md_escape(fact.get("claim_text", ""))),
        ("Value", _md_escape(f"{fact.get('value', '')} {fact.get('unit', '')}".strip())),
        ("Period", _md_escape(fact.get("temporal_context", ""))),
        ("Page", str(fact.get("page", ""))),
        ("Document", _md_escape(fact.get("doc_id", ""))),
        ("Evidence", _md_escape(fact.get("evidence_span", ""))),
    ]
    lines = ["| Field | Value |", "| --- | --- |"]
    for k, v in rows:
        lines.append(f"| **{k}** | {v} |")
    return "\n".join(lines)


def _render_case_md(label: str, case: dict | None) -> str:
    """Render a case as a Markdown section."""
    h2 = f"\n## Case: {label.title()}\n"

    if case is None:
        return h2 + "_No example found -- ingest more documents or check the log._\n"

    if label == "failure":
        return (
            h2
            + "_Lowest-confidence extracted fact -- review for quality._\n\n"
            + "### Fact\n\n"
            + _fact_to_md_table(case)
            + "\n"
        )

    edge = case.get("edge", {})
    fact_a = case.get("fact_a", {})
    fact_b = case.get("fact_b", {})

    return (
        h2
        + f"**Relation:** `{edge.get('relation', '').upper()}`  \n"
        + f"**Reasoning:** {_md_escape(edge.get('reasoning', ''))}  \n"
        + f"**Reconciling dimension:** `{edge.get('reconciling_dimension', '')}`  \n"
        + f"**Confidence:** {edge.get('confidence', '')}  \n\n"
        + "### Fact A\n\n"
        + _fact_to_md_table(fact_a)
        + "\n\n"
        + "### Fact B\n\n"
        + _fact_to_md_table(fact_b)
        + "\n"
    )


def _write_demo_cases_md(cases: dict, counts: dict, repo_root: Path) -> Path:
    """Write docs/demo-cases.md and return the path."""
    docs_dir = repo_root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / "demo-cases.md"

    header = textwrap.dedent(f"""\
        # Veritas Demo Cases

        > Auto-generated by `scripts/build_demo.py` on {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")}.
        > Do not edit manually -- re-run the script to regenerate.

        This file shows one example of each cross-document relationship type
        detected by the Veritas fact knowledge layer after ingesting the starter PDFs.

        **Layer stats:** {counts.get("docs", 0)} documents | {counts.get("facts", 0)} facts | {counts.get("edges", 0)} edges

        ---
    """)

    sections = []
    for slot in ("corroborate", "contradict", "reconcilable", "failure"):
        sections.append(_render_case_md(slot, cases.get(slot)))

    md_path.write_text(header + "\n".join(sections), encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    repo_root = _REPO_ROOT
    pdfs = _collect_pdfs(args.dataset, args.limit, repo_root)

    if not pdfs:
        print("ERROR: No PDFs found for the requested dataset(s).", file=sys.stderr)
        sys.exit(1)

    # Resolve model (from env or auto-discovery).
    model = _resolve_model(_PROVIDER, _settings)

    import dataclasses

    resolved_settings = dataclasses.replace(_settings, llm_model=model)

    print("\nVeritas Demo Builder")
    print(f"{'=' * 50}")
    print(f"  Provider   : {_PROVIDER}")
    print(f"  Model      : {model}")
    print(f"  Dataset(s) : {args.dataset}")
    print(f"  Limit      : {args.limit or 'none'}")
    print(f"  PDFs found : {len(pdfs)}")
    for p in pdfs:
        print(f"    * {p.name}  ({p.parent.name})")
    print()

    # ------------------------------------------------------------------
    # Build FRESH knowledge layer in data/demo/
    # ------------------------------------------------------------------
    demo_dir = repo_root / "data" / "demo"
    if demo_dir.exists():
        print(f"Removing existing demo layer at {demo_dir} ...")
        shutil.rmtree(demo_dir)

    demo_dir.mkdir(parents=True)
    chroma_dir = demo_dir / "chroma"
    chroma_dir.mkdir()

    print(f"Initialising fresh layer in {demo_dir}")

    store = Store(demo_dir / "veritas.db")
    store.init_schema()

    embedder = SentenceTransformerEmbedder(resolved_settings.embedding_model)
    index = VectorIndex(chroma_dir, embedder)
    llm = get_client(resolved_settings)

    pipeline = Pipeline(store, index, llm, resolved_settings)

    # ------------------------------------------------------------------
    # Ingest each PDF
    # ------------------------------------------------------------------
    for i, pdf_path in enumerate(pdfs, start=1):
        filename = pdf_path.name
        print(f"\n[{i}/{len(pdfs)}] Ingesting: {filename}")

        def _cb(stage: str, pct: float, msg: str, _fn: str = filename) -> None:
            bar_len = 30
            filled = int(bar_len * pct)
            bar = "#" * filled + "-" * (bar_len - filled)
            print(f"  [{bar}] {pct * 100:5.1f}%  [{stage}]  {msg}")

        pipeline.ingest_document(pdf_path, filename, progress_cb=_cb)

    # ------------------------------------------------------------------
    # Build cases
    # ------------------------------------------------------------------
    print("\nBuilding four showcase cases ...")
    cases = build_cases(store)
    counts = store.counts()

    # ------------------------------------------------------------------
    # Print cases to stdout
    # ------------------------------------------------------------------
    print("\n")
    for slot in ("corroborate", "contradict", "reconcilable", "failure"):
        print(_render_case(slot, cases.get(slot)))

    # ------------------------------------------------------------------
    # Write docs/demo-cases.md
    # ------------------------------------------------------------------
    md_path = _write_demo_cases_md(cases, counts, repo_root)
    print(f"\nMarkdown written to: {md_path}")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"  Documents : {counts.get('docs', 0)}")
    print(f"  Facts     : {counts.get('facts', 0)}")
    print(f"  Edges     : {counts.get('edges', 0)}")
    print("=" * 50)

    # ------------------------------------------------------------------
    # Run evaluation harness
    # ------------------------------------------------------------------
    gold_path = repo_root / "eval" / "gold.json"
    if gold_path.exists():
        _eval_mod = _load_evaluate_module()
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        metrics = _eval_mod.evaluate(store, gold)
        print("\nEvaluation Harness Results")
        print(_eval_mod.format_report(metrics))


if __name__ == "__main__":
    main()
