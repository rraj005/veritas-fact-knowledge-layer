"""LLM-powered fact extractor for the Veritas fact knowledge layer."""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from veritas.extraction.prompts import build_extraction_prompt
from veritas.models import Chunk, Fact, new_id

if TYPE_CHECKING:
    from veritas.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Required fields in each LLM-returned row.
_REQUIRED_FIELDS = {
    "subject",
    "attribute",
    "value",
    "unit",
    "temporal_context",
    "scope_qualifiers",
    "evidence_span",
    "claim_text",
    "fact_kind",
    "confidence",
}


def _collapse_whitespace(text: str) -> str:
    """Collapse all runs of whitespace to a single space and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _strip_code_fences(response: str) -> str:
    """Remove leading/trailing markdown code fences (```json ... ```)."""
    text = response.strip()
    if text.startswith("```"):
        # drop the opening fence line (``` or ```json) and any closing fence
        text = text.split("\n", 1)[-1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -3]
    return text


def _parse_json_array(response: str) -> list[dict]:
    """Robustly extract a JSON array of fact rows from an LLM response.

    Handles bare arrays, markdown-fenced arrays, preamble text, and the common
    case where a model returns an object like ``{"facts": [...]}`` instead of a
    bare array. Returns an empty list if nothing parseable is found — and LOGS a
    warning with a snippet so silent zero-fact runs are diagnosable.
    """
    if not response or not response.strip():
        logger.warning("LLM returned an empty response (no text).")
        return []

    text = _strip_code_fences(response)

    start = text.find("[")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(text[start : i + 1])
                        if isinstance(parsed, list):
                            return parsed
                    except json.JSONDecodeError as exc:
                        logger.warning("JSON decode error in LLM response: %s", exc)
                    break

    # Fallback: a JSON object whose value is the list of facts, e.g. {"facts": [...]}.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, list):
                    return value
    except json.JSONDecodeError:
        pass

    logger.warning(
        "No JSON array of facts found in LLM response; first 300 chars: %r",
        response[:300],
    )
    return []


def _is_grounded(evidence_span: str, chunk_text: str) -> bool:
    """Return True if the (whitespace-collapsed) evidence_span is a substring
    of the (whitespace-collapsed) chunk text.
    """
    if not evidence_span:
        return False
    # Case-insensitive so a model that changes capitalisation of the quoted
    # span isn't dropped; still a real substring match (no paraphrase accepted).
    collapsed_span = _collapse_whitespace(evidence_span).lower()
    collapsed_text = _collapse_whitespace(chunk_text).lower()
    return collapsed_span in collapsed_text


def extract_from_chunk(chunk: Chunk, llm: LLMClient) -> list[Fact]:
    """Extract grounded facts from a single *chunk* using the given *llm*.

    Steps:
    1. Build the extraction prompt.
    2. Call the LLM.
    3. Parse the first JSON array in the response.
    4. Validate each row (required fields present, evidence grounded).
    5. Build Fact objects with doc_id/page from the chunk and a new id.

    Malformed or ungrounded rows are dropped and logged.

    Args:
        chunk: The document chunk to extract facts from.
        llm:   An LLMClient instance (real or fake).

    Returns:
        A list of grounded Fact objects (may be empty).
    """
    system, user = build_extraction_prompt(chunk.text)
    response = llm.complete(system, user)

    rows = _parse_json_array(response)
    if not isinstance(rows, list):
        logger.warning("LLM response did not parse to a list; got %r", type(rows))
        return []

    facts: list[Fact] = []
    for row in rows:
        if not isinstance(row, dict):
            logger.warning("Skipping non-dict row: %r", row)
            continue

        # Check required fields.
        missing = _REQUIRED_FIELDS - row.keys()
        if missing:
            logger.warning("Dropping row missing fields %s: %r", missing, row)
            continue

        evidence_span: str = row["evidence_span"]

        # Grounding check.
        if not _is_grounded(evidence_span, chunk.text):
            logger.warning(
                "Dropping ungrounded fact (evidence_span=%r not found in chunk text)",
                evidence_span,
            )
            continue

        # Build Fact.
        scope_qualifiers = row["scope_qualifiers"]
        if not isinstance(scope_qualifiers, list):
            scope_qualifiers = []

        try:
            confidence = float(row["confidence"])
        except (TypeError, ValueError):
            confidence = 0.5

        fact = Fact(
            id=new_id("fact"),
            doc_id=chunk.doc_id,
            subject=str(row["subject"]),
            attribute=str(row["attribute"]),
            value=str(row["value"]),
            unit=str(row["unit"]),
            temporal_context=str(row["temporal_context"]),
            scope_qualifiers=[str(q) for q in scope_qualifiers],
            evidence_span=evidence_span,
            page=chunk.page,
            claim_text=str(row["claim_text"]),
            fact_kind=str(row["fact_kind"]),
            confidence=confidence,
        )
        facts.append(fact)

    return facts


def extract_facts(
    chunks: list[Chunk],
    llm: LLMClient,
    concurrency: int = 4,
) -> list[Fact]:
    """Extract facts from all *chunks* using a thread pool.

    Flattens the results from all chunks into a single list.

    Args:
        chunks:      List of document chunks.
        llm:         LLMClient instance.
        concurrency: Maximum number of concurrent threads.

    Returns:
        Flat list of all extracted grounded facts.
    """
    if not chunks:
        return []

    all_facts: list[Fact] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # Submit all tasks; preserve order of results.
        futures = {executor.submit(extract_from_chunk, chunk, llm): chunk for chunk in chunks}
        for future in as_completed(futures):
            try:
                all_facts.extend(future.result())
            except Exception as exc:  # noqa: BLE001
                chunk = futures[future]
                logger.warning(
                    "extract_from_chunk failed for chunk (doc=%s page=%d): %s",
                    chunk.doc_id,
                    chunk.page,
                    exc,
                )

    return all_facts
