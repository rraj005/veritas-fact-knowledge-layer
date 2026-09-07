"""Grounded Q&A over the Veritas fact knowledge layer (Task 13)."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from veritas.llm.client import LLMClient
    from veritas.store.db import Store
    from veritas.store.vectors import VectorIndex

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_QA_SYSTEM_PROMPT = """\
You are a grounded question-answering assistant. You will be given a question \
and a set of retrieved factual claims (each with a fact id, subject, attribute, \
value, temporal context, and verbatim evidence span). Your task is to:

1. Answer the question USING ONLY the information contained in the provided facts.
2. Cite the fact ids you used to compose your answer.
3. If the provided facts are insufficient to answer the question, say so clearly.

Return your response as a JSON object with exactly these keys:
  "answer": a concise natural-language answer (string)
  "cited_fact_ids": a JSON array of the fact id strings you relied on

Output ONLY the JSON object. No markdown, no preamble, no commentary.
"""


def _build_qa_prompt(question: str, facts: list[dict]) -> tuple[str, str]:
    """Return (system, user) prompts for grounded Q&A.

    Args:
        question: The user's natural-language question.
        facts:    A list of dicts, each representing a retrieved fact with at
                  minimum: fact_id, subject, attribute, value, unit,
                  temporal_context, evidence_span.

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    fact_lines = []
    for f in facts:
        line = (
            f"[{f['fact_id']}] "
            f"Subject: {f.get('subject', '')} | "
            f"Attribute: {f.get('attribute', '')} | "
            f"Value: {f.get('value', '')} {f.get('unit', '')} | "
            f"Period: {f.get('temporal_context', '')} | "
            f'Evidence: "{f.get("evidence_span", "")}"'
        )
        fact_lines.append(line)

    facts_block = "\n".join(fact_lines)

    user_prompt = (
        f"Question: {question}\n\n"
        f"Retrieved facts:\n{facts_block}\n\n"
        "Answer the question using only the facts above and cite the fact ids you used."
    )
    return _QA_SYSTEM_PROMPT, user_prompt


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------


def _extract_json_object(text: str) -> dict | None:
    """Robustly extract the first JSON object from *text*."""
    text = text.strip()
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def answer(
    question: str,
    store: Store,
    index: VectorIndex,
    llm: LLMClient,
    top_k: int = 8,
) -> dict:
    """Answer a question grounded in the indexed facts.

    Retrieval strategy:
    1. Query the vector index for the top_k most relevant facts across ALL
       documents (no where-filter — full-layer search).
    2. If retrieval is empty, return a refusal without calling the LLM.
    3. Build a grounded Q&A prompt instructing the LLM to answer using ONLY
       the provided facts and to cite their ids.
    4. Call the LLM; parse its JSON response.
    5. Resolve each cited fact id via store.get_fact; build citations.
    6. Return {"answer": ..., "citations": [...]}.

    Args:
        question: Natural-language question to answer.
        store:    SQLite store for fact lookups.
        index:    Vector index for semantic retrieval.
        llm:      LLM client for answer generation.
        top_k:    Maximum number of facts to retrieve.

    Returns:
        A dict with keys:
            "answer"    — string answer (or refusal message).
            "citations" — list of dicts with fact_id, doc_id, page, evidence_span.
    """
    # ------------------------------------------------------------------
    # Step 1: retrieve relevant facts (no doc filter — full-layer search)
    # ------------------------------------------------------------------
    if index.count() == 0:
        return {"answer": "Not enough grounded evidence.", "citations": []}

    candidates = index.query(question, top_k=top_k)

    if not candidates:
        return {"answer": "Not enough grounded evidence.", "citations": []}

    # ------------------------------------------------------------------
    # Step 2: load fact details for each retrieved id
    # ------------------------------------------------------------------
    retrieved_facts: list[dict] = []
    for cand_id, _distance, _meta in candidates:
        fact = store.get_fact(cand_id)
        if fact is None:
            logger.warning("qa: fact %r in index but not in store; skipping", cand_id)
            continue
        retrieved_facts.append(
            {
                "fact_id": fact.id,
                "subject": fact.subject,
                "attribute": fact.attribute,
                "value": fact.value,
                "unit": fact.unit,
                "temporal_context": fact.temporal_context,
                "evidence_span": fact.evidence_span,
                "doc_id": fact.doc_id,
                "page": fact.page,
            }
        )

    if not retrieved_facts:
        return {"answer": "Not enough grounded evidence.", "citations": []}

    # ------------------------------------------------------------------
    # Step 3 + 4: build prompt and call LLM
    # ------------------------------------------------------------------
    system, user = _build_qa_prompt(question, retrieved_facts)
    raw_response = llm.complete(system, user)

    # ------------------------------------------------------------------
    # Step 5: parse LLM response
    # ------------------------------------------------------------------
    parsed = _extract_json_object(raw_response)
    if parsed is None:
        logger.warning("qa: failed to parse LLM response: %r", raw_response[:200])
        return {
            "answer": "Unable to generate a grounded answer (LLM response parse error).",
            "citations": [],
        }

    answer_text: str = str(parsed.get("answer", ""))
    cited_ids: list[str] = parsed.get("cited_fact_ids", [])
    if not isinstance(cited_ids, list):
        cited_ids = []

    # ------------------------------------------------------------------
    # Step 6: resolve citations to structured records
    # ------------------------------------------------------------------
    citations: list[dict] = []
    for fact_id in cited_ids:
        fact = store.get_fact(str(fact_id))
        if fact is None:
            logger.warning("qa: cited fact id %r not found in store; skipping", fact_id)
            continue
        citations.append(
            {
                "fact_id": fact.id,
                "doc_id": fact.doc_id,
                "page": fact.page,
                "evidence_span": fact.evidence_span,
            }
        )

    return {"answer": answer_text, "citations": citations}
