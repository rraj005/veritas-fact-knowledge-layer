"""Domain-neutral extraction prompt builder for the Veritas fact knowledge layer."""

from __future__ import annotations

_SYSTEM_PROMPT = """\
You are a precise information-extraction assistant. Your task is to read the provided text \
and extract every atomic factual claim it contains — whether numerical (quantities, measurements, \
counts, rates, percentages) or semantic (states, relationships, classifications, events).

For each claim you find, return a JSON object with the following fields:
- "subject": the primary entity the claim is about (e.g. an organisation, person, product, region)
- "attribute": the property or measurement being described
- "value": the stated value or description
- "unit": the unit of measurement, if any (empty string if none)
- "temporal_context": the time period or date the claim refers to (empty string if not stated)
- "scope_qualifiers": a JSON array of strings describing any scope limitations \
(e.g. ["provisional"], ["estimated"], ["region=X"]); empty array if none
- "evidence_span": the VERBATIM span of text from the input that supports this claim — \
copy it character-for-character from the input; do NOT paraphrase or shorten it
- "claim_text": a short natural-language sentence summarising the claim in your own words
- "fact_kind": either "numerical" (involves a number/quantity) or "semantic" (qualitative/relational)
- "confidence": your confidence in the extraction, as a float between 0.0 and 1.0

Return a JSON array of such objects. If the text contains no factual claims, return an empty array [].

Rules:
- Each object must cover exactly ONE atomic claim.
- The "evidence_span" MUST appear verbatim in the input text; do not invent or alter it.
- Do NOT invent facts that are not stated in the text.
- Do NOT include the same claim more than once.
- Return ONLY the JSON array — no commentary, no markdown fences, no preamble.
"""


def build_extraction_prompt(chunk_text: str) -> tuple[str, str]:
    """Return (system, user) prompts for extracting facts from *chunk_text*.

    The system prompt is intentionally domain-neutral: it does not name any
    specific domain, industry, metric type, or entity class, so it generalises
    to arbitrary document types.

    Args:
        chunk_text: The raw text of a document chunk.

    Returns:
        A tuple (system_prompt, user_prompt) ready to pass to an LLMClient.
    """
    user_prompt = (
        "Extract all factual claims from the following text and return them as a "
        "JSON array as instructed.\n\n"
        f"Text:\n{chunk_text}"
    )
    return _SYSTEM_PROMPT, user_prompt
