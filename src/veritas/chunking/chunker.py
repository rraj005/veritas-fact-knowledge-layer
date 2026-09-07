"""Page-safe sliding-window chunker with overlap.

Design constraints
------------------
- NEVER merges blocks across pages (preserves page-level grounding).
- Consecutive same-page blocks are CONCATENATED (joined with "\\n") before
  the sliding window is applied, so the window sees the full page run at once.
- Each chunk is at most *max_chars* characters long.
- Consecutive chunks on the same page overlap by *overlap* characters so that
  facts near a chunk boundary are represented in at least two chunks.
- ``char_start`` / ``char_end`` on each ``Chunk`` are document-global offsets:
  ``merged_char_start + local_pos`` where ``merged_char_start`` is the
  ``char_start`` of the first block in the merged group.
- ``section_hint`` is derived from the merged group's leading text (first line
  of the concatenated string), not carried per-block.
"""

from __future__ import annotations

import itertools

from veritas.models import Chunk, PageBlock

# Maximum first-line length to qualify as a heading candidate.
_HEADING_MAX_LEN: int = 60
# Characters that suggest the first line is prose rather than a heading.
_HEADING_EXCLUDED_ENDINGS: tuple[str, ...] = (".", ",", ";", ":", "?", "!")


def _is_heading(line: str) -> bool:
    """Return True when *line* looks like a section heading."""
    stripped = line.strip()
    if not stripped:
        return False
    if len(stripped) > _HEADING_MAX_LEN:
        return False
    return not stripped.endswith(_HEADING_EXCLUDED_ENDINGS)


def _section_hint_for(text: str) -> str:
    """Return the section hint derived from the first line of *text*."""
    first_line = text.split("\n", 1)[0]
    return first_line.strip() if _is_heading(first_line) else ""


def _slice_merged_into_chunks(
    doc_id: str,
    page: int,
    merged_text: str,
    merged_char_start: int,
    max_chars: int,
    overlap: int,
) -> list[Chunk]:
    """Slice *merged_text* into ≤ max_chars chunks with *overlap*.

    ``char_start`` / ``char_end`` offsets are kept document-global by adding
    ``merged_char_start`` to each local position within *merged_text*.
    """
    total = len(merged_text)
    hint = _section_hint_for(merged_text)

    if total <= max_chars:
        # Fast path: merged text fits entirely in one chunk.
        return [
            Chunk(
                doc_id=doc_id,
                page=page,
                text=merged_text,
                char_start=merged_char_start,
                char_end=merged_char_start + total,
                section_hint=hint,
            )
        ]

    chunks: list[Chunk] = []
    pos = 0  # local offset into *merged_text*

    while pos < total:
        end = min(pos + max_chars, total)
        slice_text = merged_text[pos:end]
        chunks.append(
            Chunk(
                doc_id=doc_id,
                page=page,
                text=slice_text,
                char_start=merged_char_start + pos,
                char_end=merged_char_start + end,
                section_hint=hint,
            )
        )
        if end == total:
            break
        # Advance by (max_chars - overlap), but always make forward progress.
        step = max(1, max_chars - overlap)
        pos += step

    return chunks


def chunk(
    blocks: list[PageBlock],
    max_chars: int = 1200,
    overlap: int = 150,
) -> list[Chunk]:
    """Convert *blocks* into overlapping ``Chunk`` objects, never crossing pages.

    Consecutive same-page blocks are concatenated (joined with ``"\\n"``) before
    the sliding-window is applied.  The ``char_start`` of the merged group is
    the ``char_start`` of the first block in the group, so all chunk offsets
    remain document-global.

    Parameters
    ----------
    blocks:
        Source ``PageBlock`` list (typically produced by ``parse()``).
    max_chars:
        Maximum number of characters per chunk.
    overlap:
        Number of characters carried over from the end of one chunk into the
        start of the next chunk *within the same page*.
    """
    result: list[Chunk] = []

    # Group blocks by page (preserving order within each page).
    for _page, page_block_iter in itertools.groupby(blocks, key=lambda b: b.page):
        page_blocks = list(page_block_iter)
        if not page_blocks:
            continue

        # Concatenate all same-page blocks into a single merged string.
        merged_text = "\n".join(b.text for b in page_blocks)
        # Anchor offset at the first block's char_start.
        merged_char_start = page_blocks[0].char_start
        doc_id = page_blocks[0].doc_id

        result.extend(
            _slice_merged_into_chunks(
                doc_id=doc_id,
                page=_page,
                merged_text=merged_text,
                merged_char_start=merged_char_start,
                max_chars=max_chars,
                overlap=overlap,
            )
        )

    return result
