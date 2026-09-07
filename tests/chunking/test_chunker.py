"""Tests for the page-safe chunker (Task 4)."""

from veritas.chunking.chunker import chunk
from veritas.models import PageBlock


def test_chunk_respects_page_and_size():
    blocks = [
        PageBlock("d1", 1, "A" * 1000, 0, 1000),
        PageBlock("d1", 1, "B" * 1000, 1000, 2000),
        PageBlock("d1", 2, "C" * 500, 2000, 2500),
    ]
    chunks = chunk(blocks, max_chars=1200, overlap=100)
    assert all(len(c.text) <= 1200 for c in chunks)
    assert all(c.page in (1, 2) for c in chunks)
    assert {c.page for c in chunks} == {1, 2}


def test_no_cross_page_merge():
    """Blocks on different pages must never be merged into a single chunk."""
    blocks = [
        PageBlock("d1", 1, "Page one text. " * 10, 0, 150),
        PageBlock("d1", 2, "Page two text. " * 10, 150, 300),
    ]
    chunks = chunk(blocks, max_chars=1200, overlap=0)
    pages_per_chunk = [c.page for c in chunks]
    # Each chunk must belong to exactly one page; no chunk should span both
    assert set(pages_per_chunk) == {1, 2}
    for c in chunks:
        assert c.page in (1, 2)


def test_chunk_preserves_offsets():
    """char_start of each chunk must be within the source block's range."""
    blocks = [
        PageBlock("d1", 1, "Hello world. " * 100, 0, 1300),
    ]
    chunks = chunk(blocks, max_chars=500, overlap=50)
    for c in chunks:
        assert c.char_start >= 0
        assert c.char_end > c.char_start
        assert len(c.text) == c.char_end - c.char_start


def test_section_hint_heading():
    """First line that is short and title-like should become the section_hint."""
    heading_text = "Revenue Summary\nThe company reported 100 crore in FY24."
    blocks = [PageBlock("d1", 1, heading_text, 0, len(heading_text))]
    chunks = chunk(blocks, max_chars=1200, overlap=0)
    assert chunks[0].section_hint == "Revenue Summary"


def test_section_hint_not_heading():
    """Long first lines should NOT become the section_hint."""
    long_first_line = "This is a very long first line that definitely does not look like a heading at all.\nMore text."
    blocks = [PageBlock("d1", 1, long_first_line, 0, len(long_first_line))]
    chunks = chunk(blocks, max_chars=1200, overlap=0)
    assert chunks[0].section_hint == ""


def test_single_block_fits_in_one_chunk():
    blocks = [PageBlock("d1", 1, "Short text.", 0, 11)]
    chunks = chunk(blocks, max_chars=1200, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_doc_id_propagated():
    blocks = [PageBlock("xyz", 3, "Some text here.", 500, 515)]
    chunks = chunk(blocks, max_chars=1200, overlap=0)
    assert all(c.doc_id == "xyz" for c in chunks)


def test_same_page_blocks_concatenated():
    """Five small same-page blocks must be merged into fewer chunks than blocks.

    With max_chars=1200 and 5 blocks of ~100 chars each (≈500 chars total after
    joining with newlines), all content fits in a single chunk.  The resulting
    chunk must sit on page 1 and its offsets must equal those of the merged run.
    """
    # Build 5 blocks of 100 'X' chars each, with consecutive global offsets.
    block_texts = ["X" * 100] * 5
    offset = 0
    blocks: list[PageBlock] = []
    for text in block_texts:
        blocks.append(PageBlock("d1", 1, text, offset, offset + len(text)))
        offset += len(text)

    chunks = chunk(blocks, max_chars=1200, overlap=0)

    # All 5 blocks are on page 1: merged text ≈ 504 chars → fits in 1 chunk.
    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    assert chunks[0].page == 1

    # char_start must be the first block's char_start (= 0).
    assert chunks[0].char_start == 0

    # char_end - char_start must equal len(text) for the chunk.
    assert chunks[0].char_end - chunks[0].char_start == len(chunks[0].text)
