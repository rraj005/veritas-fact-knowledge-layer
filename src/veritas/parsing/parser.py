"""PDF parser: extract text and table blocks with document-global char offsets."""

from __future__ import annotations

import logging
from pathlib import Path

import pymupdf

from veritas.models import PageBlock

logger = logging.getLogger(__name__)


def _render_table(table: list[list[str | None]]) -> str:
    """Render a pdfplumber table (list-of-rows) as pipe-delimited text.

    Each row becomes a ``" | "``-joined string; rows are joined with newlines
    and a trailing newline is appended.  ``None`` cells are treated as empty
    strings.  Returns an empty string for an empty table.
    """
    if not table:
        return ""
    rows: list[str] = []
    for row in table:
        cells = [str(cell) if cell is not None else "" for cell in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows) + "\n"


def count_pages(path: str | Path) -> int:
    """Return the number of pages in a PDF file."""
    doc = pymupdf.open(str(path))
    n = doc.page_count
    doc.close()
    return n


def parse(path: str | Path, doc_id: str) -> list[PageBlock]:
    """Parse a PDF into a list of PageBlocks with document-global char offsets.

    Text blocks are extracted via PyMuPDF; table blocks are appended per page
    using pdfplumber (rendered as pipe-delimited text with kind="table").
    The running *offset* counter advances across all blocks so that
    ``char_start`` / ``char_end`` are document-global (suitable for evidence
    anchoring in downstream tasks).

    Per-page pdfplumber failures are caught, logged, and skipped so that a
    single corrupt table never aborts the full parse.
    """
    import pdfplumber  # lazy import – not needed for count_pages

    path = Path(path)
    blocks: list[PageBlock] = []
    offset: int = 0

    mupdf_doc = None
    plumber_doc = None

    mupdf_doc = pymupdf.open(str(path))
    plumber_doc = pdfplumber.open(str(path))

    try:
        for page_idx in range(mupdf_doc.page_count):
            page_num = page_idx + 1  # 1-based
            mu_page = mupdf_doc[page_idx]

            # --- text blocks via PyMuPDF ---
            raw_blocks = mu_page.get_text("blocks")
            # raw_blocks: list of (x0,y0,x1,y1,text,block_no,block_type)
            for raw in raw_blocks:
                block_type = raw[6]  # 0=text, 1=image
                if block_type != 0:
                    continue
                text: str = raw[4]
                if not text.strip():
                    continue
                char_start = offset
                char_end = offset + len(text)
                blocks.append(
                    PageBlock(
                        doc_id=doc_id,
                        page=page_num,
                        text=text,
                        char_start=char_start,
                        char_end=char_end,
                        kind="text",
                    )
                )
                offset = char_end

            # --- table blocks via pdfplumber ---
            try:
                pl_page = plumber_doc.pages[page_idx]
                tables = pl_page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    # Render table as pipe-delimited text via shared helper.
                    table_text = _render_table(table)
                    if not table_text.strip():
                        continue
                    char_start = offset
                    char_end = offset + len(table_text)
                    blocks.append(
                        PageBlock(
                            doc_id=doc_id,
                            page=page_num,
                            text=table_text,
                            char_start=char_start,
                            char_end=char_end,
                            kind="table",
                        )
                    )
                    offset = char_end
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "pdfplumber failed on page %d of %s: %s",
                    page_num,
                    path,
                    exc,
                )
    finally:
        if mupdf_doc is not None:
            mupdf_doc.close()
        if plumber_doc is not None:
            plumber_doc.close()

    return blocks
