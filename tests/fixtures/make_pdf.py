"""Fixture helper: create simple text-based PDFs for deterministic tests."""

from __future__ import annotations

from pathlib import Path


def make_pdf(path: str | Path, pages: list[str]) -> None:
    """Write a real text PDF with one page per item in *pages*.

    Uses PyMuPDF so that ``parse()`` can extract the text back deterministically.
    Each string in *pages* is inserted as text at position (72, 72) on its page.
    """
    import pymupdf

    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()
