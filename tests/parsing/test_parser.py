"""Tests for the PDF parser (Task 3)."""

from tests.fixtures.make_pdf import make_pdf
from veritas.parsing.parser import _render_table, count_pages, parse


def test_parse_pages(tmp_path):
    p = tmp_path / "doc.pdf"
    make_pdf(p, ["Revenue was 100 crore in FY24.", "Profit grew 20 percent."])
    blocks = parse(p, "d1")
    assert count_pages(p) == 2
    assert any("Revenue" in b.text for b in blocks)
    assert all(b.doc_id == "d1" for b in blocks)
    assert blocks[0].char_start == 0 and blocks[0].char_end == len(blocks[0].text)


def test_global_offset_continuity(tmp_path):
    """char_end of one block must equal char_start of the next."""
    p = tmp_path / "multi.pdf"
    make_pdf(p, ["First page text.", "Second page text."])
    blocks = parse(p, "d2")
    for i in range(len(blocks) - 1):
        assert blocks[i].char_end == blocks[i + 1].char_start


def test_doc_id_assigned(tmp_path):
    p = tmp_path / "id_test.pdf"
    make_pdf(p, ["Hello world"])
    blocks = parse(p, "my_doc")
    assert all(b.doc_id == "my_doc" for b in blocks)


def test_count_pages(tmp_path):
    p = tmp_path / "pages.pdf"
    make_pdf(p, ["p1", "p2", "p3"])
    assert count_pages(p) == 3


def test_render_table_pipe_delimited():
    """Unit test for the table-kind code path via the internal rendering helper.

    Approach: direct call to ``_render_table`` rather than round-tripping
    through a real PDF.  Reliably triggering pdfplumber's table-detection from
    a PyMuPDF-drawn PDF requires exact line-box geometry that varies across
    pdfplumber versions; testing the rendering helper directly is more
    deterministic and still exercises the ``kind="table"`` output format
    (pipe-delimited rows).
    """
    sample_table = [
        ["Company", "Revenue", "Year"],
        ["Acme Corp", "100 crore", "FY24"],
        ["Globex", None, "FY23"],  # None cell → empty string
    ]
    result = _render_table(sample_table)

    # Each row must be pipe-delimited.
    lines = result.rstrip("\n").split("\n")
    assert len(lines) == 3

    # Header row.
    assert "Company" in lines[0]
    assert "Revenue" in lines[0]
    assert "Year" in lines[0]
    assert " | " in lines[0]

    # Data rows.
    assert "100 crore" in lines[1]
    assert "Acme Corp" in lines[1]

    # None cell renders as empty string (two consecutive separators).
    assert "Globex" in lines[2]
    assert " |  | " in lines[2] or lines[2].endswith(" | ")

    # Must end with a trailing newline.
    assert result.endswith("\n")

    # Empty table yields empty string.
    assert _render_table([]) == ""
