import zipfile

import fitz
import pytest

from app import ingest


def make_pdf(path, pages):
    d = fitz.open()
    for t in pages:
        d.new_page().insert_text((72, 72), t)
    d.save(path)


def test_dedupe_and_extract(tmp_path):
    a, b = tmp_path / "a.pdf", tmp_path / "copy_of_a.pdf"
    make_pdf(a, ["Revenue was 10 million in FY2023 for the consolidated group, up from last year.", ""])
    b.write_bytes(a.read_bytes())
    groups = ingest.dedupe([a, b])
    assert len(groups) == 1 and len(next(iter(groups.values()))) == 2
    pages = ingest.extract_pdf(a)
    assert [p.pdf_page for p in pages] == [1, 2]
    assert not pages[0].low_text and pages[1].low_text
    assert all(len(c[1]) > 0 for c in ingest.chunk_page(pages[0]))


def test_unzip_rejects_traversal(tmp_path):
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("../evil.pdf", "x")
    with pytest.raises(ValueError):
        ingest.safe_unzip(z, tmp_path / "out")


def test_bad_zip(tmp_path):
    z = tmp_path / "x.zip"
    z.write_bytes(b"not a zip")
    with pytest.raises(zipfile.BadZipFile):
        ingest.safe_unzip(z, tmp_path / "o")
