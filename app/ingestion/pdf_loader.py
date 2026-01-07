from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from pypdf import PdfReader


@dataclass
class PageText:
    """
    Represents extracted text from a single PDF page.
    Keeping page numbers is critical for citations.
    """
    doc_name: str
    page: int              # 1-based page number for human-friendly citations
    text: str


def list_pdfs(raw_pdfs_dir: str) -> List[Path]:
    """
    Discover PDFs in the raw PDFs folder (non-recursive).
    """
    p = Path(raw_pdfs_dir)
    if not p.exists():
        raise FileNotFoundError(f"Raw PDF folder not found: {p.resolve()}")
    return sorted([x for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".pdf"])


def load_pdf_pages(pdf_path: Path, *, max_pages: Optional[int] = None) -> List[PageText]:
    """
    Load a PDF and extract text page-by-page.

    Args:
        pdf_path: path to a PDF file
        max_pages: optionally limit pages for quick local testing
    """
    reader = PdfReader(str(pdf_path))
    doc_name = pdf_path.name

    pages: List[PageText] = []
    total = len(reader.pages)
    limit = min(total, max_pages) if max_pages is not None else total

    for i in range(limit):
        page_obj = reader.pages[i]
        text = page_obj.extract_text() or ""
        # normalize whitespace lightly (don't over-clean here)
        text = "\n".join([line.rstrip() for line in text.splitlines()]).strip()
        pages.append(PageText(doc_name=doc_name, page=i + 1, text=text))

    return pages


def load_all_pdfs(raw_pdfs_dir: str, *, max_pages_per_pdf: Optional[int] = None) -> List[PageText]:
    """
    Load all PDFs from a folder and return a flat list of PageText.
    """
    all_pages: List[PageText] = []
    for pdf in list_pdfs(raw_pdfs_dir):
        all_pages.extend(load_pdf_pages(pdf, max_pages=max_pages_per_pdf))
    return all_pages
