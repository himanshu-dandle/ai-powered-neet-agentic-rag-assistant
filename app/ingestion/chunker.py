from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.ingestion.pdf_loader import PageText


@dataclass
class Chunk:
    """
    A chunk of text that will be embedded and stored in the vector DB.

    We keep doc_name + page so that later we can produce citations like:
      [S1] Current Electricity.pdf (p. 12)
    """
    chunk_id: str
    doc_name: str
    page: int
    text: str
    doc_type: str  # "chapter" | "question_paper"


def infer_doc_type(doc_name: str) -> str:
    """
    Very simple heuristic for now:
    - 'neet_physics_2023.pdf' / 'neet_physics_2024.pdf' => question_paper
    - everything else => chapter

    We can refine later without changing the system design.
    """
    name = doc_name.lower()
    if re.search(r"neet.*(20\d{2})", name):
        return "question_paper"
    return "chapter"


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Split text into overlapping chunks (character-based).
    This is stable, fast, and works for PDFs where sentence boundaries are messy.

    chunk_overlap must be < chunk_size.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap
        if start < 0:
            start = 0
        if end == len(text):
            break
    return chunks


def chunk_pages(pages: List[PageText], *, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    """
    Convert extracted PDF pages into chunks with metadata.
    """
    out: List[Chunk] = []
    counters: Dict[str, int] = {}

    for p in pages:
        doc_type = infer_doc_type(p.doc_name)

        # Track chunk numbering per document
        counters.setdefault(p.doc_name, 0)

        for part in split_text(p.text, chunk_size, chunk_overlap):
            counters[p.doc_name] += 1
            chunk_id = f"{p.doc_name}::p{p.page}::c{counters[p.doc_name]}"
            out.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_name=p.doc_name,
                    page=p.page,
                    text=part,
                    doc_type=doc_type,
                )
            )

    return out
