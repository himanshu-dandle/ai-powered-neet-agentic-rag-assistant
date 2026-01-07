from app.utils.config import get_settings
from app.ingestion.pdf_loader import load_all_pdfs
from app.ingestion.chunker import chunk_pages
from app.ingestion.embedder import upsert_chunks, similarity_search

s = get_settings()

##pages = load_all_pdfs(s.raw_pdfs_dir, max_pages_per_pdf=1)
pages = load_all_pdfs(s.raw_pdfs_dir, max_pages_per_pdf=10)

chunks = chunk_pages(
    pages,
    chunk_size=s.chunk_size,
    chunk_overlap=s.chunk_overlap,
)

n = upsert_chunks(
    persist_dir=s.chroma_dir,
    embedding_model_name=s.embedding_model_name,
    chunks=chunks,
)

print("Upserted:", n)

hits = similarity_search(
    persist_dir=s.chroma_dir,
    embedding_model_name=s.embedding_model_name,
    query="What is Newton first law?",
    top_k=3,
)

print("Hits:", len(hits))
print("Top hit:")
print("  ID:", hits[0][0])
print("  Distance:", hits[0][3])
print("  Metadata:", hits[0][2])
