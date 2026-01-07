from __future__ import annotations

from typing import List, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.ingestion.chunker import Chunk


COLLECTION_NAME = "studycoach_chunks"


def get_chroma_client(persist_dir: str) -> chromadb.PersistentClient:
    """
    Create a persistent Chroma client that stores vectors on disk.
    """
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_or_create_collection(client: chromadb.PersistentClient):
    """
    Use a single collection to store all chunks with metadata.
    """
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def build_embeddings(model_name: str, texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts using SentenceTransformer.
    Returns vectors as plain Python lists for Chroma.
    """
    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    return [v.tolist() for v in vectors]


def upsert_chunks(
    *,
    persist_dir: str,
    embedding_model_name: str,
    chunks: List[Chunk],
    batch_size: int = 128,
) -> int:
    """
    Upsert chunks into Chroma with embeddings + metadata.

    Idempotent behavior:
    - uses chunk_id as the vector id
    - re-running ingestion will overwrite/update existing ids
    """
    if not chunks:
        return 0

    client = get_chroma_client(persist_dir)
    col = get_or_create_collection(client)

    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]
        ids = [c.chunk_id for c in batch]

        metadatas = [
            {
                "doc_name": c.doc_name,
                "page": c.page,
                "doc_type": c.doc_type,
            }
            for c in batch
        ]

        embeddings = build_embeddings(embedding_model_name, texts)

        col.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        total += len(batch)


    return total


def similarity_search(
    *,
    persist_dir: str,
    embedding_model_name: str,
    query: str,
    top_k: int = 6,
    where: dict | None = None,
) -> List[Tuple[str, str, dict, float]]:
    """
    Query Chroma and return:
      [(id, document_text, metadata, distance), ...]

    distance is cosine distance when hnsw:space=cosine (lower is better).
    """
    client = get_chroma_client(persist_dir)
    col = get_or_create_collection(client)

    model = SentenceTransformer(embedding_model_name)
    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()

    res = col.query(
        query_embeddings=[qvec],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    out = []
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    for _id, doc, meta, dist in zip(ids, docs, metas, dists):
        out.append((_id, doc, meta, float(dist)))

    return out
