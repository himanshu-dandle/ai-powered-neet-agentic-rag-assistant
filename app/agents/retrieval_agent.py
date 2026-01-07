from __future__ import annotations

from typing import Dict, List, Optional

from app.core.agent_logger import AgentLogger, timed
from app.core.state import SourceChunk
from app.ingestion.embedder import similarity_search
from app.utils.config import Settings


class RetrievalAgent:
    """
    Retrieval Agent: performs semantic search over the vector store.

    Responsibilities:
    - run similarity search (Chroma)
    - apply optional metadata filters (e.g., doc_type)
    - convert raw hits into SourceChunk objects with stable citation ids (S1, S2, ...)
    - log what it did for transparency and debugging
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def retrieve(
        self,
        *,
        user_query: str,
        logger: AgentLogger,
        top_k: Optional[int] = None,
        where: Optional[Dict] = None,
        retrieval_query_override: Optional[str] = None,
    ) -> List[SourceChunk]:
        k = top_k or self.settings.top_k
        retrieval_query = retrieval_query_override or user_query

        # Chroma expects where=None when no filter is applied (where={} can raise in some versions)
        where = where or None


        with timed() as t:
            hits = similarity_search(
                persist_dir=self.settings.chroma_dir,
                embedding_model_name=self.settings.embedding_model_name,
                query=retrieval_query,
                top_k=k,
                where=where,
            )

        sources: List[SourceChunk] = []
        for idx, (chunk_id, text, meta, dist) in enumerate(hits, start=1):
            source_id = f"S{idx}"
            sources.append(
                SourceChunk(
                    source_id=source_id,
                    text=text,
                    doc_name=str(meta.get("doc_name", "")),
                    page=int(meta["page"]) if meta.get("page") is not None else None,
                    chunk_id=str(chunk_id),
                    score=float(dist),
                )
            )

        # Log a compact summary for Streamlit
        where_summary = str(where) if where else "None"
        outputs_summary = (
            f"Retrieved {len(sources)} chunks (top_k={k}). "
            f"Top doc: {sources[0]['doc_name'] if sources else 'N/A'}."
        )

        logger.log_step(
            agent_name="RetrievalAgent",
            action="semantic_search",
            inputs_summary=f"query='{retrieval_query[:200]}' | where={where_summary}",
            outputs_summary=outputs_summary,
            confidence=0.85 if sources else 0.2,
            elapsed_ms=t.elapsed_ms,
            meta={
                "top_k": k,
                "where": where or {},
                "top_hits": [
                    {
                        "source_id": s["source_id"],
                        "doc_name": s["doc_name"],
                        "page": s["page"],
                        "chunk_id": s["chunk_id"],
                        "distance": s["score"],
                    }
                    for s in sources[:3]
                ],
            },
        )

        return sources
