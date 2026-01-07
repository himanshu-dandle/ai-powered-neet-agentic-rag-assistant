from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict


Intent = Literal["concept_explain", "exam_qa", "mcq_generate", "doc_lookup"]


class SourceChunk(TypedDict):
    """
    A single retrieved chunk that can be cited.
    """
    source_id: str            # e.g., "S1"
    text: str                 # chunk text
    doc_name: str             # pdf filename
    page: Optional[int]       # page number if available
    chunk_id: str             # unique chunk identifier
    score: Optional[float]    # similarity score (if available)


@dataclass
class AgenticState:
    """
    Shared state passed through LangGraph nodes (agents).

    This is the 'blackboard' that each agent reads/writes.
    Keeping it explicit makes the system debuggable and production-ready.
    """

    # ---- Input ----
    user_query: str = ""

    # ---- Router outputs ----
    intent: Optional[Intent] = None
    retrieval_needed: bool = True
    router_confidence: float = 0.0
    router_plan: str = ""

    # ---- Retrieval outputs ----
    retrieved: List[SourceChunk] = field(default_factory=list)
    retrieval_query: str = ""
    retrieval_k: int = 0

    # ---- Answer outputs ----
    draft_answer: str = ""
    final_answer: str = ""

    # ---- Verification outputs ----
    verification_score: float = 0.0
    verification_notes: str = ""
    needs_iteration: bool = False

    # ---- Control ----
    iteration: int = 0
    max_iterations: int = 1

    # ---- Misc ----
    meta: Dict[str, Any] = field(default_factory=dict)
