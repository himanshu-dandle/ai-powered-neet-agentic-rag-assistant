from __future__ import annotations

import re
from typing import Dict, Tuple

from app.core.agent_logger import AgentLogger, timed
from app.core.state import Intent
from app.utils.config import Settings


class QueryRouterAgent:
    """
    Query Router Agent: decides what to do with the user's query.

    Output decisions:
    - intent: concept_explain | exam_qa | mcq_generate | doc_lookup
    - retrieval_needed: True/False
    - retrieval_filter: metadata filter for vector DB (e.g., doc_type)
    - plan: short human-readable plan for UI transparency
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def route(self, *, user_query: str, logger: AgentLogger) -> Tuple[Intent, bool, Dict, str, float]:
        q = user_query.strip().lower()

        with timed() as t:
            # Simple, deterministic heuristics (fast + reliable for v1).
            # Later we can upgrade to an LLM-based router if needed.
          ##  if re.search(r"\b(mcq|multiple choice|4 options|generate questions)\b", q):
            if re.search(r"\b(mcq|mcqs|multiple choice|4 options|generate\s+\d*\s*mcq|generate\s+mcq|generate\s+questions)\b", q):

                intent: Intent = "mcq_generate"
                retrieval_needed = True
                retrieval_filter = {"doc_type": "question_paper"}
                plan = "Generate MCQs → retrieve from past papers first; fallback to chapters if needed."
                confidence = 0.9

            elif re.search(r"\b(explain|what is|define|derivation|intuition|concept)\b", q):
                intent = "concept_explain"
                retrieval_needed = True
                retrieval_filter = {"doc_type": "chapter"}
                plan = "Explain concept → retrieve from chapters/notes and answer with citations."
                confidence = 0.85

            elif re.search(r"\b(answer|solve|find|calculate|prove)\b", q):
                intent = "exam_qa"
                retrieval_needed = True
                retrieval_filter = {}  # search everything
                plan = "Exam-style answer → retrieve relevant chunks from all documents and answer step-by-step with citations."
                confidence = 0.8

            else:
                intent = "doc_lookup"
                retrieval_needed = True
                retrieval_filter = {}  # search everything
                plan = "Lookup → retrieve most relevant passages and respond with direct citations."
                confidence = 0.7

        logger.log_step(
            agent_name="QueryRouterAgent",
            action="route",
            inputs_summary=f"user_query='{user_query[:250]}'",
            outputs_summary=f"intent={intent} | retrieval_needed={retrieval_needed} | filter={retrieval_filter} | plan={plan}",
            confidence=confidence,
            elapsed_ms=t.elapsed_ms,
            meta={
                "intent": intent,
                "retrieval_needed": retrieval_needed,
                "retrieval_filter": retrieval_filter,
                "plan": plan,
            },
        )

        return intent, retrieval_needed, retrieval_filter, plan, confidence
