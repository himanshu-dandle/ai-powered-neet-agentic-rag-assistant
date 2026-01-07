from __future__ import annotations

import re
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.agent_logger import AgentLogger, timed
from app.core.state import Intent, SourceChunk
from app.utils.config import Settings


def format_sources_block(sources: List[SourceChunk]) -> str:
    """
    Prepare sources for the LLM in a strict, citation-friendly format.
    """
    lines = []
    for s in sources:
        header = f"[{s['source_id']}] {s['doc_name']} (p. {s['page']}) | chunk_id={s['chunk_id']}"
        body = s["text"].replace("\n", " ").strip()
        lines.append(header + "\n" + body)
    return "\n\n".join(lines)


def has_citations(text: str) -> bool:
    return "[S" in text


def any_paragraph_missing_citation(text: str, allowed_ids: List[str]) -> bool:
    """
    Returns True if any non-empty paragraph lacks at least one allowed citation token.
    Paragraphs are separated by blank lines.
    """
    if not text:
        return True

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return True

    # Match any allowed citation like [S1], [S2], ...
    allowed_pattern = r"(" + "|".join([re.escape(f"[{sid}]") for sid in allowed_ids]) + r")"

    for p in paragraphs:
        if not re.search(allowed_pattern, p):
            return True
    return False


class AnswerAgent:
    """
    Reasoning / Answer Agent:
    - produces grounded answers using retrieved sources
    - outputs citations like [S1], [S2]
    - supports concept explanations, exam-style answers, and MCQ generation
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = ChatOpenAI(
            model=self.settings.llm_model_name,
            api_key=self.settings.openai_api_key,
            temperature=0.2,
        )

    def answer(
        self,
        *,
        intent: Intent,
        user_query: str,
        sources: List[SourceChunk],
        logger: AgentLogger,
    ) -> str:
        sources_block = format_sources_block(sources)
        allowed_citations = ", ".join([f"[{s['source_id']}]" for s in sources])

        system_rules = (
            "You are a careful assistant that MUST stay grounded in the provided sources.\n"
            "Rules:\n"
            "1) Use ONLY the sources below as evidence. If something is not supported, say so.\n"
            "2) You MUST include at least ONE citation [S#] in EVERY paragraph (including the first paragraph). If you cannot cite, write: 'Not found in sources.'\n"
            "2a) If you write a bullet list, EACH bullet must end with at least one allowed citation like [S1].\n"
            f"Allowed citation ids are ONLY: {allowed_citations}. Do NOT cite anything else.\n"
            "3) Do NOT invent examples. Only use examples explicitly supported by sources.\n"
            "4) If sources are insufficient, ask for clarification OR state what is missing.\n"
            "5) Keep the final answer clear and exam-appropriate.\n"
        )

        if intent == "concept_explain":
            task_instructions = (
                "Task: Explain the concept clearly and simply.\n"
                "Include key definitions and conditions.\n"
                "IMPORTANT: Do NOT add any example unless an example is explicitly present in the sources.\n"
            )
        elif intent == "exam_qa":
            task_instructions = (
                "Task: Provide an exam-style solution.\n"
                "Use step-by-step reasoning and show the final result.\n"
            )
      

        elif intent == "mcq_generate":
            task_instructions = (
                "Task: Generate MCQs based ONLY on the sources.\n"
                "Output format (STRICT):\n"
                "For i=1..5 produce:\n"
                "MCQ i: <question> [S#]\n"
                "A) ... [S#]\n"
                "B) ... [S#]\n"
                "C) ... [S#]\n"
                "D) ... [S#]\n"
                "Correct: <A/B/C/D> [S#]\n"
                "Difficulty: Easy/Medium/Hard (heuristic) [S#]\n"
                "\n"
                "Rules:\n"
                "- Exactly 5 MCQs.\n"
                "- 4 options each (A-D).\n"
                "- Every line MUST end with at least one allowed citation [S#].\n"
                "- Do NOT invent facts not present in sources.\n"
                "- Difficulty is a heuristic label; choose based on complexity of the sourced concept.\n"
            )

        else:
            task_instructions = (
                "Task: Provide a direct, source-grounded response.\n"
                "Prefer quoting/pointing to the relevant parts with citations.\n"
            )

        user_prompt = (
            f"{task_instructions}\n"
            f"User question: {user_query}\n\n"
            f"Sources:\n{sources_block}\n"
        )

        with timed() as t:
            resp = self.llm.invoke(
                [
                    SystemMessage(content=system_rules),
                    HumanMessage(content=user_prompt),
                ]
            )

        answer_text = (resp.content or "").strip()

        # Hard guardrail: enforce at least one ALLOWED citation in EVERY paragraph.
        allowed_ids = [s["source_id"] for s in sources]
        if sources and any_paragraph_missing_citation(answer_text, allowed_ids):
            revision_system = (
                "You are revising an existing answer to make it strictly grounded.\n"
                "Rules:\n"
                "1) Do NOT add new claims.\n"
                "2) Add at least ONE citation like [S1] to EVERY paragraph (including the first).\n"
                f"Allowed citation ids are ONLY: {allowed_citations}. Do NOT cite anything else.\n"
                "3) If a sentence is not supported by sources, remove it or replace it with: 'Not found in sources.'\n"
                "4) Keep the meaning but ensure citations exist.\n"
            )

            revision_prompt = (
                f"User question: {user_query}\n\n"
                f"Sources:\n{sources_block}\n\n"
                f"Draft answer (revise it to add citations):\n{answer_text}\n"
            )

            with timed() as t2:
                resp2 = self.llm.invoke(
                    [
                        SystemMessage(content=revision_system),
                        HumanMessage(content=revision_prompt),
                    ]
                )

            answer_text = (resp2.content or "").strip()

            logger.log_step(
                agent_name="AnswerAgent",
                action="revise_for_citations",
                inputs_summary="Auto-revision triggered because a paragraph was missing citations.",
                outputs_summary=f"Revised answer length={len(answer_text)} chars.",
                confidence=0.75,
                elapsed_ms=t2.elapsed_ms,
                meta={"reason": "paragraph_missing_citation"},
            )

        logger.log_step(
            agent_name="AnswerAgent",
            action="generate_answer",
            inputs_summary=f"intent={intent} | user_query='{user_query[:220]}' | sources={len(sources)}",
            outputs_summary=f"Generated answer length={len(answer_text)} chars.",
            confidence=0.8 if sources else 0.2,
            elapsed_ms=t.elapsed_ms,
            meta={
                "intent": intent,
                "model": self.settings.llm_model_name,
                "sources_used": [s["source_id"] for s in sources[:6]],
            },
        )

        return answer_text
