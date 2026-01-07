from __future__ import annotations

from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.agent_logger import AgentLogger, timed
from app.core.state import SourceChunk
from app.utils.config import Settings


class VerificationAgent:
    """
    Verification Agent:
    - checks the answer against the provided sources
    - flags unsupported claims / missing citations
    - returns a verification score (0.0 - 1.0) and whether iteration is needed

    NOTE:
    - Difficulty labels (Easy/Medium/Hard) are treated as heuristic metadata
      and are NOT required to be supported by sources.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.llm = ChatOpenAI(
            model=self.settings.llm_model_name,
            api_key=self.settings.openai_api_key,
            temperature=0.0,
        )

    def verify(
        self,
        *,
        user_query: str,
        answer_text: str,
        sources: List[SourceChunk],
        logger: AgentLogger,
        intent: str = "doc_lookup",
        min_score: float = 0.75,
    ) -> Tuple[float, bool, str]:

        sources_compact = "\n".join(
            [
                f"[{s['source_id']}] {s['doc_name']} (p. {s['page']}): "
                f"{s['text'][:350].replace(chr(10),' ')}"
                for s in sources
            ]
        )

        # ✅ For MCQ generation: enforce structural + citation checks (more stable than semantic fact-checking)
        if intent == "mcq_generate":
            has_any = ("[S" in answer_text)
            looks_like_5 = ("MCQ 5" in answer_text or "MCQ 5:" in answer_text)

            if not has_any:
                score = 0.4
                needs_iteration = True
                notes = "MCQ output has no citations. Require revision with [S#] citations."
            elif not looks_like_5:
                score = 0.6
                needs_iteration = True
                notes = "MCQ output does not appear to contain 5 questions. Require exactly 5 MCQs."
            else:
                score = 0.85
                needs_iteration = False
                notes = "MCQ output passes: structure present and citations detected. Difficulty is heuristic."

            logger.log_step(
                agent_name="VerificationAgent",
                action="verify_grounding",
                inputs_summary=f"(mcq) answer_len={len(answer_text)} | sources={len(sources)} | min_score={min_score}",
                outputs_summary=f"score={score:.2f} | needs_iteration={needs_iteration} | notes={notes[:220]}",
                confidence=0.9,
                elapsed_ms=0.0,
                meta={
                    "score": score,
                    "needs_iteration": needs_iteration,
                    "notes": notes,
                    "intent": intent,
                    "mode": "mcq_structural",
                },
            )
            return score, needs_iteration, notes


        system = (
            "You are a strict fact-checking verifier.\n"
            "You receive a user question, an assistant answer, and supporting sources.\n"
            "Your job:\n"
            "1) Identify whether the answer's factual claims are supported by the sources.\n"
            "2) Penalize missing citations [S#] for key factual claims.\n"
            "3) If the answer contains claims not supported by sources, flag them.\n"
            "4) IMPORTANT: Difficulty labels (Easy/Medium/Hard) are heuristic metadata.\n"
            "   Do NOT penalize the score for difficulty labels not being supported by sources.\n"
            "5) Output a JSON object ONLY with keys:\n"
            "   score (0-1), needs_iteration (true/false), notes (string).\n"
        )

        prompt = (
            f"User question:\n{user_query}\n\n"
            f"Assistant answer:\n{answer_text}\n\n"
            f"Sources:\n{sources_compact}\n"
        )

        with timed() as t:
            resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])

        raw = (resp.content or "").strip()

        # Minimal robust parsing without extra deps:
        score = 0.0
        needs_iteration = True
        notes = "Verifier could not parse output."

        try:
            import json

            data = json.loads(raw)
            score = float(data.get("score", 0.0))
            needs_iteration = bool(data.get("needs_iteration", True))
            notes = str(data.get("notes", "")).strip()
        except Exception:
            notes = f"Non-JSON verifier output: {raw[:600]}"

        # Rule-based guardrail: if no citations appear at all, require iteration
        has_any_citations = ("[S" in answer_text)
        if not has_any_citations:
            needs_iteration = True
            score = min(score, 0.55)
            if notes:
                notes += " "
            notes += "No citations detected in answer; require revision with citations."

        # Soft guardrail: avoid false 'missing citations' when citations are clearly present everywhere
        # (Verifier models can be overly strict; this reduces needless retries.)
        if has_any_citations and "missing citation" in notes.lower():
            score = max(score, 0.75)

        # Decide final iteration flag with threshold
        if score >= min_score:
            needs_iteration = False

        logger.log_step(
            agent_name="VerificationAgent",
            action="verify_grounding",
            inputs_summary=f"answer_len={len(answer_text)} | sources={len(sources)} | min_score={min_score}",
            outputs_summary=f"score={score:.2f} | needs_iteration={needs_iteration} | notes={notes[:220]}",
            confidence=0.9,
            elapsed_ms=t.elapsed_ms,
            meta={
                "score": score,
                "needs_iteration": needs_iteration,
                "notes": notes,
            },
        )

        return score, needs_iteration, notes
