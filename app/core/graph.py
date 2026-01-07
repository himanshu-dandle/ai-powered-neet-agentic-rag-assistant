# app/core/graph.py
from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.core.state import AgenticState
from app.agents.query_router import QueryRouterAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.answer_agent import AnswerAgent
from app.agents.verification_agent import VerificationAgent
from app.utils.config import Settings


def build_graph(settings: Settings):
    """
    Builds the LangGraph workflow for the Agentic RAG system.

    NOTE:
    - In this LangGraph version, node functions receive ONLY `state`.
    - We pass the runtime logger via `state.meta['logger']`.
    """

    router = QueryRouterAgent(settings)
    retriever = RetrievalAgent(settings)
    answerer = AnswerAgent(settings)
    verifier = VerificationAgent(settings)

    graph = StateGraph(AgenticState)

    # ---------------- Nodes ----------------

    def route_node(state: AgenticState) -> AgenticState:
        logger = state.meta["logger"]

        # ✅ Forced intent from UI (Concept / Exam / MCQ)
        forced_intent = state.meta.get("forced_intent")
        if forced_intent:
            state.intent = forced_intent
            state.retrieval_needed = True
            state.router_confidence = 1.0
            state.router_plan = f"Forced mode → intent={forced_intent}"

            # allow full retrieval by default; retry logic can still broaden further
            state.meta["retrieval_filter"] = {}

            logger.log_step(
                agent_name="QueryRouterAgent",
                action="force_intent",
                inputs_summary=f"user_query='{state.user_query[:200]}'",
                outputs_summary=(
                    f"intent={forced_intent} | retrieval_needed=True | "
                    f"filter={{}} | plan=Forced mode"
                ),
                confidence=1.0,
                elapsed_ms=0.0,
                meta={"forced_intent": forced_intent},
            )
            return state

        # Normal auto routing
        intent, retrieval_needed, retrieval_filter, plan, conf = router.route(
            user_query=state.user_query,
            logger=logger,
        )
        state.intent = intent
        state.retrieval_needed = retrieval_needed
        state.router_confidence = conf
        state.router_plan = plan

        # keep filter in meta (simple + flexible)
        state.meta["retrieval_filter"] = retrieval_filter
        return state

    def retrieval_node(state: AgenticState) -> AgenticState:
        logger = state.meta["logger"]

        where = state.meta.get("retrieval_filter") or {}
        sources = retriever.retrieve(
            user_query=state.user_query,
            logger=logger,
            top_k=settings.top_k,
            where=where,
        )
        state.retrieved = sources
        state.retrieval_query = state.user_query
        state.retrieval_k = settings.top_k
        return state

    def answer_node(state: AgenticState) -> AgenticState:
        logger = state.meta["logger"]

        sources = state.retrieved or []
        draft = answerer.answer(
            intent=state.intent or "doc_lookup",
            user_query=state.user_query,
            sources=sources,
            logger=logger,
        )
        state.draft_answer = draft
        state.final_answer = draft
        return state

    def verify_node(state: AgenticState) -> AgenticState:
        logger = state.meta["logger"]

        score, needs_iteration, notes = verifier.verify(
            user_query=state.user_query,
            answer_text=state.final_answer,
            sources=state.retrieved or [],
            logger=logger,
            intent=state.intent or "doc_lookup",
        )

        state.verification_score = score
        state.needs_iteration = needs_iteration
        state.verification_notes = notes
        return state

    def iterate_node(state: AgenticState) -> AgenticState:
        # Increase iteration and broaden retrieval scope on retry (industry pattern)
        state.iteration += 1

        # broaden search: remove doc_type filter on retry
        state.meta["retrieval_filter"] = {}

        # increase k slightly
        state.meta["retry_top_k"] = min(settings.top_k + 3, 12)
        return state

    def retrieval_retry_node(state: AgenticState) -> AgenticState:
        logger = state.meta["logger"]

        top_k = int(state.meta.get("retry_top_k", settings.top_k))
        sources = retriever.retrieve(
            user_query=state.user_query,
            logger=logger,
            top_k=top_k,
            where=state.meta.get("retrieval_filter") or {},
        )
        state.retrieved = sources
        state.retrieval_query = state.user_query
        state.retrieval_k = top_k
        return state

    # ---------------- Control ----------------

    def should_iterate(state: AgenticState) -> str:
        # Optional iteration: if verifier says iterate and we haven't exceeded max
        if state.needs_iteration and state.iteration < state.max_iterations:
            return "iterate"
        return "finish"

    # ---------------- Graph wiring ----------------

    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("answer", answer_node)
    graph.add_node("verify", verify_node)
    graph.add_node("iterate", iterate_node)
    graph.add_node("retrieve_retry", retrieval_retry_node)

    graph.set_entry_point("route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", "verify")

    graph.add_conditional_edges(
        "verify",
        should_iterate,
        {
            "iterate": "iterate",
            "finish": END,
        },
    )

    graph.add_edge("iterate", "retrieve_retry")
    graph.add_edge("retrieve_retry", "answer")

    return graph.compile()
