import streamlit as st

from app.utils.config import get_settings
from app.core.agent_logger import AgentLogger
from app.core.state import AgenticState
from app.core.graph import build_graph


def render_sources(sources):
    if not sources:
        st.info("No sources retrieved.")
        return
    for s in sources:
        title = f"[{s['source_id']}] {s['doc_name']} (p. {s['page']})"
        with st.expander(title):
            st.caption(f"chunk_id: {s['chunk_id']} | distance: {s['score']}")
            st.write(s["text"])


def render_agent_steps(steps):
    if not steps:
        st.info("No agent steps logged.")
        return

    for i, step in enumerate(steps, start=1):
        # Support both dict steps and AgentStep objects
        if isinstance(step, dict):
            agent_name = step.get("agent_name")
            action = step.get("action")
            confidence = step.get("confidence")
            inputs_summary = step.get("inputs_summary")
            outputs_summary = step.get("outputs_summary")
            elapsed_ms = step.get("elapsed_ms")
            meta = step.get("meta")
        else:
            agent_name = getattr(step, "agent_name", None)
            action = getattr(step, "action", None)
            confidence = getattr(step, "confidence", None)
            inputs_summary = getattr(step, "inputs_summary", None)
            outputs_summary = getattr(step, "outputs_summary", None)
            elapsed_ms = getattr(step, "elapsed_ms", None)
            meta = getattr(step, "meta", None)

        header = f"{i}. {agent_name} → {action} (confidence={confidence})"
        with st.expander(header, expanded=(i <= 2)):
            st.write(f"**Inputs:** {inputs_summary}")
            st.write(f"**Outputs:** {outputs_summary}")
            st.write(f"**Elapsed (ms):** {elapsed_ms}")
            if meta:
                st.json(meta)



def main():
    st.set_page_config(page_title="AgenticRAG StudyCoach", layout="wide")
    st.title("AgenticRAG StudyCoach (Agentic RAG Demo)")

    settings = get_settings()
    graph = build_graph(settings)



    with st.sidebar:
        st.header("Controls")

        mode = st.radio(
            "Mode",
            options=["Auto (Agent decides)", "Concept Explanation", "Exam-style Q&A", "MCQ Generator"],
            index=0,
        )

        max_iter = st.slider("Max iterations (verification-driven retry)", 0, 2, 1)
        st.caption("Tip: if verification fails, the graph retries retrieval with broader scope.")
        st.divider()
        st.caption("Vector DB: Chroma (persistent)")
        st.caption(f"Embeddings: {settings.embedding_model_name}")
        st.caption(f"LLM: {settings.llm_model_name}")


    q = st.text_area("Ask a question", height=90, placeholder="e.g., Explain Newton's first law in simple terms")
    run = st.button("Run Agentic RAG", type="primary")

    if run:
        if not q.strip():
            st.warning("Please enter a question.")
            st.stop()

        logger = AgentLogger()
        state = AgenticState(user_query=q.strip(), max_iterations=max_iter)
        
        # If user selects a mode, force the intent through the state meta
        if mode == "Concept Explanation":
            state.meta["forced_intent"] = "concept_explain"
        elif mode == "Exam-style Q&A":
            state.meta["forced_intent"] = "exam_qa"
        elif mode == "MCQ Generator":
            state.meta["forced_intent"] = "mcq_generate"

        state.meta["logger"] = logger  # inject logger for LangGraph nodes

        with st.spinner("Running agents..."):
            out = graph.invoke(state)  # returns dict in your LangGraph version

        col1, col2 = st.columns([1.1, 0.9], gap="large")

        with col1:
            st.subheader("Final Answer")
            st.write(out.get("final_answer", ""))

            st.subheader("Sources Used")
            render_sources(out.get("retrieved", []))

        with col2:
            st.subheader("Agent Steps")
            render_agent_steps(logger.steps)

            st.subheader("Verification")
            st.write(f"**Score:** {out.get('verification_score')}")
            st.write(f"**Needs iteration:** {out.get('needs_iteration')}")
            st.write(f"**Notes:** {out.get('verification_notes')}")

            st.subheader("Router Plan")
            st.write(out.get("router_plan", ""))


if __name__ == "__main__":
    main()
