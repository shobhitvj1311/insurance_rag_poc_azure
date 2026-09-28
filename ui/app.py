"""
Compare Assistant — a small local UI to run one question through all four
retrieval configurations at once: dense, hybrid, dense+rerank, hybrid+rerank.

This calls the same functions ask_dense.py / ask_hybrid.py use
(retrieval.common.base + retrieval.common.reranker), just fanned out over
four configurations in one page instead of one CLI session per config.

Run:
    pip install streamlit
    streamlit run ui/compare_app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

# this file lives at <repo_root>/ui/compare_app.py
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import (
    DENSE_OUTPUT_DIRECTORY,
    HYBRID_OUTPUT_DIRECTORY,
    RERANK_CANDIDATE_POOL,
    RERANK_MODEL_NAME,
    TOP_K,
)
from retrieval.common.base import (
    CHAT_DEPLOYMENT,
    EMBEDDING_DEPLOYMENT,
    create_openai_client,
    generate_answer,
    get_question_embedding,
    load_dense_index,
    load_hybrid_index,
    retrieve_dense_chunks,
    retrieve_hybrid_chunks,
)

st.set_page_config(page_title="RAG Config Comparison", layout="wide")

CONFIGS = [
    {"key": "dense", "label": "Dense", "mode": "dense", "rerank": False},
    {"key": "dense_rerank", "label": "Dense + Rerank", "mode": "dense", "rerank": True},
    {"key": "hybrid", "label": "Hybrid", "mode": "hybrid", "rerank": False},
    {"key": "hybrid_rerank", "label": "Hybrid + Rerank", "mode": "hybrid", "rerank": True},
]


# ---------------------------------------------------------------------------
# Cached resources — loaded once per Streamlit server process, not per question
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to Azure OpenAI...")
def get_client():
    return create_openai_client()


@st.cache_resource(show_spinner="Loading dense index...")
def get_dense_index():
    return load_dense_index(DENSE_OUTPUT_DIRECTORY)


@st.cache_resource(show_spinner="Loading hybrid index...")
def get_hybrid_index():
    return load_hybrid_index(HYBRID_OUTPUT_DIRECTORY)


@st.cache_resource(show_spinner="Loading reranker model (first run only, ~1GB download)...")
def get_reranker():
    from retrieval.common.reranker import rerank_chunks
    return rerank_chunks


# ---------------------------------------------------------------------------
# Core: run one question through one configuration
# ---------------------------------------------------------------------------
def run_config(client, question, question_embedding, config, dense_data, hybrid_data, rerank_fn):
    """Retrieve, optionally rerank, and generate an answer for one config.
    Returns a dict with everything the UI needs to render this cell."""

    start = time.time()

    retrieval_top_k = RERANK_CANDIDATE_POOL if config["rerank"] else TOP_K

    if config["mode"] == "hybrid":
        embeddings, metadata, corpus_texts = hybrid_data
        chunks = retrieve_hybrid_chunks(
            question_embedding, embeddings, metadata, corpus_texts, question,
            top_k=retrieval_top_k,
        )
    else:
        embeddings, metadata = dense_data
        chunks = retrieve_dense_chunks(
            question_embedding, embeddings, metadata, top_k=retrieval_top_k,
        )

    if config["rerank"]:
        chunks = rerank_fn(question, chunks, top_k=TOP_K)

    try:
        answer = generate_answer(client, question, chunks, chat_name=CHAT_DEPLOYMENT)
        error = None
    except Exception as e:
        answer = None
        error = str(e)

    return {
        "config": config,
        "chunks": chunks,
        "answer": answer,
        "error": error,
        "elapsed_seconds": round(time.time() - start, 2),
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("Policy and Claims Knowledge Assistant")
st.caption(
    f"Chat model: `GPT-5-mini` · Embedding model: `text-embedding-3-small` · "
    f"Reranker: `{RERANK_MODEL_NAME}` · top_k: {TOP_K} · rerank candidate pool: {RERANK_CANDIDATE_POOL}"
)

question = st.text_input(
    "Ask a question",
    placeholder="e.g. What is the Total Standard Excess payable on a claim?",
)

st.markdown("**Choose which configuration(s) to run:**")
checkbox_columns = st.columns(4)
selected_keys = []
for column, config in zip(checkbox_columns, CONFIGS):
    with column:
        checked = st.checkbox(config["label"], value=False, key=f"select_{config['key']}")
        if checked:
            selected_keys.append(config["key"])

selected_configs = [c for c in CONFIGS if c["key"] in selected_keys]

run_clicked = st.button(
    "Run selected configuration(s)",
    type="primary",
    disabled=not (question.strip() and selected_configs),
)

if question.strip() and not selected_configs:
    st.caption("Select at least one configuration above to enable Run.")

if run_clicked and question.strip() and selected_configs:
    client = get_client()

    # Only load the index(es) actually needed for the selected configs.
    needs_dense = any(c["mode"] == "dense" for c in selected_configs)
    needs_hybrid = any(c["mode"] == "hybrid" for c in selected_configs)
    needs_rerank = any(c["rerank"] for c in selected_configs)

    dense_data = get_dense_index() if needs_dense else None
    hybrid_data = get_hybrid_index() if needs_hybrid else None
    rerank_fn = get_reranker() if needs_rerank else None

    with st.spinner("Embedding question..."):
        question_embedding = get_question_embedding(client, question, embedding_name=EMBEDDING_DEPLOYMENT)

    columns = st.columns(len(selected_configs))
    results = {}

    for column, config in zip(columns, selected_configs):
        with column:
            st.subheader(config["label"])
            with st.spinner(f"Running {config['label']}..."):
                result = run_config(
                    client, question, question_embedding, config,
                    dense_data, hybrid_data, rerank_fn,
                )
            results[config["key"]] = result

            st.caption(f"{result['elapsed_seconds']}s")

            if result["error"]:
                st.error(result["error"])
            else:
                st.markdown(result["answer"])

            with st.expander(f"Retrieved sources ({len(result['chunks'])})"):
                for chunk in result["chunks"]:
                    st.markdown(
                        f"**[{chunk['rank']}] {chunk['document_name']}**, "
                        f"page {chunk['page_number']} — score {chunk['score']:.4f}"
                    )
                    preview = chunk["content"].replace("\n", " ").strip()
                    if len(preview) > 300:
                        preview = preview[:300] + "..."
                    st.caption(preview)

    if len(selected_configs) > 1:
        st.divider()
        st.subheader("Side-by-side answers")
        st.table({
            config["label"]: [results[config["key"]]["answer"] or f"ERROR: {results[config['key']]['error']}"]
            for config in selected_configs
        })
elif not (question.strip() and selected_configs):
    st.info("Enter a question, choose one or more configurations, then click Run.")