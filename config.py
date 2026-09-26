"""
Central configuration for the insurance RAG pipeline (dense + hybrid retrieval)
and its evaluation scripts.

Secrets (endpoint, API key, deployment names) stay in .env — this file holds
only the tunable, non-secret parameters, so they can be changed and reviewed
in one place instead of being scattered across scripts.

Place this file at the repository root: insurance_rag_poc_azure/config.py
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPOSITORY_ROOT = Path(__file__).resolve().parent

DOCUMENTS_DIRECTORY = REPOSITORY_ROOT / "documents"
DENSE_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "rag_data" / "dense"
HYBRID_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "rag_data" / "hybrid"
EVALUATION_DIRECTORY = REPOSITORY_ROOT / "evaluation"
RESULTS_DIRECTORY = EVALUATION_DIRECTORY / "results"

EMBEDDINGS_FILENAME = "embeddings.npy"
METADATA_FILENAME = "metadata.json"
BM25_CORPUS_FILENAME = "bm25_corpus.json"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
# Dense and hybrid currently share one extraction/chunking step
# (extract_all_documents() is called by both build_dense_index() and
# build_hybrid_index()), so both approaches are evaluated on identical
# chunks — the only difference is how those chunks are retrieved. Keeping
# one shared chunk config preserves that apples-to-apples comparison.
#
# If you later want to test chunk size as its own variable per approach,
# add DENSE_CHUNK_SIZE / HYBRID_CHUNK_SIZE here and update
# extract_all_documents() in base.py to accept a size/overlap argument.
CHUNK_SIZE = 3000        # characters
CHUNK_OVERLAP = 500      # characters; must be < CHUNK_SIZE

# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------
EMBEDDING_BATCH_SIZE = 16
EMBEDDING_MAX_RETRIES = 3
EMBEDDING_RETRY_BACKOFF_SECONDS = 5   # multiplied by attempt number

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 5

# Default retrieval mode for evaluate_rag.py when --mode is not passed on the
# command line. Override per run with --mode dense / --mode hybrid.
DEFAULT_RETRIEVAL_MODE = "hybrid"

# Hybrid fusion weights (dense cosine similarity vs. BM25 lexical score).
# Must sum to 1.0 — validated at import time below.
DENSE_WEIGHT = 0.5
BM25_WEIGHT = 0.5

if abs((DENSE_WEIGHT + BM25_WEIGHT) - 1.0) > 1e-6:
    raise ValueError("DENSE_WEIGHT + BM25_WEIGHT must sum to 1.0")

# ---------------------------------------------------------------------------
# Evaluation (generate_questions.py / evaluate_rag.py)
# ---------------------------------------------------------------------------
EVAL_TARGET_QUESTIONS = 50
EVAL_MAX_DOC_CHARS = 12000
EVAL_TOPUP_ROUNDS = 3
EVAL_TOP_K = TOP_K   # separate knob in case eval should use a different k than production

# ---------------------------------------------------------------------------
# Reranking (BGE cross-encoder, via sentence-transformers)
# ---------------------------------------------------------------------------
# Retrieval over-fetches RERANK_CANDIDATE_POOL chunks; the reranker then scores
# each (question, chunk) pair and returns the best RERANK_TOP_K, discarding the
# retriever's original order. Set RERANK_ENABLED = False to bypass this and use
# the retriever's own top-k, e.g. for an apples-to-apples evaluation comparison.
RERANK_ENABLED = False
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"  # or "BAAI/bge-reranker-large" / "BAAI/bge-reranker-v2-m3"
RERANK_CANDIDATE_POOL = 20   # chunks retrieved before reranking; must be >= RERANK_TOP_K
RERANK_TOP_K = TOP_K         # final chunks returned after reranking
RERANK_BATCH_SIZE = 16
RERANK_DEVICE = "cpu"        # set to "cuda" if a GPU is available

if RERANK_CANDIDATE_POOL < RERANK_TOP_K:
    raise ValueError("RERANK_CANDIDATE_POOL must be >= RERANK_TOP_K")