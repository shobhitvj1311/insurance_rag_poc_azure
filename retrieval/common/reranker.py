"""
Cross-encoder reranking using a BGE reranker model (BAAI/bge-reranker-*).

Retrieval (dense or hybrid) over-fetches a wider candidate pool
(config.RERANK_CANDIDATE_POOL); this module scores every (question, chunk)
pair with the cross-encoder and returns the top config.RERANK_TOP_K in the
new order, discarding the retriever's original ranking.

pip install sentence-transformers
"""

import sys
from pathlib import Path

# this file lives at <repo_root>/retrieval/common/reranker.py
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from config import RERANK_BATCH_SIZE, RERANK_DEVICE, RERANK_MODEL_NAME, RERANK_TOP_K

_model = None  # lazy singleton — loading a cross-encoder is expensive, load once


def get_reranker_model():
    """Load and cache the BGE cross-encoder model."""

    global _model

    if _model is None:
        from sentence_transformers import CrossEncoder

        print(f"Loading reranker model: {RERANK_MODEL_NAME} ({RERANK_DEVICE})...")
        _model = CrossEncoder(RERANK_MODEL_NAME, device=RERANK_DEVICE)

    return _model


def rerank_chunks(question, candidates, top_k=RERANK_TOP_K):
    """
    Rerank a candidate pool of retrieved chunks against the question using a
    BGE cross-encoder, and return the top_k in the new order.

    candidates: list of chunk dicts (as returned by retrieve_dense_chunks /
                retrieve_hybrid_chunks), each with a "content" key.
    """

    if not candidates:
        return []

    model = get_reranker_model()
    pairs = [(question, candidate["content"]) for candidate in candidates]
    scores = model.predict(pairs, batch_size=RERANK_BATCH_SIZE)

    scored_candidates = list(zip(candidates, scores))
    scored_candidates.sort(key=lambda pair: pair[1], reverse=True)
    top_candidates = scored_candidates[:top_k]

    results = []
    for rank, (chunk, score) in enumerate(top_candidates, start=1):
        result = chunk.copy()
        result["retrieval_rank"] = chunk.get("rank")   # position before reranking
        result["retrieval_score"] = chunk.get("score")  # score before reranking
        result["rank"] = rank
        result["rerank_score"] = float(score)
        result["score"] = float(score)  # downstream code (print/build_context) reads "score"
        results.append(result)

    return results