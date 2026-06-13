"""Reranking: a cross-encoder re-scores the top candidates by looking at the
(query, document) pair together. More accurate, so used only on a short list."""

from sentence_transformers import CrossEncoder

from .config import CROSS_ENCODER_MODEL


def get_cross_encoder() -> CrossEncoder:
    """Load the cross-encoder (only when reranking is used)."""
    return CrossEncoder(CROSS_ENCODER_MODEL)


def rerank(query: str, candidates: list[dict], model: CrossEncoder, top_k: int = 10):
    """Re-score and re-sort candidates. Returns [(document, score), ...]."""
    pairs = [(query, doc["text"]) for doc in candidates]
    scores = model.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return reranked[:top_k]
