"""Retriever: one interface over all methods (bm25, tfidf, semantic, hybrid),
with optional query expansion and reranking."""

import numpy as np

from .corpus import load_corpus
from .bm25_index import load_bm25, search_bm25
from .tfidf_index import load_tfidf, search_tfidf
from .semantic_index import load_faiss, load_embeddings, get_model, search_semantic
from .query_expansion import expand_query
from .occasion import load_occasion_map, expand_with_occasion
from .rerank import get_cross_encoder, rerank
from .preprocessing import preprocess


def _min_max(scores: np.ndarray) -> np.ndarray:
    """Scale scores to [0, 1] so two methods can be combined fairly."""
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class Retriever:
    def __init__(self, load_cross_encoder: bool = False):
        # Load all saved indexes once (a few seconds).
        self.documents = load_corpus()
        self.bm25 = load_bm25()
        self.tfidf_vectorizer, self.tfidf_matrix = load_tfidf()
        self.faiss_index = load_faiss()
        self.embeddings = load_embeddings()
        self.model = get_model()                 # to embed the query
        self.cross_encoder = get_cross_encoder() if load_cross_encoder else None
        self.occasion_map = load_occasion_map()  # empty if not built yet

    def _bm25(self, query, top_k):
        return search_bm25(self.bm25, query, top_k)

    def _tfidf(self, query, top_k):
        return search_tfidf(self.tfidf_vectorizer, self.tfidf_matrix, query, top_k)

    def _semantic(self, query, top_k):
        return search_semantic(self.faiss_index, self.model, query, top_k)

    def _hybrid(self, query, top_k, alpha=0.5):
        """Blend BM25 and semantic scores over the whole corpus.
        alpha: 1.0 = pure semantic, 0.0 = pure BM25."""
        bm25_scores = np.array(self.bm25.get_scores(preprocess(query)))
        q = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        sem_scores = self.embeddings @ q[0]
        blended = alpha * _min_max(sem_scores) + (1 - alpha) * _min_max(bm25_scores)
        ranked = sorted(enumerate(blended), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def search(
        self,
        query: str,
        method: str = "hybrid",
        top_k: int = 10,
        expand: bool = False,
        occasion: bool = False,
        do_rerank: bool = False,
    ) -> list[dict]:
        """Run a search; return result dicts (id, text, image_path, metadata, score)."""
        if occasion:
            query = expand_with_occasion(query, self.occasion_map)
        if expand:
            query = expand_query(query)

        # Fetch extra candidates if we will rerank them afterwards.
        first_stage_k = max(top_k * 5, 30) if do_rerank else top_k
        if method == "bm25":
            hits = self._bm25(query, first_stage_k)
        elif method == "tfidf":
            hits = self._tfidf(query, first_stage_k)
        elif method == "semantic":
            hits = self._semantic(query, first_stage_k)
        elif method == "hybrid":
            hits = self._hybrid(query, first_stage_k)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Map (index, score) to full document dicts.
        results = [{**self.documents[idx], "score": float(s)} for idx, s in hits]

        if do_rerank:
            if self.cross_encoder is None:
                self.cross_encoder = get_cross_encoder()
            reranked = rerank(query, results, self.cross_encoder, top_k)
            results = [{**doc, "score": float(s)} for doc, s in reranked]
        else:
            results = results[:top_k]

        return results
