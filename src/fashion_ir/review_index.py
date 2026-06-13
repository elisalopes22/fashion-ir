"""Semantic index for the opinion corpus. Reuses the same S-BERT model, so
reviews and products live in the same vector space."""

import numpy as np
import faiss

from .semantic_index import get_model
from .config import REVIEW_EMBEDDINGS_FILE, REVIEW_FAISS_FILE


def build_review_index(documents: list[dict], model=None, batch_size: int = 64):
    """Embed every review and save embeddings + a FAISS index."""
    model = model or get_model()
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    np.save(REVIEW_EMBEDDINGS_FILE, embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(REVIEW_FAISS_FILE))
    return embeddings, index


def load_review_faiss():
    """Load the saved review FAISS index."""
    return faiss.read_index(str(REVIEW_FAISS_FILE))


def search_reviews(index, model, query: str, top_k: int = 5):
    """Return [(review_index, score), ...] for the most relevant opinions."""
    query_vec = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, idxs = index.search(query_vec, top_k)
    return list(zip(idxs[0].tolist(), scores[0].tolist()))
