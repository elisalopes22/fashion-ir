"""Semantic search: embed text with S-BERT, store vectors in FAISS.

Index types:
  flat = exact search (compares query to every vector).
  ivf  = clusters vectors, searches only the nearest clusters (faster, approximate).
  hnsw = graph-based search (fast queries, more memory).
Vectors are normalized, so inner product == cosine similarity.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, EMBEDDINGS_FILE, FAISS_FILE


def get_model() -> SentenceTransformer:
    """Load the embedding model (uses GPU automatically if present)."""
    return SentenceTransformer(EMBEDDING_MODEL)


def build_embeddings(documents: list[dict], model=None, batch_size: int = 64):
    """Encode every document into a vector and save the matrix. (Slowest step.)"""
    model = model or get_model()
    texts = [doc["text"] for doc in documents]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")
    np.save(EMBEDDINGS_FILE, embeddings)
    return embeddings


def build_faiss(embeddings: np.ndarray, index_type: str = "flat"):
    """Build a FAISS index of the chosen type and save it."""
    dim = embeddings.shape[1]

    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

    elif index_type == "ivf":
        nlist = 256                       # number of clusters
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)           # learn the clusters
        index.add(embeddings)
        index.nprobe = 16                 # clusters to search per query

    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.add(embeddings)

    else:
        raise ValueError(f"Unknown index_type: {index_type}")

    faiss.write_index(index, str(FAISS_FILE))
    return index


def load_embeddings() -> np.ndarray:
    """Load the saved vectors (used by the hybrid method)."""
    return np.load(EMBEDDINGS_FILE)


def load_faiss():
    """Load the saved FAISS index."""
    return faiss.read_index(str(FAISS_FILE))


def search_semantic(index, model, query: str, top_k: int = 10):
    """Embed the query and return [(doc_index, score), ...] from FAISS."""
    query_vec = model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, idxs = index.search(query_vec, top_k)
    return list(zip(idxs[0].tolist(), scores[0].tolist()))
