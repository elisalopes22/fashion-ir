"""BM25: lexical (keyword) retrieval."""

import pickle

from rank_bm25 import BM25Okapi

from .config import BM25_FILE
from .preprocessing import preprocess


def build_bm25(documents: list[dict]) -> BM25Okapi:
    """Tokenize all documents, build the BM25 index, save it."""
    tokenized_corpus = [preprocess(doc["text"]) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    return bm25


def load_bm25() -> BM25Okapi:
    """Load the saved BM25 index."""
    with open(BM25_FILE, "rb") as f:
        return pickle.load(f)


def search_bm25(bm25: BM25Okapi, query: str, top_k: int = 10):
    """Return [(doc_index, score), ...] for the top_k documents."""
    scores = bm25.get_scores(preprocess(query))   # query uses same preprocessing
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
