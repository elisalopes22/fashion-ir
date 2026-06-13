"""TF-IDF: a sparse baseline ranked by cosine similarity."""

import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import TFIDF_FILE
from .preprocessing import preprocess_joined


def build_tfidf(documents: list[dict]):
    """Fit the vectorizer on the corpus and save (vectorizer, matrix)."""
    texts = [preprocess_joined(doc["text"]) for doc in documents]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(texts)
    with open(TFIDF_FILE, "wb") as f:
        pickle.dump((vectorizer, matrix), f)
    return vectorizer, matrix


def load_tfidf():
    """Load the saved (vectorizer, matrix)."""
    with open(TFIDF_FILE, "rb") as f:
        return pickle.load(f)


def search_tfidf(vectorizer, matrix, query: str, top_k: int = 10):
    """Return [(doc_index, score), ...] for the top_k documents."""
    query_vec = vectorizer.transform([preprocess_joined(query)])
    scores = cosine_similarity(query_vec, matrix)[0]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
