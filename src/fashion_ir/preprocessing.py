"""Text preprocessing. Documents and queries use the same function so their
words match in the index."""

import re

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "to", "with",
}


def preprocess(text: str, remove_stopwords: bool = True) -> list[str]:
    """Lowercase, strip punctuation, split into tokens, drop stop-words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def preprocess_joined(text: str) -> str:
    """Same as preprocess() but returns a single string (for TF-IDF)."""
    return " ".join(preprocess(text))
