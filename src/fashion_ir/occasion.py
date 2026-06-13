"""Occasion translator.

Lets the user type an occasion (e.g. "wedding") even though the H&M catalogue
has no occasion field. For each occasion in the Rent the Runway data we learn
the clothing words that best characterise it (class-based TF-IDF), keeping only
words that also appear in the H&M catalogue so the expansion actually helps.

At query time, if an occasion word is in the query, we append those terms.
"""

import json
import re

from sklearn.feature_extraction.text import TfidfVectorizer

from .config import OCCASION_MAP_FILE
from .preprocessing import STOPWORDS
from .reviews import load_reviews_df
from .corpus import load_corpus


def _product_vocabulary() -> set:
    """All words (length >= 3) that appear in the product corpus."""
    vocab = set()
    for doc in load_corpus():
        for token in re.sub(r"[^a-z\s]", " ", doc["text"].lower()).split():
            if len(token) >= 3:
                vocab.add(token)
    return vocab


def build_occasion_map(top_n: int = 12, use_product_vocab: bool = True) -> dict:
    """Learn occasion -> characteristic clothing terms, and save it as JSON."""
    df = load_reviews_df()
    df = df[df["rented for"].notna()]

    # Group all review text by occasion; each occasion becomes one "document".
    groups = {}
    for _, row in df.iterrows():
        occ = str(row["rented for"]).strip().lower()
        groups.setdefault(occ, []).append(str(row["review_text"]))
    occasions = list(groups.keys())
    docs = [" ".join(texts) for texts in groups.values()]

    # Class-based TF-IDF: terms frequent in one occasion but rare across the rest.
    vectorizer = TfidfVectorizer(
        stop_words="english", token_pattern=r"[a-zA-Z]{3,}", max_features=20000
    )
    matrix = vectorizer.fit_transform(docs)
    terms = vectorizer.get_feature_names_out()

    product_vocab = _product_vocabulary() if use_product_vocab else None

    mapping = {}
    for i, occ in enumerate(occasions):
        row = matrix[i].toarray()[0]
        ranked = row.argsort()[::-1]            # term indices, highest score first
        chosen = []
        for j in ranked:
            term = terms[j]
            if term in STOPWORDS:
                continue
            if product_vocab is not None and term not in product_vocab:
                continue
            chosen.append(term)
            if len(chosen) >= top_n:
                break
        mapping[occ] = chosen

    OCCASION_MAP_FILE.write_text(json.dumps(mapping, indent=2))
    print(f"Built occasion map for {len(mapping)} occasions -> {OCCASION_MAP_FILE}")
    return mapping


def load_occasion_map() -> dict:
    """Load the saved occasion map (empty dict if it doesn't exist yet)."""
    if not OCCASION_MAP_FILE.exists():
        return {}
    return json.loads(OCCASION_MAP_FILE.read_text())


def expand_with_occasion(query: str, occasion_map: dict, max_terms: int = 8) -> str:
    """If the query mentions an occasion, append that occasion's clothing terms."""
    q_lower = query.lower()
    extra = []
    for occasion, terms in occasion_map.items():
        if re.search(r"\b" + re.escape(occasion) + r"\b", q_lower):
            extra.extend(terms[:max_terms])
    if not extra:
        return query
    # Dedupe while keeping order.
    extra = list(dict.fromkeys(extra))
    return query + " " + " ".join(extra)
