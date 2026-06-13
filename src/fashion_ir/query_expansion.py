"""Query expansion: add a few WordNet synonyms to the query to improve recall.
Example: 'dress' -> 'dress frock gown'."""

import nltk
from nltk.corpus import wordnet


def ensure_wordnet() -> None:
    """Download WordNet on first use."""
    try:
        wordnet.synsets("test")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")


def expand_query(query: str, max_synonyms: int = 2) -> str:
    """Append up to max_synonyms synonyms per word."""
    ensure_wordnet()
    expanded_terms = []
    for word in query.split():
        expanded_terms.append(word)
        synonyms = set()
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                name = lemma.name().replace("_", " ").lower()
                if name != word.lower():
                    synonyms.add(name)
            if len(synonyms) >= max_synonyms:
                break
        expanded_terms.extend(list(synonyms)[:max_synonyms])
    return " ".join(expanded_terms)
