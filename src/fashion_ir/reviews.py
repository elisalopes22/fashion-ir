"""Build the opinion corpus from the Rent the Runway reviews (used by RAG).

Each review includes the occasion it was worn for (wedding, party, work...),
which matches this project's occasion-based search nicely.
"""

import pickle

import pandas as pd

from .config import REVIEWS_FILE, REVIEW_CORPUS_FILE, REVIEW_MAX_ROWS


def load_reviews_df(path=REVIEWS_FILE) -> pd.DataFrame:
    """Read the reviews JSON (one JSON object per line) and keep rows with text."""
    try:
        df = pd.read_json(path, lines=True)
    except ValueError:
        df = pd.read_json(path)               # fallback for a plain JSON array
    df = df[df["review_text"].notna()].reset_index(drop=True)
    if REVIEW_MAX_ROWS:
        df = df.head(REVIEW_MAX_ROWS)
    return df


def build_review_corpus(df: pd.DataFrame) -> list[dict]:
    """One document per review: id, text (summary + body), and a little metadata."""
    documents = []
    for i, row in df.iterrows():
        summary = str(row.get("review_summary", "") or "")
        body = str(row["review_text"])
        text = f"{summary}. {body}" if summary and summary != "nan" else body
        documents.append(
            {
                "id": int(i),
                "text": text.strip(),
                "metadata": {
                    "rating": row.get("rating"),
                    "occasion": row.get("rented for"),   # note: column has a space
                    "category": row.get("category"),
                    "fit": row.get("fit"),
                },
            }
        )
    return documents


def save_review_corpus(documents: list[dict]) -> None:
    with open(REVIEW_CORPUS_FILE, "wb") as f:
        pickle.dump(documents, f)


def load_review_corpus() -> list[dict]:
    with open(REVIEW_CORPUS_FILE, "rb") as f:
        return pickle.load(f)


def build_and_save_reviews() -> list[dict]:
    """Read JSON, build the opinion corpus, save, return it."""
    df = load_reviews_df()
    documents = build_review_corpus(df)
    save_review_corpus(documents)
    print(f"Built {len(documents)} review documents -> {REVIEW_CORPUS_FILE}")
    return documents
