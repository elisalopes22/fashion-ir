"""Build the corpus from the H&M articles.csv: one document per product.

The H&M descriptions (detail_desc) are real sentences, so the searchable text
is much richer than a simple list of attributes.
"""

import pickle

import pandas as pd

from .config import ARTICLES_CSV, IMAGES_DIR, CORPUS_FILE


def image_path_for(article_id: str) -> str:
    """H&M images are nested by the first 3 digits of the 10-digit id,
    e.g. id 108775015 -> images/010/0108775015.jpg."""
    padded = str(article_id).zfill(10)
    return str(IMAGES_DIR / padded[:3] / f"{padded}.jpg")


def load_metadata(csv_path=ARTICLES_CSV) -> pd.DataFrame:
    """Read articles.csv and keep rows that have a real description."""
    # Read article_id as text to preserve the leading-zero image naming.
    df = pd.read_csv(csv_path, dtype={"article_id": str})
    df = df.dropna(subset=["prod_name", "product_type_name",
                           "colour_group_name", "detail_desc"]).reset_index(drop=True)
    return df


def build_description(row: pd.Series) -> str:
    """Build the searchable text from name, attributes and the rich detail_desc."""
    return (
        f"{row['prod_name']}. {row['colour_group_name']} "
        f"{row['product_type_name']} "
        f"({row['garment_group_name']}, {row['index_name']}). "
        f"{row['detail_desc']}"
    )


def build_corpus(df: pd.DataFrame) -> list[dict]:
    """Make one document dict per product."""
    documents = []
    for _, row in df.iterrows():
        documents.append(
            {
                "id": row["article_id"],
                "text": build_description(row),
                "image_path": image_path_for(row["article_id"]),  # shown, not indexed
                "metadata": {
                    "name": row["prod_name"],
                    "type": row["product_type_name"],
                    "colour": row["colour_group_name"],
                    "section": row.get("section_name"),
                    "index": row.get("index_name"),
                    "appearance": row.get("graphical_appearance_name"),
                },
            }
        )
    return documents


def save_corpus(documents: list[dict]) -> None:
    with open(CORPUS_FILE, "wb") as f:
        pickle.dump(documents, f)


def load_corpus() -> list[dict]:
    with open(CORPUS_FILE, "rb") as f:
        return pickle.load(f)


def build_and_save() -> list[dict]:
    """Read CSV, build corpus, save, return it."""
    df = load_metadata()
    documents = build_corpus(df)
    save_corpus(documents)
    print(f"Built {len(documents)} documents -> {CORPUS_FILE}")
    return documents
