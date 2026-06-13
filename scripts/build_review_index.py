"""Build the opinion (reviews) index for RAG. Run once:

    uv run python scripts/build_review_index.py

Needs data/reviews.csv. The embedding step is the slow part (a couple of
minutes). Output goes to the indexes/ folder.
"""

from fashion_ir.reviews import build_and_save_reviews
from fashion_ir.review_index import build_review_index


def main():
    documents = build_and_save_reviews()
    print("Encoding reviews with S-BERT (the slow part)...")
    build_review_index(documents)
    print("\nReview index built. RAG is ready to use in the demo.")


if __name__ == "__main__":
    main()
