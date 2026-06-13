"""Build all indexes from the dataset. Run once:

    uv run python scripts/build_indexes.py

The embedding step is the slowest part (a few minutes); everything else is fast.
Output goes to the indexes/ folder, which search then loads instantly.
"""

from fashion_ir.corpus import build_and_save
from fashion_ir.bm25_index import build_bm25
from fashion_ir.tfidf_index import build_tfidf
from fashion_ir.semantic_index import build_embeddings, build_faiss


def main():
    documents = build_and_save()

    print("Building BM25 index...")
    build_bm25(documents)
    print("Building TF-IDF index...")
    build_tfidf(documents)

    print("Encoding documents with S-BERT (the slow part)...")
    embeddings = build_embeddings(documents)
    print("Building FAISS index...")
    build_faiss(embeddings, index_type="flat")   # try "ivf" or "hnsw" too

    print("\nAll indexes built. You can now run the demo or evaluation.")


if __name__ == "__main__":
    main()
