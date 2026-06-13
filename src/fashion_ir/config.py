"""All paths, model names and constants in one place."""

from pathlib import Path

# Project root, derived from this file's location.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"          # raw datasets (you download these)
INDEX_DIR = ROOT / "indexes"      # generated indexes (built once, reused)
INDEX_DIR.mkdir(exist_ok=True)

# Raw datasets.
ARTICLES_CSV = DATA_DIR / "articles.csv"   # H&M product metadata
IMAGES_DIR = DATA_DIR / "images"           # H&M images, nested as 0xx/0xxxxxxxxx.jpg
REVIEWS_FILE = DATA_DIR / "reviews.json"   # Rent the Runway opinions (JSON lines)

# Saved index files.
CORPUS_FILE = INDEX_DIR / "corpus.pkl"
BM25_FILE = INDEX_DIR / "bm25.pkl"
TFIDF_FILE = INDEX_DIR / "tfidf.pkl"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
FAISS_FILE = INDEX_DIR / "faiss.index"

# Models.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL = "llama3.2"          # local LLM used for the RAG recommendation

# Opinion corpus (for RAG) index files.
REVIEW_CORPUS_FILE = INDEX_DIR / "reviews_corpus.pkl"
REVIEW_EMBEDDINGS_FILE = INDEX_DIR / "reviews_embeddings.npy"
REVIEW_FAISS_FILE = INDEX_DIR / "reviews_faiss.index"

# Occasion -> clothing terms map (learned from the reviews).
OCCASION_MAP_FILE = INDEX_DIR / "occasion_map.json"

# Optional cap on how many reviews to index (None = all). The reviews dataset
# is large, so set e.g. 50000 if the embedding step feels too slow.
REVIEW_MAX_ROWS = None
