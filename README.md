# 👗 Fashion-IR — A Text-Based Clothing Retrieval System

Recommends clothing from a natural-language query about colour, garment type,
material and occasion. Each item is a document; search runs on **text only**, and the
item's image is shown in the demo (never indexed).

## Layout

```
fashion-ir/
├── src/fashion_ir/        # the library
│   ├── config.py          # paths + model names
│   ├── preprocessing.py   # shared text/query cleaning
│   ├── corpus.py          # build documents from articles.csv
│   ├── bm25_index.py      # BM25 (lexical)
│   ├── tfidf_index.py     # TF-IDF baseline
│   ├── semantic_index.py  # S-BERT + FAISS (flat / ivf / hnsw)
│   ├── query_expansion.py # WordNet synonyms
│   ├── occasion.py        # occasion -> clothing-terms translator
│   ├── rerank.py          # cross-encoder reranking
│   ├── reviews.py         # opinion corpus (Rent the Runway)
│   ├── review_index.py    # opinion semantic index
│   ├── rag.py             # explained recommendation (local LLM)
│   ├── search.py          # Retriever: all methods behind one interface
│   └── evaluate.py        # nDCG + MAP
├── scripts/
│   ├── build_indexes.py        # build the product indexes (run once)
│   ├── build_review_index.py   # build the opinion index (RAG)
│   ├── build_occasion_map.py   # learn the occasion translator
│   └── run_evaluation.py       # compare methods
├── app/streamlit_app.py   # the live demo (shows images)
├── golden_dataset.csv     # you fill this in (query -> ranked ids)
└── data/                  # articles.csv + images/ + reviews.json (git-ignored)
```

## Datasets

**Products - H&M Personalized Fashion Recommendations** (Kaggle competition
`h-and-m-personalized-fashion-recommendations`). You need `articles.csv`
(rich product descriptions) in `data/`, and the `images/` folder in `data/`
(images are nested as `0xx/0xxxxxxxxx.jpg`). Images are optional for search -
without them the demo simply shows text. `data/` is git-ignored.

**Opinions (for RAG) - Rent the Runway** (from McAuley's UCSD datasets page,
"Clothing Fit" data). Each review has the occasion it was worn for. Save the
decompressed file as `data/reviews.json`.

## Setup

```bash
uv sync
```

## Run (all local)

```bash
# 1. Build the indexes once (embedding step takes a few minutes).
uv run python scripts/build_indexes.py

# 2. Build the opinion index for RAG.
uv run python scripts/build_review_index.py

# 3. Learn the occasion -> clothing-terms map (lets users type an occasion).
uv run python scripts/build_occasion_map.py

# 4. Launch the demo.
uv run streamlit run app/streamlit_app.py
```

## Evaluation

`golden_dataset.csv` holds 10 queries, each with a ranked list of expected
item IDs. To score and compare every method:

```bash
uv run python scripts/run_evaluation.py
```

Prints a MAP / nDCG@10 table across TF-IDF, BM25, semantic, hybrid and
hybrid+expand.

## RAG recommendation (optional, local LLM)

Retrieves products + customer opinions and asks a local LLM to write a short,
grounded recommendation.

1. Install Ollama from https://ollama.com and pull a small model:
   ```bash
   ollama pull llama3.2
   ```
   Keep the Ollama app running (it serves the model in the background).
2. Save the Rent the Runway file as `data/reviews.json`.
3. Build the opinion index once:
   ```bash
   uv run python scripts/build_review_index.py
   ```
4. In the demo, tick **Smart pick (RAG)** and search.

Note: on 8 GB RAM, close other heavy apps; a ~3B model fits but is snug.

## Presentation notes

- **Topic & data:** clothing recommendation; documents are H&M product descriptions; images shown but not indexed.
- **Indexing:** sparse (BM25 / TF-IDF) + dense (S-BERT vectors in FAISS, with
  exact vs approximate index options).
- **Retrieval:** preprocess query → bm25 / semantic / hybrid → optional query
  expansion and reranking.
- **Evaluation:** golden dataset + nDCG / MAP.
- **Challenges:** weak descriptions in the first dataset (switched to H&M);
  broken image paths after moving folders; an ID-type mismatch that zeroed the
  evaluation; combining scores on different scales (min-max in the hybrid).
- **Future work:** ColBERT / SPLADE, and a knowledge graph linking opinions,
  brands and styles.
```
