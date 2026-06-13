"""RAG: retrieve products + opinions, then ask a local LLM (via Ollama) to write
a short, grounded recommendation that explains its choices.
"""

import ollama

from .config import OLLAMA_MODEL
from .search import Retriever
from .reviews import load_review_corpus
from .review_index import load_review_faiss, search_reviews


def build_prompt(query: str, products: list[dict], opinions: list[dict]) -> str:
    """Assemble the context (products + opinions) into one grounded prompt."""
    product_lines = "\n".join(
        f"- {p['metadata']['name']} "
        f"({p['metadata']['colour']} {p['metadata']['type']}, {p['metadata']['section']})"
        for p in products
    )
    opinion_lines = "\n".join(
        f'- "{o["text"][:300]}" '
        f'(worn for: {o["metadata"].get("occasion")}, rating: {o["metadata"].get("rating")})'
        for o in opinions
    )
    return (
        f'A user is shopping and searched for: "{query}".\n\n'
        f"Candidate items from the catalogue:\n{product_lines}\n\n"
        f"Relevant customer opinions:\n{opinion_lines}\n\n"
        "Recommend 2-3 of the candidate items above. Use ONLY the items and "
        "opinions provided; do not invent products or facts. Briefly explain why, "
        "referring to what the opinions suggest. Answer in one short, friendly "
        "paragraph."
    )


class RagRecommender:
    def __init__(self, retriever: Retriever = None):
        # Reuse one Retriever (and its S-BERT model) for both products and reviews.
        self.retriever = retriever or Retriever()
        self.model = self.retriever.model
        self.reviews = load_review_corpus()
        self.review_index = load_review_faiss()

    def recommend(self, query: str, n_products: int = 5, n_opinions: int = 4,
                  model_name: str = OLLAMA_MODEL) -> dict:
        """Return {answer, products, opinions}."""
        # 1. Retrieve the context (occasion expansion helps match the catalogue).
        products = self.retriever.search(
            query, method="hybrid", top_k=n_products, occasion=True
        )
        opinion_hits = search_reviews(self.review_index, self.model, query, n_opinions)
        opinions = [self.reviews[idx] for idx, _ in opinion_hits]

        # 2. Build the prompt and call the local LLM.
        prompt = build_prompt(query, products, opinions)
        resp = ollama.chat(
            model=model_name, messages=[{"role": "user", "content": prompt}]
        )
        # Works whether the client returns a dict or a typed response object.
        answer = resp["message"]["content"] if isinstance(resp, dict) else resp.message.content

        return {"answer": answer, "products": products, "opinions": opinions}
