"""Compare all methods on the golden dataset.

    uv run python scripts/run_evaluation.py
"""

from fashion_ir.search import Retriever
from fashion_ir.evaluate import load_golden, compare_methods


def main():
    golden = load_golden()
    print(f"Loaded {len(golden)} golden queries.\n")
    retriever = Retriever(load_cross_encoder=True)   # needed for the +rerank row
    compare_methods(retriever, golden, top_k=10)


if __name__ == "__main__":
    main()
