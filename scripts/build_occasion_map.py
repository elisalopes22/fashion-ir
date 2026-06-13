"""Learn the occasion -> clothing-terms map from the reviews. Run once:

    uv run python scripts/build_occasion_map.py

Needs the product corpus (run build_indexes.py first) and data/reviews.json.
Fast: it only reads text, no embeddings.
"""

from fashion_ir.occasion import build_occasion_map


def main():
    mapping = build_occasion_map()
    print("\nLearned terms per occasion:")
    for occ, terms in mapping.items():
        print(f"  {occ}: {', '.join(terms)}")


if __name__ == "__main__":
    main()
