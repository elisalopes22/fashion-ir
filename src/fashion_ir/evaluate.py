"""Evaluation: compare methods on the golden dataset with nDCG and MAP."""

import csv
import math

from .config import ROOT

GOLDEN_FILE = ROOT / "golden_dataset.csv"


def load_golden(path=GOLDEN_FILE) -> list[dict]:
    """Read golden_dataset.csv -> [{query, relevant_ids (best-first)}, ...]."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ids = [int(x) for x in row["relevant_ids"].split(",") if x.strip()]
            rows.append({"query": row["query"].strip(), "relevant_ids": ids})
    return rows


def average_precision(retrieved_ids: list[int], relevant_ids: list[int]) -> float:
    """Average Precision for one query (binary relevance)."""
    if not relevant_ids:
        return 0.0
    hits = 0
    score = 0.0
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            hits += 1
            score += hits / rank
    return score / len(relevant_ids)


def ndcg_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    """nDCG@k; relevance grade comes from the position in the golden ranking."""
    gain = {doc_id: len(relevant_ids) - i for i, doc_id in enumerate(relevant_ids)}

    def dcg(ids):
        return sum(
            gain.get(doc_id, 0) / math.log2(rank + 1)
            for rank, doc_id in enumerate(ids[:k], start=1)
        )

    actual = dcg(retrieved_ids)
    ideal = dcg(relevant_ids)
    return actual / ideal if ideal > 0 else 0.0


def evaluate_method(retriever, golden, method, top_k=10, **search_kwargs) -> dict:
    """Average MAP and nDCG over all golden queries for one method."""
    aps, ndcgs = [], []
    for item in golden:
        results = retriever.search(
            item["query"], method=method, top_k=top_k, **search_kwargs
        )
        retrieved_ids = [r["id"] for r in results]
        aps.append(average_precision(retrieved_ids, item["relevant_ids"]))
        ndcgs.append(ndcg_at_k(retrieved_ids, item["relevant_ids"], top_k))
    n = len(golden)
    return {"method": method, "MAP": sum(aps) / n, f"nDCG@{top_k}": sum(ndcgs) / n}


def compare_methods(retriever, golden, top_k=10) -> list[dict]:
    """Evaluate several configurations and print a comparison table."""
    configs = [
        ("tfidf", {}),
        ("bm25", {}),
        ("semantic", {}),
        ("hybrid", {}),
        ("hybrid", {"expand": True}),
        ("hybrid", {"do_rerank": True}),
    ]
    rows = []
    for method, kwargs in configs:
        res = evaluate_method(retriever, golden, method, top_k, **kwargs)
        label = method + ("+expand" if kwargs.get("expand") else "")
        label += "+rerank" if kwargs.get("do_rerank") else ""
        res["config"] = label
        rows.append(res)

    print(f"{'config':<20}{'MAP':>10}{'nDCG@'+str(top_k):>12}")
    print("-" * 42)
    for r in rows:
        print(f"{r['config']:<20}{r['MAP']:>10.3f}{r[f'nDCG@{top_k}']:>12.3f}")
    return rows
