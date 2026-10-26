
"""Evaluation metrics for information retrieval."""

from collections.abc import Iterable, Mapping
from typing import Any

from src.retrieval.keyword_search import keyword_search


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """Fraction of the first k retrieved results that are relevant."""
    if k < 1:
        raise ValueError("k must be at least 1")

    relevant = set(relevant_ids)
    if not retrieved_ids:
        return 0.0

    top_results = retrieved_ids[:k]
    return sum(item in relevant for item in top_results) / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: Iterable[str],
    k: int,
) -> float:
    """Fraction of all known relevant results found in the first k."""
    if k < 1:
        raise ValueError("k must be at least 1")

    relevant = set(relevant_ids)
    if not relevant:
        return 0.0

    top_results = set(retrieved_ids[:k])
    return len(top_results & relevant) / len(relevant)


def evaluate_query(
    query: str,
    chunks: Iterable[Mapping[str, Any]],
    relevant_ids: Iterable[str],
    k: int = 3,
) -> dict[str, Any]:
    """Run keyword retrieval and calculate precision and recall."""
    if k < 1:
        raise ValueError("k must be at least 1")

    chunk_list = list(chunks)
    results = keyword_search(query, chunk_list, top_k=k)
    retrieved_ids = [
        str(result["chunk_id"])
        for result in results
        if result.get("chunk_id") is not None
    ]
    relevant = set(relevant_ids)

    return {
        "query": query,
        "retrieved_chunk_ids": retrieved_ids,
        "relevant_chunk_ids": sorted(relevant),
        "precision_at_k": precision_at_k(retrieved_ids, relevant, k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant, k),
    }
