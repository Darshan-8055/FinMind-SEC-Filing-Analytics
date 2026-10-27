
"""Evaluation metrics for information retrieval."""

from collections.abc import Iterable, Mapping
from typing import Any

from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.keyword_search import keyword_search
from src.retrieval.semantic_search import semantic_search


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


def _retrieved_ids(results: list[dict[str, Any]]) -> list[str]:
    """Extract stable chunk IDs from retrieval results."""
    return [
        str(result["chunk_id"])
        for result in results
        if result.get("chunk_id") is not None
    ]


def _score_results(
    query: str,
    results: list[dict[str, Any]],
    relevant_ids: set[str],
    k: int,
) -> dict[str, Any]:
    """Calculate evaluation metrics for one ranked result list."""
    retrieved_ids = _retrieved_ids(results)

    return {
        "query": query,
        "retrieved_chunk_ids": retrieved_ids,
        "relevant_chunk_ids": sorted(relevant_ids),
        "precision_at_k": precision_at_k(retrieved_ids, relevant_ids, k),
        "recall_at_k": recall_at_k(retrieved_ids, relevant_ids, k),
    }


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
    return _score_results(query, results, set(relevant_ids), k)


def evaluate_all_retrievers(
    query: str,
    chunks: Iterable[Mapping[str, Any]],
    relevant_ids: Iterable[str],
    model: Any,
    k: int = 3,
    candidate_k: int = 30,
) -> dict[str, Any]:
    """Evaluate keyword, semantic, and hybrid retrieval for one query."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1")

    chunk_list = list(chunks)
    relevant = set(relevant_ids)

    keyword_results = keyword_search(query, chunk_list, top_k=k)
    semantic_results = semantic_search(
        query, chunk_list, model, top_k=k
    )
    hybrid_results = hybrid_search(
        query,
        chunk_list,
        model,
        top_k=k,
        candidate_k=candidate_k,
    )

    return {
        "query": query,
        "keyword": _score_results(query, keyword_results, relevant, k),
        "semantic": _score_results(query, semantic_results, relevant, k),
        "hybrid": _score_results(query, hybrid_results, relevant, k),
    }
