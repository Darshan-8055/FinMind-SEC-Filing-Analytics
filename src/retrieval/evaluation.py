
"""Evaluation metrics for information retrieval.

Metric conventions
------------------
- Precision@K: (relevant hits in top-K) / K.
  Returns 0.0 when the retrieved list is empty; denominator K is always
  a positive integer so division by zero cannot occur.
- Recall@K: (relevant hits in top-K) / |relevant set|.
  Returns 0.0 when the relevant set is empty; the denominator is the size
  of the relevant set which is checked before division.
- Macro-average: each query is weighted equally.
  Returns zeros when the query list is empty.
"""

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
    """Fraction of the first k retrieved results that are relevant.

    Returns 0.0 when *retrieved_ids* is empty.
    Raises ValueError when k < 1.
    """
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
    """Fraction of all known relevant results found in the first k.

    Returns 0.0 when *relevant_ids* is empty (recall is undefined; the
    convention is to treat it as 0.0 to avoid division by zero).
    Raises ValueError when k < 1.
    """
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
    """Run keyword retrieval and calculate precision and recall.

    This is the baseline keyword-only evaluation function.  Use
    :func:`evaluate_all_retrievers` to compare all three methods at once.
    """
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
    """Evaluate keyword, semantic, and hybrid retrieval for one query.

    All three methods use the same relevance judgments and the same
    evaluation cut-off *k* so results are directly comparable.

    Returns a dict with keys ``"query"``, ``"keyword"``, ``"semantic"``,
    and ``"hybrid"``.  Each method value is the output of
    :func:`_score_results`.
    """
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


def macro_average_metrics(
    per_query_results: list[dict[str, Any]],
) -> dict[str, float]:
    """Return macro-average Precision@K and Recall@K across queries.

    Each element of *per_query_results* must contain ``precision_at_k`` and
    ``recall_at_k`` keys (as produced by :func:`_score_results`).

    The macro-average weights every query equally regardless of the size of
    its relevant set.  Returns zeros when the input list is empty so callers
    never encounter a division by zero.

    Returns
    -------
    dict with keys ``precision_at_k`` and ``recall_at_k``.
    """
    if not per_query_results:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0}

    n = len(per_query_results)
    avg_precision = sum(r["precision_at_k"] for r in per_query_results) / n
    avg_recall = sum(r["recall_at_k"] for r in per_query_results) / n
    return {"precision_at_k": avg_precision, "recall_at_k": avg_recall}


def evaluate_all_queries(
    queries: Iterable[Mapping[str, Any]],
    chunks: Iterable[Mapping[str, Any]],
    model: Any,
    k: int = 3,
    candidate_k: int = 30,
) -> dict[str, Any]:
    """Evaluate all three retrievers across every query in *queries*.

    Each element of *queries* must have:

    - ``query`` (str): the natural-language query text.
    - ``relevant_chunk_ids`` (list[str]): ground-truth relevant chunk IDs.

    Parameters
    ----------
    queries:
        Iterable of query dicts (as stored in the retrieval eval fixture).
    chunks:
        All candidate chunks to search through.
    model:
        Embedding model with an ``encode(texts)`` method.
    k:
        Evaluation cut-off for Precision@K and Recall@K.
    candidate_k:
        Maximum candidates passed to the dense and sparse retrievers inside
        hybrid search.

    Returns
    -------
    dict with:

    - ``k``: the evaluation cut-off used.
    - ``per_query``: list of per-query dicts from
      :func:`evaluate_all_retrievers`.
    - ``macro_average``: dict keyed by method name (``"keyword"``,
      ``"semantic"``, ``"hybrid"``), each holding the macro-average
      Precision@K and Recall@K from :func:`macro_average_metrics`.

    Notes
    -----
    Results produced from the synthetic fixture in
    ``tests/fixtures/retrieval_eval.json`` reflect a small, hand-crafted
    chunk set.  They should not be reported as production retrieval quality
    figures.  Real-data validation requires a live Qdrant service and actual
    SEC filing embeddings.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1")

    chunk_list = list(chunks)
    per_query: list[dict[str, Any]] = []

    for query_item in queries:
        result = evaluate_all_retrievers(
            query=query_item["query"],
            chunks=chunk_list,
            relevant_ids=query_item["relevant_chunk_ids"],
            model=model,
            k=k,
            candidate_k=candidate_k,
        )
        per_query.append(result)

    macro: dict[str, dict[str, float]] = {}
    for method in ("keyword", "semantic", "hybrid"):
        method_results = [r[method] for r in per_query]
        macro[method] = macro_average_metrics(method_results)

    return {
        "k": k,
        "per_query": per_query,
        "macro_average": macro,
    }
