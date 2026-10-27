"""Hybrid semantic and BM25 retrieval for FinMind."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from rank_bm25 import BM25Okapi

from src.retrieval.semantic_search import semantic_search


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words and numbers."""
    return re.findall(r"\b\w+\b", text.lower())


def _chunk_key(chunk: Mapping[str, Any]) -> str:
    """Return a stable identity for a retrieved chunk."""
    chunk_id = chunk.get("chunk_id")
    if chunk_id is not None:
        return str(chunk_id)

    return "|".join(
        str(chunk.get(field, ""))
        for field in ("ticker", "filing_date", "filename", "chunk_index", "text")
    )


def hybrid_search(
    query: str,
    chunks: Sequence[Mapping[str, Any]],
    model: Any,
    *,
    top_k: int = 5,
    candidate_k: int = 30,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Combine semantic and BM25 rankings using weighted reciprocal-rank fusion.

    Args:
        query: Natural-language search query.
        chunks: Candidate document chunks containing a text field.
        model: Existing embedding model used by semantic_search.
        top_k: Maximum number of final results.
        candidate_k: Maximum candidates considered from each retriever.
        dense_weight: Weight applied to semantic ranks.
        sparse_weight: Weight applied to BM25 ranks.
        rrf_k: Rank constant used for reciprocal-rank fusion.

    Returns:
        Copies of matching chunks, ordered by fused score. Each result
        includes semantic_score, bm25_score, and hybrid_score.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        return []

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1")
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")

    if dense_weight < 0 or sparse_weight < 0:
        raise ValueError("Retrieval weights cannot be negative")
    if dense_weight + sparse_weight <= 0:
        raise ValueError("At least one retrieval weight must be positive")

    valid_chunks = [
        dict(chunk)
        for chunk in chunks
        if isinstance(chunk.get("text"), str) and chunk["text"].strip()
    ]
    if not valid_chunks:
        return []

    # Normalize weights so they sum to one.
    total_weight = dense_weight + sparse_weight
    dense_weight /= total_weight
    sparse_weight /= total_weight

    limit = min(candidate_k, len(valid_chunks))
    results: dict[str, dict[str, Any]] = {
        _chunk_key(chunk): chunk for chunk in valid_chunks
    }
    fused_scores = {key: 0.0 for key in results}
    semantic_scores: dict[str, float] = {}
    bm25_scores: dict[str, float] = {}

    # Dense semantic ranking.
    if dense_weight > 0:
        dense_results = semantic_search(
            query,
            valid_chunks,
            model,
            top_k=limit,
        )
        for rank, item in enumerate(dense_results, start=1):
            key = _chunk_key(item)
            if key not in results:
                continue
            semantic_scores[key] = float(item.get("semantic_score", 0.0))
            fused_scores[key] += dense_weight / (rrf_k + rank)

    # Sparse lexical ranking.
    if sparse_weight > 0:
        tokenized_docs = [_tokenize(chunk["text"]) for chunk in valid_chunks]
        query_tokens = _tokenize(query)

        if query_tokens and any(tokenized_docs):
            bm25 = BM25Okapi(tokenized_docs)
            raw_scores = bm25.get_scores(query_tokens)

            ranked_indices = sorted(
                range(len(valid_chunks)),
                key=lambda index: float(raw_scores[index]),
                reverse=True,
            )

            positive_indices = [
                index
                for index in ranked_indices
                if float(raw_scores[index]) > 0
            ][:limit]

            for rank, index in enumerate(positive_indices, start=1):
                chunk = valid_chunks[index]
                key = _chunk_key(chunk)
                bm25_scores[key] = float(raw_scores[index])
                fused_scores[key] += sparse_weight / (rrf_k + rank)

    # Return only chunks found by at least one retriever.
    matched_keys = [
        key for key, score in fused_scores.items() if score > 0
    ]
    matched_keys.sort(key=lambda key: fused_scores[key], reverse=True)

    output: list[dict[str, Any]] = []
    for key in matched_keys[:top_k]:
        result = dict(results[key])
        result["semantic_score"] = semantic_scores.get(key, 0.0)
        result["bm25_score"] = bm25_scores.get(key, 0.0)
        result["hybrid_score"] = fused_scores[key]
        output.append(result)

    return output
