
"""Baseline keyword retrieval for SEC filing chunks."""

import re
from collections.abc import Iterable, Mapping
from typing import Any


def _tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def keyword_search(
    query: str,
    chunks: Iterable[Mapping[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return the top matching filing chunks using keyword overlap.

    Each chunk should contain a text field. Other metadata fields are
    preserved in the returned results.

    Results are ranked by:
    1. Number of query-token occurrences matched in the chunk.
    2. Number of distinct query tokens matched.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored_results: list[tuple[int, int, int, dict[str, Any]]] = []

    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping):
            raise TypeError("Each chunk must be a mapping")

        text = chunk.get("text", "")
        if not isinstance(text, str):
            continue

        chunk_tokens = _tokenize(text)
        if not chunk_tokens:
            continue

        token_counts: dict[str, int] = {}
        for token in chunk_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1

        matched_tokens = set(query_tokens) & set(token_counts)
        if not matched_tokens:
            continue

        occurrence_score = sum(
             token_counts[token] for token in matched_tokens
        )
        distinct_match_count = len(matched_tokens)

        result = dict(chunk)
        result["keyword_score"] = occurrence_score
        result["matched_terms"] = sorted(matched_tokens)

        scored_results.append(
            (occurrence_score, distinct_match_count, -index, result)
        )

    scored_results.sort(key=lambda item: item[:3], reverse=True)

    return [item[3] for item in scored_results[:top_k]]
