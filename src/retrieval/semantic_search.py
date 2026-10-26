
"""Semantic retrieval for SEC filing chunks."""

from collections.abc import Iterable, Mapping
from math import sqrt
from typing import Any


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """Load the local sentence-transformer model on demand."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Semantic retrieval requires sentence-transformers. "
            "Install it with: python -m pip install sentence-transformers"
        ) from exc

    return SentenceTransformer(model_name)


def _as_list(vector: Any) -> list[float]:
    """Convert a model vector to a regular Python list."""
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def _cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(vector_a) != len(vector_b):
        raise ValueError("Embedding vectors must have equal dimensions")

    magnitude_a = sqrt(sum(value * value for value in vector_a))
    magnitude_b = sqrt(sum(value * value for value in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )
    return dot_product / (magnitude_a * magnitude_b)


def semantic_search(
    query: str,
    chunks: Iterable[Mapping[str, Any]],
    model: Any,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Rank text chunks by embedding cosine similarity.

    The model must provide an encode(texts) method.
    Each chunk should contain a text field. Other metadata is preserved.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if not query.strip():
        return []

    valid_chunks: list[dict[str, Any]] = []

    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            raise TypeError("Each chunk must be a mapping")

        text = chunk.get("text", "")
        if isinstance(text, str) and text.strip():
            valid_chunks.append(dict(chunk))

    if not valid_chunks:
        return []

    texts = [chunk["text"] for chunk in valid_chunks]

    # Encode the query and documents in one batch.
    embeddings = model.encode([query, *texts])

    if len(embeddings) != len(texts) + 1:
        raise ValueError("The embedding model returned an unexpected result")

    query_vector = _as_list(embeddings[0])
    scored_results: list[tuple[float, int, dict[str, Any]]] = []

    for index, (chunk, embedding) in enumerate(
        zip(valid_chunks, embeddings[1:])
    ):
        document_vector = _as_list(embedding)
        score = _cosine_similarity(query_vector, document_vector)

        result = dict(chunk)
        result["semantic_score"] = score
        scored_results.append((score, -index, result))

    # Higher similarity first; preserve input order for ties.
    scored_results.sort(key=lambda item: (item[0], item[1]), reverse=True)

    return [item[2] for item in scored_results[:top_k]]
