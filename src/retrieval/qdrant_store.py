
"""Qdrant-backed vector storage for FinMind filing chunks."""

from collections.abc import Mapping, Sequence
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models


DEFAULT_COLLECTION_NAME = "finmind_chunks"
REQUIRED_CHUNK_FIELDS = (
    "chunk_id",
    "company_name",
    "ticker",
    "form_type",
    "filing_date",
    "section",
    "text",
    "source_url",
)


def _validate_chunk(chunk: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy a chunk following the shared FinMind interface."""
    if not isinstance(chunk, Mapping):
        raise TypeError("Each chunk must be a mapping")

    result = dict(chunk)

    for field in REQUIRED_CHUNK_FIELDS:
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Chunk field '{field}' must be a non-empty string")

    if "period_end" not in result:
        result["period_end"] = None

    if result["period_end"] is not None and not isinstance(
        result["period_end"], str
    ):
        raise TypeError("period_end must be a string or None")

    # Qdrant payloads must be JSON-compatible; omit optional null values.
    return {key: value for key, value in result.items() if value is not None}


def _point_id(chunk_id: str) -> str:
    """Produce a stable Qdrant-compatible UUID for a FinMind chunk ID."""
    return str(uuid5(NAMESPACE_URL, f"finmind:chunk:{chunk_id}"))


def _as_vector(vector: Any) -> list[float]:
    """Convert an embedding to a list of Python floats."""
    if hasattr(vector, "tolist"):
        vector = vector.tolist()

    values = [float(value) for value in vector]
    if not values:
        raise ValueError("Embedding vectors must not be empty")
    return values


class QdrantChunkStore:
    """Store and search FinMind filing chunks using Qdrant."""

    def __init__(
        self,
        client: QdrantClient,
        *,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        vector_size: int,
    ) -> None:
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string")
        if isinstance(vector_size, bool) or not isinstance(vector_size, int):
            raise TypeError("vector_size must be an integer")
        if vector_size < 1:
            raise ValueError("vector_size must be at least 1")

        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size

    def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        if self.client.collection_exists(self.collection_name):
            info = self.client.get_collection(self.collection_name)
            configured_size = info.config.params.vectors.size
            if configured_size != self.vector_size:
                raise ValueError(
                    f"Collection '{self.collection_name}' has vector size "
                    f"{configured_size}; expected {self.vector_size}"
                )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self,
        chunks: Sequence[Mapping[str, Any]],
        embeddings: Sequence[Any],
    ) -> int:
        """Insert or replace chunks and their vectors; return count written."""
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")

        if not chunks:
            return 0

        self.ensure_collection()
        points: list[models.PointStruct] = []
        seen_ids: set[str] = set()

        for chunk, embedding in zip(chunks, embeddings):
            payload = _validate_chunk(chunk)
            chunk_id = payload["chunk_id"]

            if chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk_id in batch: {chunk_id}")
            seen_ids.add(chunk_id)

            vector = _as_vector(embedding)
            if len(vector) != self.vector_size:
                raise ValueError(
                    f"Embedding for '{chunk_id}' has {len(vector)} dimensions; "
                    f"expected {self.vector_size}"
                )

            points.append(
                models.PointStruct(
                    id=_point_id(chunk_id),
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def search(
        self,
        query_embedding: Any,
        *,
        top_k: int = 5,
        filters: Mapping[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search vectors and return chunk metadata with similarity scores."""
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k must be an integer")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        vector = _as_vector(query_embedding)
        if len(vector) != self.vector_size:
            raise ValueError(
                f"Query embedding has {len(vector)} dimensions; "
                f"expected {self.vector_size}"
            )

        if not self.client.collection_exists(self.collection_name):
            return []

        query_filter = None
        if filters:
            conditions = []
            for field, value in filters.items():
                if not isinstance(field, str) or not field.strip():
                    raise ValueError("Filter field names must be non-empty strings")
                if not isinstance(value, str):
                    raise TypeError("Filter values must be strings")
                conditions.append(
                    models.FieldCondition(
                        key=field,
                        match=models.MatchValue(value=value),
                    )
                )
            query_filter = models.Filter(must=conditions)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        results: list[dict[str, Any]] = []
        for point in response.points:
            payload = dict(point.payload or {})
            if "period_end" not in payload:
                payload["period_end"] = None
            payload["semantic_score"] = float(point.score)
            results.append(payload)

        return results

    def get_chunks_by_ids(
        self,
        chunk_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Retrieve specific chunks by chunk_id from Qdrant without scanning the corpus.

        Uses deterministic UUIDv5 point IDs. Safely handles missing IDs by returning
        only those chunks that exist in the collection, preserving all metadata fields.
        """
        if not chunk_ids:
            return []

        if not self.client.collection_exists(self.collection_name):
            return []

        clean_ids = [str(cid).strip() for cid in chunk_ids if str(cid).strip()]
        if not clean_ids:
            return []

        point_ids = [_point_id(cid) for cid in clean_ids]
        records = self.client.retrieve(
            collection_name=self.collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )

        results: list[dict[str, Any]] = []
        for record in records:
            payload = dict(record.payload or {})
            if "chunk_id" in payload:
                if "period_end" not in payload:
                    payload["period_end"] = None
                results.append(payload)
        return results
