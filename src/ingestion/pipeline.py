
"""End-to-end SEC filing ingestion into Qdrant."""

from collections.abc import Mapping
from typing import Any

from src.ingestion.chunk_adapter import adapt_chunk
from src.ingestion.narrative_parser import parse_filing
from src.retrieval.qdrant_store import QdrantChunkStore


class FilingIngestionPipeline:
    """Parse a filing, adapt its chunks, embed them, and store them."""

    def __init__(self, store: QdrantChunkStore, model: Any) -> None:
        self.store = store
        self.model = model

        dimension = model.get_sentence_embedding_dimension()
        if dimension != store.vector_size:
            raise ValueError(
                f"Embedding dimension {dimension} does not match "
                f"Qdrant vector size {store.vector_size}"
            )

    def ingest_filing(
        self,
        text: str,
        *,
        filename: str,
        filing_type: str,
        ticker: str,
        company_name: str,
        filing_date: str,
        source_url: str,
        period_end: str | None = None,
    ) -> int:
        """Parse and store one filing; return the number of chunks stored."""
        parsed_chunks = parse_filing(
            text=text,
            filename=filename,
            filing_type=filing_type,
        )

        if not parsed_chunks:
            return 0

        adapted_chunks = []
        for parsed in parsed_chunks:
            chunk_id = f"{ticker.upper()}:{filing_date}:{filename}:{parsed['chunk_index']}"

            adapted_chunks.append(
                adapt_chunk(
                    parsed,
                    chunk_id=chunk_id,
                    ticker=ticker,
                    company_name=company_name,
                    filing_date=filing_date,
                    source_url=source_url,
                    period_end=period_end,
                )
            )

        embeddings = self.model.encode(
            [chunk["text"] for chunk in adapted_chunks]
        )

        return self.store.upsert_chunks(adapted_chunks, embeddings)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Embed a search query and retrieve matching filing chunks."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            return []

        query_embedding = self.model.encode([query])[0]
        return self.store.search(
            query_embedding,
            top_k=top_k,
            filters=filters,
        )
