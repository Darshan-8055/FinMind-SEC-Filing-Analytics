
"""End-to-end SEC filing ingestion and hybrid retrieval into Qdrant."""

import logging
from collections.abc import Mapping
from typing import Any

from src.ingestion.chunk_adapter import adapt_chunk
from src.ingestion.narrative_parser import parse_filing
from src.retrieval.qdrant_store import QdrantChunkStore
from src.retrieval.sparse_index import SparseKeywordIndex

logger = logging.getLogger(__name__)


class FilingIngestionPipeline:
    """Parse a filing, adapt its chunks, embed them, and store them."""

    def __init__(
        self,
        store: QdrantChunkStore,
        model: Any,
        sparse_index: SparseKeywordIndex | None = None,
    ) -> None:
        self.store = store
        self.model = model
        self.sparse_index = (
            sparse_index if sparse_index is not None else SparseKeywordIndex()
        )

        dimension = model.get_sentence_embedding_dimension()
        if dimension != store.vector_size:
            raise ValueError(
                f"Embedding dimension {dimension} does not match "
                f"Qdrant vector size {store.vector_size}"
            )

    def check_reconciliation_needed(self) -> tuple[bool, str]:
        """Check whether the sparse index requires reconciliation with Qdrant."""
        if self.sparse_index is None:
            return True, "No sparse index configured"
        return self.sparse_index.check_reconciliation_needed(self.store)

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
        """Parse, embed, and store one filing in Qdrant and the sparse index.

        Ordering: Qdrant vector upsert executes first. If successful, chunks are
        indexed into the persistent sparse index.
        """
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

        # 1. Synchronous Qdrant upsert
        count = self.store.upsert_chunks(adapted_chunks, embeddings)

        # 2. Update sparse keyword index
        if self.sparse_index is not None and count > 0:
            try:
                self.sparse_index.add_chunks(adapted_chunks)
            except Exception as exc:
                logger.error(
                    "Qdrant upsert succeeded for %s but sparse indexing failed: %s",
                    filename,
                    exc,
                )
                self.sparse_index.set_meta("status", "stale")
                raise

        return count

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """Embed a search query and retrieve matching filing chunks from Qdrant."""
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

    def hybrid_search(
        self,
        query: str,
        *,
        top_k: int = 5,
        filters: Mapping[str, str] | None = None,
        candidate_k: int = 30,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Perform scalable Qdrant-backed hybrid retrieval using weighted RRF.

        Combines dense vector retrieval from Qdrant with BM25 sparse keyword
        retrieval, fusing candidate rankings using Reciprocal Rank Fusion (RRF).
        Does not require loading the entire corpus into memory.
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

        dense_enabled = dense_weight > 0
        sparse_enabled = sparse_weight > 0

        dense_candidates: list[dict[str, Any]] = []
        dense_success = False
        if dense_enabled:
            try:
                dense_candidates = self.search(
                    query,
                    top_k=candidate_k,
                    filters=filters,
                )
                dense_success = True
            except Exception as exc:
                logger.warning("Dense search failed: %s; falling back to sparse branch", exc)
                dense_success = False

        sparse_candidates: list[tuple[str, float, dict[str, Any]]] = []
        sparse_success = False
        if sparse_enabled:
            if self.sparse_index is not None and self.sparse_index.is_usable():
                if not self.sparse_index.is_healthy():
                    logger.info(
                        "Sparse index status is '%s'; querying previously verified index data",
                        self.sparse_index.get_meta("status"),
                    )
                try:
                    sparse_candidates = self.sparse_index.search(
                        query,
                        top_k=candidate_k,
                        filters=filters,
                    )
                    sparse_success = True
                except Exception as exc:
                    logger.warning("Sparse search failed: %s", exc)
                    sparse_success = False
            else:
                status = self.sparse_index.get_meta("status") if self.sparse_index else "missing"
                logger.warning(
                    "Sparse index is unavailable or not in a usable state (status: %s)",
                    status,
                )
                sparse_success = False

        # Weight normalization handling branch failures
        # Surviving branch weight is normalized only among branches that were originally enabled
        if not dense_success and not sparse_success:
            return []

        if dense_success and sparse_success:
            total_w = dense_weight + sparse_weight
            dense_w = dense_weight / total_w
            sparse_w = sparse_weight / total_w
        elif dense_success:
            # Only dense succeeded (either sparse was disabled or sparse failed)
            dense_w = 1.0
            sparse_w = 0.0
        else:
            # Only sparse succeeded (either dense was disabled or dense failed)
            dense_w = 0.0
            sparse_w = 1.0


        # Map candidates by chunk_id
        dense_dict: dict[str, dict[str, Any]] = {
            chunk["chunk_id"]: chunk for chunk in dense_candidates if "chunk_id" in chunk
        }
        sparse_dict: dict[str, dict[str, Any]] = {
            cid: chunk for cid, _, chunk in sparse_candidates
        }

        # Identify sparse candidates missing from dense results for selective hydration
        sparse_cids_to_hydrate = [
            cid for cid in sparse_dict if cid not in dense_dict
        ]

        hydrated_from_qdrant: dict[str, dict[str, Any]] = {}
        if sparse_cids_to_hydrate:
            try:
                qdrant_chunks = self.store.get_chunks_by_ids(sparse_cids_to_hydrate)
                for c in qdrant_chunks:
                    if "chunk_id" in c:
                        hydrated_from_qdrant[c["chunk_id"]] = c
            except Exception as exc:
                logger.debug("Selective Qdrant point hydration omitted: %s", exc)

        fused_scores: dict[str, float] = {}
        semantic_scores: dict[str, float] = {}
        bm25_scores: dict[str, float] = {}
        final_payloads: dict[str, dict[str, Any]] = {}

        # 1. Process dense candidates
        # Pre-filter to candidates with a valid chunk_id and strictly positive semantic score
        # before assigning ranks, so that zero/negative-score candidates do not consume
        # rank positions and reduce the RRF contribution of valid candidates.
        if dense_w > 0:
            positive_dense = [
                chunk for chunk in dense_candidates
                if chunk.get("chunk_id") and float(chunk.get("semantic_score", 0.0)) > 0.0
            ]
            for rank, chunk in enumerate(positive_dense, start=1):
                cid = chunk["chunk_id"]
                sc = float(chunk["semantic_score"])
                final_payloads[cid] = chunk
                semantic_scores[cid] = sc
                fused_scores[cid] = fused_scores.get(cid, 0.0) + (dense_w / (rrf_k + rank))


        # 2. Process sparse candidates
        # Filter to strictly positive BM25 scores before assigning ranks, so that
        # zero/negative-score candidates do not consume rank positions in RRF.
        if sparse_w > 0:
            positive_sparse = [
                (cid, score, payload)
                for cid, score, payload in sparse_candidates
                if score > 0.0
            ]
            for rank, (cid, score, chunk_payload) in enumerate(positive_sparse, start=1):
                if cid not in final_payloads:
                    # Prefer hydrated Qdrant payload if available, else SQLite full payload
                    final_payloads[cid] = hydrated_from_qdrant.get(cid, chunk_payload)
                bm25_scores[cid] = float(score)
                fused_scores[cid] = fused_scores.get(cid, 0.0) + (sparse_w / (rrf_k + rank))

        # Return only matched candidates with positive fused scores
        matched_cids = [cid for cid, score in fused_scores.items() if score > 0.0]

        # Deterministic ranking: fused_score desc, semantic_score desc, bm25_score desc, chunk_id asc
        matched_cids.sort(
            key=lambda cid: (
                -fused_scores[cid],
                -semantic_scores.get(cid, 0.0),
                -bm25_scores.get(cid, 0.0),
                cid,
            )
        )

        results: list[dict[str, Any]] = []
        for cid in matched_cids[:top_k]:
            result = dict(final_payloads[cid])
            result["semantic_score"] = semantic_scores.get(cid, 0.0)
            result["bm25_score"] = bm25_scores.get(cid, 0.0)
            result["hybrid_score"] = fused_scores[cid]
            results.append(result)
        return results
