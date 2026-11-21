"""Benchmark runner for FinMind Phase 2 Step 3.

Executes Dense-only, Sparse-only, and Hybrid retrieval over the 593-chunk
real SEC benchmark dataset across 54 evaluation queries (42 answerable, 12 unanswerable).
Produces isolated Qdrant and SQLite stores and saves JSON result reports.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

# Ensure offline model execution
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from src.ingestion.pipeline import FilingIngestionPipeline
from src.retrieval.evaluation import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
    mean_reciprocal_rank_at_k,
)
from src.retrieval.qdrant_store import QdrantChunkStore
from src.retrieval.semantic_search import load_embedding_model
from src.retrieval.sparse_index import SparseKeywordIndex

logger = logging.getLogger(__name__)

BENCHMARK_DIR = Path(__file__).resolve().parent / "benchmarks" / "sec_filings"
CORPUS_PATH = BENCHMARK_DIR / "corpus_chunks.json"
QUERIES_PATH = BENCHMARK_DIR / "evaluation_queries.json"
RESULTS_DIR = BENCHMARK_DIR / "results"
EVAL_SPARSE_DB = BENCHMARK_DIR / "eval_sparse_index.db"

BENCHMARK_COLLECTION_NAME = "benchmark_eval_chunks"
TOP_K = 5
CANDIDATE_K = 30
RRF_K = 60


def run_benchmark(
    *,
    corpus_path: Path = CORPUS_PATH,
    queries_path: Path = QUERIES_PATH,
    results_dir: Path = RESULTS_DIR,
    sparse_db_path: Path = EVAL_SPARSE_DB,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Execute the full benchmark run across Dense, Sparse, and Hybrid pipelines."""
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load benchmark dataset
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    answerable_queries = [q for q in queries if q["answerable"]]
    unanswerable_queries = [q for q in queries if not q["answerable"]]

    # 2. Initialize production embedding model offline
    model = load_embedding_model()
    dim = model.get_sentence_embedding_dimension()

    # 3. Setup strictly isolated stores
    # Remove previous eval sparse DB if exists to guarantee clean rebuild
    if sparse_db_path.exists():
        sparse_db_path.unlink()

    qdrant_client = QdrantClient(":memory:")
    chunk_store = QdrantChunkStore(
        client=qdrant_client,
        collection_name=BENCHMARK_COLLECTION_NAME,
        vector_size=dim,
    )
    sparse_index = SparseKeywordIndex(db_path=sparse_db_path)

    pipeline = FilingIngestionPipeline(
        store=chunk_store,
        model=model,
        sparse_index=sparse_index,
    )

    # 4. Prepare and index benchmark chunks
    # Note: QdrantChunkStore requires a non-empty string for 'section'.
    # Chunks before the first SEC Item heading have an empty section string from the parser;
    # following the production pattern in AnswerGenerator, we populate missing section with 'Unknown section'.
    chunks_to_index = []
    for c in corpus:
        c_copy = dict(c)
        if not c_copy.get("section") or not str(c_copy["section"]).strip():
            c_copy["section"] = "Unknown section"
        chunks_to_index.append(c_copy)

    chunk_texts = [c["text"] for c in chunks_to_index]
    embeddings = model.encode(chunk_texts, batch_size=64, show_progress_bar=False)

    upserted_qdrant = chunk_store.upsert_chunks(chunks_to_index, embeddings)
    upserted_sparse = sparse_index.add_chunks(chunks_to_index)

    if upserted_qdrant != len(corpus) or upserted_sparse != len(corpus):
        raise RuntimeError(
            f"Indexing mismatch: corpus has {len(corpus)} chunks, but Qdrant indexed "
            f"{upserted_qdrant} and SQLite indexed {upserted_sparse}"
        )

    # 5. Execute queries for each retrieval configuration
    configurations = {
        "dense": {"dense_weight": 1.0, "sparse_weight": 0.0},
        "sparse": {"dense_weight": 0.0, "sparse_weight": 1.0},
        "hybrid": {"dense_weight": 0.7, "sparse_weight": 0.3},
    }

    timestamp = datetime.now(timezone.utc).isoformat()
    benchmark_reports: dict[str, Any] = {}

    for mode, weights in configurations.items():
        mode_results = []
        ans_precisions = []
        ans_recalls = []
        ans_rrs = []

        unans_with_retrieval = 0

        for q in queries:
            qid = q["query_id"]
            qtext = q["query"]
            is_ans = q["answerable"]
            rel_ids = set(q.get("relevant_chunk_ids", []))

            # Execute pipeline hybrid_search with specified branch weights
            retrieved = pipeline.hybrid_search(
                qtext,
                top_k=top_k,
                candidate_k=CANDIDATE_K,
                dense_weight=weights["dense_weight"],
                sparse_weight=weights["sparse_weight"],
                rrf_k=RRF_K,
            )

            retrieved_ids = [
                str(r["chunk_id"]) for r in retrieved if r.get("chunk_id") is not None
            ]

            entry: dict[str, Any] = {
                "query_id": qid,
                "query": qtext,
                "answerable": is_ans,
                "retrieved_chunk_ids": retrieved_ids,
                "relevant_chunk_ids": sorted(rel_ids),
                "top_k": top_k,
            }

            if is_ans:
                p = precision_at_k(retrieved_ids, rel_ids, top_k)
                r = recall_at_k(retrieved_ids, rel_ids, top_k)
                rr = reciprocal_rank_at_k(retrieved_ids, rel_ids, top_k)
                entry["precision_at_k"] = p
                entry["recall_at_k"] = r
                entry["reciprocal_rank_at_k"] = rr
                ans_precisions.append(p)
                ans_recalls.append(r)
                ans_rrs.append(rr)
            else:
                # Unanswerable query evaluation: track whether any chunks were retrieved
                # Note: Ground-truth contains no valid answer in corpus.
                # We record retrieved chunk count and ids without claiming false precision/recall.
                entry["retrieved_chunks_count"] = len(retrieved_ids)
                entry["corpus_evidence_retrieved"] = len(retrieved_ids) > 0
                entry["evidence_abstention_supported"] = False  # Pipeline does not implement threshold abstention
                if len(retrieved_ids) > 0:
                    unans_with_retrieval += 1

            mode_results.append(entry)

        # Calculate macro averages over answerable queries
        macro_precision = sum(ans_precisions) / len(ans_precisions) if ans_precisions else 0.0
        macro_recall = sum(ans_recalls) / len(ans_recalls) if ans_recalls else 0.0
        macro_mrr = mean_reciprocal_rank_at_k(ans_rrs)

        report = {
            "mode": mode,
            "timestamp": timestamp,
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_dimension": dim,
            "qdrant_collection": BENCHMARK_COLLECTION_NAME,
            "sparse_db_path": str(sparse_db_path),
            "total_corpus_chunks": len(corpus),
            "total_queries": len(queries),
            "answerable_queries_count": len(answerable_queries),
            "unanswerable_queries_count": len(unanswerable_queries),
            "top_k": top_k,
            "candidate_k": CANDIDATE_K,
            "weights": weights,
            "rrf_k": RRF_K,
            "metrics": {
                "precision_at_k": macro_precision,
                "recall_at_k": macro_recall,
                "mrr_at_k": macro_mrr,
            },
            "unanswerable_summary": {
                "total_unanswerable": len(unanswerable_queries),
                "queries_with_retrieval": unans_with_retrieval,
                "abstention_tested": False,
            },
            "per_query_results": mode_results,
        }

        # Save to results file
        out_file = results_dir / f"{mode}_results.json"
        with open(out_file, "w", encoding="utf-8") as f_out:
            json.dump(report, f_out, indent=2)

        benchmark_reports[mode] = report

    return benchmark_reports


if __name__ == "__main__":
    print("Executing FinMind SEC Benchmark Runner (Step 3)...")
    reports = run_benchmark()
    print("\nBenchmark Execution Complete!")
    for mode, rep in reports.items():
        m = rep["metrics"]
        print(
            f"  {mode.upper():<8} -> "
            f"Precision@5: {m['precision_at_k']:.4f} | "
            f"Recall@5: {m['recall_at_k']:.4f} | "
            f"MRR@5: {m['mrr_at_k']:.4f}"
        )
