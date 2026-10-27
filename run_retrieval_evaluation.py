"""Run the FinMind retrieval evaluation for keyword, semantic, and hybrid search.

This script evaluates all three retrieval methods against every query in the
fixture file and prints per-query and macro-average Precision@K and Recall@K.

Usage
-----
    python run_retrieval_evaluation.py

Requirements
------------
- The sentence-transformers model ``all-MiniLM-L6-v2`` must be available
  locally (downloaded automatically the first time).
- No live Qdrant service is required; evaluation runs in memory.

Notes
-----
Results are produced from the synthetic fixture in
``tests/fixtures/retrieval_eval.json``.  They reflect a small, hand-crafted
chunk set and should not be reported as production retrieval quality figures.
"""

import json
from pathlib import Path

from src.retrieval.evaluation import evaluate_all_queries
from src.retrieval.semantic_search import load_embedding_model

FIXTURE_PATH = Path("tests/fixtures/retrieval_eval.json")
K = 3

with FIXTURE_PATH.open("r", encoding="utf-8") as fh:
    dataset = json.load(fh)

print("Loading embedding model...")
model = load_embedding_model()

print(f"\nRunning retrieval evaluation at k={K} across "
      f"{len(dataset['queries'])} queries...\n")

output = evaluate_all_queries(
    queries=dataset["queries"],
    chunks=dataset["chunks"],
    model=model,
    k=K,
)

METHODS = ("keyword", "semantic", "hybrid")
LABEL_WIDTH = 10

# Per-query results
for entry in output["per_query"]:
    print(f"Query: {entry['query']!r}")
    for method in METHODS:
        m = entry[method]
        print(
            f"  {method:<{LABEL_WIDTH}}"
            f"  Precision@{K}: {m['precision_at_k']:.3f}"
            f"  Recall@{K}: {m['recall_at_k']:.3f}"
            f"  Retrieved: {m['retrieved_chunk_ids']}"
        )
    print()

# Macro-average summary
print("--- Macro-average ---")
for method in METHODS:
    macro = output["macro_average"][method]
    print(
        f"  {method:<{LABEL_WIDTH}}"
        f"  Precision@{K}: {macro['precision_at_k']:.3f}"
        f"  Recall@{K}: {macro['recall_at_k']:.3f}"
    )

print()
print("NOTE: Results are from synthetic fixture data. "
      "Do not report as production figures.")
