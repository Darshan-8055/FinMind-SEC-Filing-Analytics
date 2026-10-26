
"""Run the sample evaluation for FinMind keyword retrieval."""

import json
from pathlib import Path

from src.retrieval.evaluation import evaluate_query

fixture_path = Path("tests/fixtures/retrieval_eval.json")

with fixture_path.open("r", encoding="utf-8") as file:
    dataset = json.load(file)

results = []

for item in dataset["queries"]:
    result = evaluate_query(
        query=item["query"],
        chunks=dataset["chunks"],
        relevant_ids=item["relevant_chunk_ids"],
        k=3,
    )
    results.append(result)

for result in results:
    print(f"\nQuery: {result['query']}")
    print(f"Retrieved: {result['retrieved_chunk_ids']}")
    print(f"Relevant:  {result['relevant_chunk_ids']}")
    print(f"Precision@3: {result['precision_at_k']:.2f}")
    print(f"Recall@3:    {result['recall_at_k']:.2f}")

if results:
    avg_precision = sum(
        item["precision_at_k"] for item in results
    ) / len(results)
    avg_recall = sum(
        item["recall_at_k"] for item in results
    ) / len(results)

    print("\n--- Overall Results ---")
    print(f"Average Precision@3: {avg_precision:.2f}")
    print(f"Average Recall@3:    {avg_recall:.2f}")
