
"""Evaluate semantic retrieval on the existing sample dataset."""

import json
from pathlib import Path
from statistics import mean

from src.retrieval.evaluation import precision_at_k, recall_at_k
from src.retrieval.semantic_search import load_embedding_model, semantic_search


def main():
    fixture_path = Path("tests/fixtures/retrieval_eval.json")

    with fixture_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    chunks = dataset["chunks"]
    queries = dataset["queries"]

    print("Loading embedding model...")
    model = load_embedding_model()
    print("Model loaded successfully.\n")

    precision_scores = []
    recall_scores = []

    for case in queries:
        query = case["query"]
        relevant_ids = set(case["relevant_chunk_ids"])

        results = semantic_search(
            query=query,
            chunks=chunks,
            model=model,
            top_k=3,
        )

        retrieved_ids = [
            str(result.get("chunk_id", result.get("id", "")))
            for result in results
        ]

        precision = precision_at_k(retrieved_ids, relevant_ids, k=3)
        recall = recall_at_k(retrieved_ids, relevant_ids, k=3)

        precision_scores.append(precision)
        recall_scores.append(recall)

        print(f"Query: {query}")
        print(f"Retrieved: {retrieved_ids}")
        print(f"Relevant:  {sorted(relevant_ids)}")
        print(f"Precision@3: {precision:.2f}")
        print(f"Recall@3:    {recall:.2f}")
        print("-" * 60)

    print("\nSEMANTIC RETRIEVAL SUMMARY")
    print(f"Queries evaluated: {len(queries)}")
    print(f"Average Precision@3: {mean(precision_scores):.2f}")
    print(f"Average Recall@3:    {mean(recall_scores):.2f}")


if __name__ == "__main__":
    main()
