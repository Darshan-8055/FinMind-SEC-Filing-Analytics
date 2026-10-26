
from qdrant_client import QdrantClient
from src.retrieval.qdrant_store import QdrantChunkStore

COLLECTION = "finmind_chunks_test"

client = QdrantClient(url="http://localhost:6333")

store = QdrantChunkStore(
    client,
    collection_name=COLLECTION,
    vector_size=3,
)

chunks = [
    {
        "chunk_id": "test-chunk-001",
        "company_name": "Example Corporation",
        "ticker": "EXM",
        "form_type": "10-K",
        "filing_date": "2026-02-01",
        "section": "Item 1 - Business",
        "text": "Example Corporation develops financial analytics software.",
        "source_url": "https://example.com/filing/001",
    },
    {
        "chunk_id": "test-chunk-002",
        "company_name": "Sample Industries",
        "ticker": "SMP",
        "form_type": "10-Q",
        "filing_date": "2026-04-15",
        "section": "Item 1A - Risk Factors",
        "text": "Sample Industries identifies supply chain risks.",
        "source_url": "https://example.com/filing/002",
    },
]

embeddings = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
]

try:
    # This collection is dedicated to this test.
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    count = store.upsert_chunks(chunks, embeddings)
    print("Chunks inserted:", count)

    results = store.search([0.95, 0.05, 0.0], top_k=2)

    print("Search results:", len(results))
    for result in results:
        print(
            result["ticker"],
            "| score:", round(result["semantic_score"], 4),
            "| section:", result["section"],
        )

    assert count == 2, "Expected two chunks to be inserted"
    assert results, "Expected at least one search result"
    assert results[0]["ticker"] == "EXM", "Expected EXM to rank first"
    assert results[0]["text"] == chunks[0]["text"], "Chunk text was not preserved"

    print("QDRANT INTEGRATION TEST PASSED")

finally:
    # Clean up only the dedicated test collection.
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.close()
