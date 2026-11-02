"""End-to-end integration tests for filing retrieval and answer generation."""

import numpy as np
from qdrant_client import QdrantClient

from src.agents.answer_generator import AnswerGenerator
from src.ingestion.pipeline import FilingIngestionPipeline
from src.retrieval.qdrant_store import QdrantChunkStore


class FakeEmbeddingModel:
    """Deterministic embedding model for integration tests."""

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = np.zeros(384, dtype=np.float32)
            vector[0] = 1.0
            vectors.append(vector)
        return np.array(vectors)


def test_filing_ingestion_to_cited_answer():
    """Test ingestion, Qdrant retrieval, and cited answer generation."""

    client = QdrantClient(":memory:")

    try:
        store = QdrantChunkStore(
            client,
            collection_name="answer_generator_integration",
            vector_size=384,
        )

        model = FakeEmbeddingModel()
        pipeline = FilingIngestionPipeline(store, model)

        filing_text = """
ITEM 1. BUSINESS

Example Corporation provides financial analytics software.
Its subscription revenue increased during the fiscal year.
The company serves enterprise customers.
"""

        count = pipeline.ingest_filing(
            filing_text,
            filename="example-10k.txt",
            filing_type="10-K",
            ticker="EXM",
            company_name="Example Corporation",
            filing_date="2026-02-15",
            source_url="https://www.sec.gov/example",
            period_end="2025-12-31",
        )

        assert count > 0

        generated_prompts = []

        def fake_generate(prompt):
            generated_prompts.append(prompt)
            return (
                "Example Corporation provides financial analytics "
                "software and serves enterprise customers. [S1]"
            )

        answer_generator = AnswerGenerator(
            retrieve=pipeline.search,
            generate=fake_generate,
        )

        result = answer_generator.answer(
            "What does Example Corporation do?",
            top_k=5,
        )

        # Answer generation must use the retrieved evidence.
        assert generated_prompts
        assert "Example Corporation" in generated_prompts[0]

        # Verify the generated answer.
        assert result["insufficient_evidence"] is False
        assert "financial analytics software" in result["answer"]

        # Verify trusted citation metadata.
        assert len(result["citations"]) == 1

        citation = result["citations"][0]

        assert citation["label"] == "S1"
        assert citation["ticker"] == "EXM"
        assert citation["company_name"] == "Example Corporation"
        assert citation["form_type"] == "10-K"
        assert citation["filing_date"] == "2026-02-15"
        assert citation["period_end"] == "2025-12-31"
        assert citation["source_url"] == "https://www.sec.gov/example"
        assert citation["chunk_id"]
        assert citation["section"]
    finally:
        client.close()