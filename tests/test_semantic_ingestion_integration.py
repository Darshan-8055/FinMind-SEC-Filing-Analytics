
from src.ingestion.narrative_parser import parse_filing
from src.ingestion.chunk_adapter import adapt_chunk
from src.retrieval.semantic_search import semantic_search


class FakeEmbeddingModel:
    """Provide deterministic embeddings without downloading a model."""

    def encode(self, texts):
        vectors = {
            "revenue increased by 18 percent during the fiscal year.": [
                1.0, 0.0
            ],
            "the company faces competition in its markets.": [
                0.0, 1.0
            ],
        }

        return [
            [1.0, 0.0] if text == "revenue growth" else vectors[text]
            for text in texts
        ]


def test_parser_adapter_semantic_search_pipeline():
    filing_text = """
ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Revenue increased by 18 percent during the fiscal year.

ITEM 1A. RISK FACTORS

The company faces competition in its markets.
"""

    parsed_chunks = parse_filing(
        text=filing_text,
        filename="example-10k.txt",
        filing_type="10-K",
    )

    assert parsed_chunks

    adapted_chunks = [
        adapt_chunk(
            chunk,
            chunk_id=f"example-{index}",
            ticker="EXM",
            company_name="Example Corporation",
            filing_date="2026-02-15",
            source_url="https://www.sec.gov/example",
        )
        for index, chunk in enumerate(parsed_chunks)
    ]

    # The fake model expects these exact texts. Check that parsing and
    # chunking produced the intended passages before embedding them.
    revenue_chunk = next(
        chunk for chunk in adapted_chunks
        if "Revenue increased by 18 percent during the fiscal year."
        in chunk["text"]
    )
    risk_chunk = next(
        chunk for chunk in adapted_chunks
        if "The company faces competition in its markets."
        in chunk["text"]
    )

    # Use a model that can handle the actual chunk text by assigning
    # vectors based on which filing passage each text contains.
    class FilingFakeEmbeddingModel:
        def encode(self, texts):
            result = []
            for text in texts:
                if text == "revenue growth":
                    result.append([1.0, 0.0])
                elif "Revenue increased by 18 percent" in text:
                    result.append([0.99, 0.01])
                elif "company faces competition" in text:
                    result.append([0.0, 1.0])
                else:
                    result.append([0.5, 0.5])
            return result

    results = semantic_search(
        query="revenue growth",
        chunks=adapted_chunks,
        model=FilingFakeEmbeddingModel(),
        top_k=3,
    )

    assert results
    assert results[0]["chunk_id"] == revenue_chunk["chunk_id"]
    assert results[0]["ticker"] == "EXM"
    assert results[0]["company_name"] == "Example Corporation"
    assert results[0]["form_type"] == "10-K"
    assert results[0]["source_url"] == "https://www.sec.gov/example"
    assert "semantic_score" in results[0]
