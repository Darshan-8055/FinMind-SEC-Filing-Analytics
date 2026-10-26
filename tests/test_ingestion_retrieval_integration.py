
from src.ingestion.narrative_parser import parse_filing
from src.ingestion.chunk_adapter import adapt_chunk
from src.retrieval.keyword_search import keyword_search


def test_parser_adapter_keyword_search_pipeline():
    filing_text = """
ITEM 1. BUSINESS

Example Corporation develops cloud software for businesses.
Its subscription revenue increased during the fiscal year.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Revenue increased by 18 percent compared with the prior year.
Operating expenses also increased as the company invested in growth.
"""

    parsed_chunks = parse_filing(
        text=filing_text,
        filename="example-10k.txt",
        filing_type="10-K",
    )

    assert parsed_chunks
    assert all(chunk["text"].strip() for chunk in parsed_chunks)

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

    assert all(chunk["form_type"] == "10-K" for chunk in adapted_chunks)
    assert all(chunk["ticker"] == "EXM" for chunk in adapted_chunks)

    results = keyword_search(
        query="subscription revenue increased",
        chunks=adapted_chunks,
        top_k=3,
    )

    assert results
    assert "subscription revenue increased" in results[0]["text"].lower()
    assert results[0]["ticker"] == "EXM"
    assert results[0]["company_name"] == "Example Corporation"
    assert results[0]["form_type"] == "10-K"
    assert results[0]["source_url"] == "https://www.sec.gov/example"
