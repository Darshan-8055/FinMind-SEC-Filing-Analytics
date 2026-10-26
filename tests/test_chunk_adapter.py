
import pytest

from src.ingestion.chunk_adapter import adapt_chunk


@pytest.fixture
def parsed_chunk():
    return {
        "text": "Revenue increased during the fiscal year.",
        "filename": "example-10k.txt",
        "filing_type": "10-K",
        "section_heading": "Item 7 - Management's Discussion and Analysis",
        "chunk_index": 0,
    }


def test_maps_parser_fields_to_shared_schema(parsed_chunk):
    result = adapt_chunk(
        parsed_chunk,
        chunk_id="chunk-001",
        ticker="EXM",
        company_name="Example Corporation",
        filing_date="2026-02-15",
        period_end="2025-12-31",
        source_url="https://www.sec.gov/example",
    )

    assert result["form_type"] == "10-K"
    assert result["section"] == parsed_chunk["section_heading"]
    assert result["text"] == parsed_chunk["text"]
    assert result["chunk_id"] == "chunk-001"
    assert result["ticker"] == "EXM"
    assert result["company_name"] == "Example Corporation"
    assert result["filing_date"] == "2026-02-15"
    assert result["period_end"] == "2025-12-31"
    assert result["source_url"] == "https://www.sec.gov/example"


def test_preserves_parser_traceability_fields(parsed_chunk):
    result = adapt_chunk(
        parsed_chunk,
        chunk_id="chunk-001",
        ticker="EXM",
        company_name="Example Corporation",
        filing_date="2026-02-15",
        source_url="https://www.sec.gov/example",
    )

    assert result["filename"] == "example-10k.txt"
    assert result["chunk_index"] == 0
    assert result["period_end"] is None


def test_does_not_mutate_original_chunk(parsed_chunk):
    original = dict(parsed_chunk)

    adapt_chunk(
        parsed_chunk,
        chunk_id="chunk-001",
        ticker="EXM",
        company_name="Example Corporation",
        filing_date="2026-02-15",
        source_url="https://www.sec.gov/example",
    )

    assert parsed_chunk == original


def test_rejects_missing_parser_field(parsed_chunk):
    del parsed_chunk["text"]

    with pytest.raises(ValueError, match="missing required field"):
        adapt_chunk(
            parsed_chunk,
            chunk_id="chunk-001",
            ticker="EXM",
            company_name="Example Corporation",
            filing_date="2026-02-15",
            source_url="https://www.sec.gov/example",
        )


def test_rejects_empty_chunk_id(parsed_chunk):
    with pytest.raises(ValueError, match="chunk_id"):
        adapt_chunk(
            parsed_chunk,
            chunk_id="",
            ticker="EXM",
            company_name="Example Corporation",
            filing_date="2026-02-15",
            source_url="https://www.sec.gov/example",
        )
