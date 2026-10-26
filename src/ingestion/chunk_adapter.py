
"""Map narrative parser output to the shared FinMind chunk interface."""

from collections.abc import Mapping
from typing import Any


def adapt_chunk(
    parsed_chunk: Mapping[str, Any],
    *,
    chunk_id: str,
    ticker: str,
    company_name: str,
    filing_date: str,
    source_url: str,
    period_end: str | None = None,
) -> dict[str, Any]:
    """Convert one parser chunk into the shared interface schema.

    Required filing metadata must be supplied by the caller. This function
    does not invent company, filing, or source details.
    """
    if not isinstance(parsed_chunk, Mapping):
        raise TypeError("parsed_chunk must be a mapping")

    required_parser_fields = ("text", "filing_type", "section_heading")
    for field in required_parser_fields:
        if field not in parsed_chunk:
            raise ValueError(f"parsed_chunk is missing required field: {field}")

    for field, value in (
        ("chunk_id", chunk_id),
        ("ticker", ticker),
        ("company_name", company_name),
        ("filing_date", filing_date),
        ("source_url", source_url),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")

    text = parsed_chunk["text"]
    filing_type = parsed_chunk["filing_type"]
    section_heading = parsed_chunk["section_heading"]

    if not isinstance(text, str) or not text.strip():
        raise ValueError("parsed_chunk text must be a non-empty string")
    if not isinstance(filing_type, str) or not filing_type.strip():
        raise ValueError("filing_type must be a non-empty string")
    if not isinstance(section_heading, str):
        raise ValueError("section_heading must be a string")
    if period_end is not None and not isinstance(period_end, str):
        raise TypeError("period_end must be a string or None")

    adapted = {
        "chunk_id": chunk_id,
        "ticker": ticker,
        "company_name": company_name,
        "form_type": filing_type,
        "filing_date": filing_date,
        "period_end": period_end,
        "section": section_heading,
        "text": text,
        "source_url": source_url,
    }

    # Keep useful parser traceability without changing the shared fields.
    if "filename" in parsed_chunk:
        adapted["filename"] = parsed_chunk["filename"]
    if "chunk_index" in parsed_chunk:
        adapted["chunk_index"] = parsed_chunk["chunk_index"]

    return adapted
