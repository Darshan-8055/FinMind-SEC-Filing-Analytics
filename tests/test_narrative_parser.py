"""
tests/test_narrative_parser.py
===============================
Pytest test suite for src.ingestion.narrative_parser (Phase 1).

Design goals
------------
* All tests use synthetic in-memory text – no real SEC filings, no network
  calls, no large model downloads.
* Tiktoken and Unstructured are mocked where necessary so the suite runs even
  when only one (or neither) of those packages is installed.
* Tests are grouped by concern so failures pinpoint the exact behaviour that
  broke.
"""

from __future__ import annotations

import sys
import types
import importlib
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers to import the module under test
# ---------------------------------------------------------------------------

MODULE_PATH = "src.ingestion.narrative_parser"


def _reload_parser():
    """Force-reload the module so module-level import flags are re-evaluated."""
    if MODULE_PATH in sys.modules:
        del sys.modules[MODULE_PATH]
    return importlib.import_module(MODULE_PATH)


# ---------------------------------------------------------------------------
# Synthetic filing fixtures
# ---------------------------------------------------------------------------

SIMPLE_FILING = """\
ANNUAL REPORT ON FORM 10-K

ITEM 1. BUSINESS

Acme Corp is a diversified technology company founded in 2001.
We operate three business segments: Cloud Services, Hardware,
and Professional Services. Our flagship product is the Acme Cloud Platform,
which serves over 50,000 enterprise customers worldwide.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

The following discussion should be read in conjunction with our consolidated
financial statements. Total revenues increased 18% year-over-year to $4.2B.
Operating income improved by 310 basis points reflecting continued operational
leverage and disciplined cost management across all segments.
"""

MULTI_SECTION_FILING = """\
ITEM 1. BUSINESS

Globex International is a leading provider of industrial automation solutions.
The company was incorporated in Delaware in 1995 and operates in 42 countries.
Our core products include robotic assembly systems and smart sensor arrays.

ITEM 1A. RISK FACTORS

Investing in our common stock involves a high degree of risk. The following
factors could adversely affect our business, operating results, and financial
condition. Competition in the automation sector is intense. Rapid technological
change may render our products obsolete. Our international operations expose us
to currency fluctuation and geopolitical risk.

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

We discuss herein the results of operations for the fiscal year ended December
31 2023 compared to the fiscal year ended December 31 2022. Revenues grew by
24 percent driven by strong demand in Asia-Pacific markets. Gross margin
expanded to 48 percent from 44 percent in the prior year.

ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK

Market risk is the risk of loss arising from adverse changes in market rates
and prices. We are exposed to interest rate risk and foreign currency exchange
rate risk in the normal course of our business activities.
"""

LONG_TEXT = " ".join(["word"] * 3000)  # ~3 000 words → several 512-token chunks

MIXED_HEADING_FILING = """\
PART II  ITEM 5. MARKET FOR REGISTRANT'S COMMON EQUITY

Our common stock is listed on NASDAQ under the ticker symbol ACME. As of the
record date there were 12,450 holders of record.

ITEM 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA

The financial statements required by this item are incorporated herein by
reference from pages F-1 through F-45 of this Annual Report.
"""

HEADING_ONLY_FILING = """\
ITEM 1. BUSINESS

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS
"""


# ---------------------------------------------------------------------------
# 1. Basic smoke tests – module imports and default config
# ---------------------------------------------------------------------------

class TestModuleImport:
    """The module must import cleanly and expose the public API."""

    def test_parse_filing_is_callable(self):
        from src.ingestion import narrative_parser as np_mod
        assert callable(np_mod.parse_filing)

    def test_parse_config_defaults(self):
        from src.ingestion.narrative_parser import ParseConfig, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
        cfg = ParseConfig()
        assert cfg.chunk_size == DEFAULT_CHUNK_SIZE == 512
        assert cfg.chunk_overlap == DEFAULT_CHUNK_OVERLAP == 50


# ---------------------------------------------------------------------------
# 2. Empty / invalid input handling
# ---------------------------------------------------------------------------

class TestEmptyAndInvalidInput:
    """parse_filing must handle edge cases gracefully."""

    def test_empty_string_returns_empty_list(self):
        from src.ingestion.narrative_parser import parse_filing
        result = parse_filing("", "empty.txt", "10-K")
        assert result == []

    def test_whitespace_only_returns_empty_list(self):
        from src.ingestion.narrative_parser import parse_filing
        result = parse_filing("   \n\t\n  ", "blank.txt", "10-Q")
        assert result == []

    def test_non_string_raises_type_error(self):
        from src.ingestion.narrative_parser import parse_filing
        with pytest.raises(TypeError, match="text must be str"):
            parse_filing(None, "bad.txt", "10-K")

    def test_non_string_bytes_raises_type_error(self):
        from src.ingestion.narrative_parser import parse_filing
        with pytest.raises(TypeError):
            parse_filing(b"some bytes", "bad.txt", "10-K")

    def test_integer_input_raises_type_error(self):
        from src.ingestion.narrative_parser import parse_filing
        with pytest.raises(TypeError):
            parse_filing(42, "bad.txt", "10-Q")


# ---------------------------------------------------------------------------
# 3. Metadata correctness
# ---------------------------------------------------------------------------

class TestMetadata:
    """Every chunk record must carry exactly the right metadata fields."""

    def test_required_keys_present(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(SIMPLE_FILING, "apple-10k-2023.txt", "10-K")
        assert records, "Expected at least one chunk"
        required = {"text", "filename", "filing_type", "section_heading", "chunk_index"}
        for rec in records:
            assert required == set(rec.keys()), f"Missing or extra keys in {rec}"

    def test_filename_propagated(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(SIMPLE_FILING, "apple-10k-2023.txt", "10-K")
        assert all(r["filename"] == "apple-10k-2023.txt" for r in records)

    def test_filing_type_10k_propagated(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(SIMPLE_FILING, "test.txt", "10-K")
        assert all(r["filing_type"] == "10-K" for r in records)

    def test_filing_type_10q_propagated(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(SIMPLE_FILING, "test.txt", "10-Q")
        assert all(r["filing_type"] == "10-Q" for r in records)

    def test_chunk_index_sequential(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(MULTI_SECTION_FILING, "multi.txt", "10-K")
        assert records
        indices = [r["chunk_index"] for r in records]
        assert indices == list(range(len(records))), "chunk_index must be sequential from 0"

    def test_chunk_index_starts_at_zero(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(SIMPLE_FILING, "test.txt", "10-K")
        assert records[0]["chunk_index"] == 0

    def test_text_field_is_non_empty_string(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(SIMPLE_FILING, "test.txt", "10-K")
        for rec in records:
            assert isinstance(rec["text"], str)
            assert rec["text"].strip() != ""


# ---------------------------------------------------------------------------
# 4. Section heading detection & preservation
# ---------------------------------------------------------------------------

class TestSectionHeadings:
    """Headings must be detected and normalised correctly."""

    def test_item1_heading_detected(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(SIMPLE_FILING, "test.txt", "10-K")
        headings = {r["section_heading"] for r in records}
        # At least one chunk should carry a heading that references Item 1
        assert any("Item 1" in h or "Business" in h for h in headings), (
            f"Item 1 heading not found. Headings found: {headings}"
        )

    def test_item7_heading_detected(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(SIMPLE_FILING, "test.txt", "10-K")
        headings = {r["section_heading"] for r in records}
        assert any("Item 7" in h or "Management" in h for h in headings), (
            f"Item 7 heading not found. Headings found: {headings}"
        )

    def test_item1a_heading_detected(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(MULTI_SECTION_FILING, "test.txt", "10-K")
        headings = {r["section_heading"] for r in records}
        assert any("1A" in h or "Risk" in h for h in headings), (
            f"Item 1A heading not found. Headings found: {headings}"
        )

    def test_item7a_heading_detected(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(MULTI_SECTION_FILING, "test.txt", "10-K")
        headings = {r["section_heading"] for r in records}
        assert any("7A" in h or "Market Risk" in h for h in headings), (
            f"Item 7A heading not found. Headings found: {headings}"
        )

    def test_section_heading_is_string(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(SIMPLE_FILING, "test.txt", "10-K")
        for rec in records:
            assert isinstance(rec["section_heading"], str)

    def test_normalise_heading_item7(self):
        from src.ingestion.narrative_parser import _normalise_heading
        raw = "  ITEM 7.  MANAGEMENT'S DISCUSSION AND ANALYSIS  "
        result = _normalise_heading(raw)
        assert "7" in result
        assert "Management" in result or "Discussion" in result

    def test_normalise_heading_unknown_item(self):
        """An unrecognised item number should be returned verbatim (stripped)."""
        from src.ingestion.narrative_parser import _normalise_heading
        raw = "  ITEM 99. UNKNOWN SECTION  "
        result = _normalise_heading(raw)
        assert result.strip() == result     # no leading/trailing whitespace
        assert "99" in result

    def test_heading_only_filing_no_crash(self):
        """A filing with headings but no body text should not raise."""
        from src.ingestion.narrative_parser import parse_filing
        # A filing that has headings followed by nothing produces no body chunks
        result = parse_filing(HEADING_ONLY_FILING, "empty_body.txt", "10-K")
        # Either empty list or list with empty-text items – but must not raise
        assert isinstance(result, list)

    def test_mixed_case_heading(self):
        from src.ingestion.narrative_parser import parse_filing
        text = "item 7. Management's Discussion\n\nSome body text here."
        records = parse_filing(text, "test.txt", "10-K")
        headings = {r["section_heading"] for r in records}
        assert any("7" in h or "Management" in h for h in headings)

    def test_part_ii_item5_heading(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(MIXED_HEADING_FILING, "test.txt", "10-K")
        headings = {r["section_heading"] for r in records}
        assert any("5" in h or "Market" in h for h in headings), (
            f"Item 5 heading not found. Headings found: {headings}"
        )


# ---------------------------------------------------------------------------
# 5. Token-aware chunking behaviour
# ---------------------------------------------------------------------------

class TestChunking:
    """Chunks must respect size limits and overlap configuration."""

    def test_short_text_produces_one_chunk(self):
        from src.ingestion.narrative_parser import parse_filing
        short = "This is a very short sentence that fits in one chunk."
        records = parse_filing(short, "short.txt", "10-K")
        assert len(records) == 1

    def test_long_text_produces_multiple_chunks(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(LONG_TEXT, "long.txt", "10-K")
        assert len(records) > 1, "Long text should be split into multiple chunks"

    def test_custom_small_chunk_size(self):
        """With a tiny chunk size, even short text must produce multiple chunks."""
        from src.ingestion.narrative_parser import parse_filing, ParseConfig
        text = " ".join(["hello"] * 200)  # 200 tokens
        records = parse_filing(text, "tiny.txt", "10-K", config=ParseConfig(chunk_size=30, chunk_overlap=5))
        assert len(records) > 1

    def test_chunk_overlap_invalid_raises(self):
        from src.ingestion.narrative_parser import _chunk_text
        with pytest.raises(ValueError, match="chunk_overlap"):
            _chunk_text("some text here", None, chunk_size=10, chunk_overlap=10)

    def test_chunk_overlap_must_be_less_than_chunk_size(self):
        from src.ingestion.narrative_parser import _chunk_text
        with pytest.raises(ValueError):
            _chunk_text("some text here", None, chunk_size=5, chunk_overlap=7)

    def test_empty_text_chunk_returns_empty_list(self):
        from src.ingestion.narrative_parser import _chunk_text
        assert _chunk_text("", None) == []

    def test_whitespace_only_chunk_returns_empty_list(self):
        from src.ingestion.narrative_parser import _chunk_text
        assert _chunk_text("   \n\t  ", None) == []

    def test_overlap_creates_shared_content(self):
        """With overlap > 0, consecutive chunks should share tail/head tokens."""
        from src.ingestion.narrative_parser import _chunk_text, _get_encoder
        # Use word-level fallback for deterministic test
        text = " ".join(["word"] * 200)
        encoder = _get_encoder()
        chunks = _chunk_text(text, encoder, chunk_size=50, chunk_overlap=10)
        assert len(chunks) >= 2
        # The tail words of chunk[0] should appear at the start of chunk[1]
        tail_words = set(chunks[0].split()[-5:])
        head_words = set(chunks[1].split()[:5])
        overlap = tail_words & head_words
        assert len(overlap) > 0, (
            f"Expected overlap between consecutive chunks.\n"
            f"Tail of chunk[0]: {chunks[0].split()[-5:]}\n"
            f"Head of chunk[1]: {chunks[1].split()[:5]}"
        )

    def test_all_chunks_non_empty(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(MULTI_SECTION_FILING, "test.txt", "10-K")
        for rec in records:
            assert rec["text"].strip(), "Each chunk text must be non-empty"


# ---------------------------------------------------------------------------
# 6. Heading regex unit tests
# ---------------------------------------------------------------------------

class TestHeadingRegex:
    """_HEADING_RE must match valid headings and ignore non-headings."""

    @pytest.mark.parametrize("heading", [
        "ITEM 1. BUSINESS",
        "Item 1. Business",
        "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS",
        "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
        "Item 1A. Risk Factors",
        "PART II  ITEM 5. MARKET FOR REGISTRANT'S COMMON EQUITY",
        "item 9a. controls and procedures",
        "ITEM 15. EXHIBITS",
    ])
    def test_heading_matched(self, heading):
        from src.ingestion.narrative_parser import _HEADING_RE
        assert _HEADING_RE.search(heading), f"Expected match for: {heading!r}"

    @pytest.mark.parametrize("non_heading", [
        "The company was founded in 2001.",
        "Total revenues increased 18%.",
        "See Note 7 for further details.",
        "F-7",
        "",
        "    ",
    ])
    def test_non_heading_not_matched(self, non_heading):
        from src.ingestion.narrative_parser import _HEADING_RE
        assert not _HEADING_RE.match(non_heading), (
            f"Expected no match for: {non_heading!r}"
        )


# ---------------------------------------------------------------------------
# 7. Section extraction logic
# ---------------------------------------------------------------------------

class TestSectionExtraction:
    """_extract_headings_and_sections must split text into correct pairs."""

    def test_two_sections_extracted(self):
        from src.ingestion.narrative_parser import _extract_headings_and_sections
        text = (
            "ITEM 1. BUSINESS\n\nAcme Corp is great.\n\n"
            "ITEM 7. MD&A\n\nRevenues grew 20%."
        )
        sections = _extract_headings_and_sections(text)
        # Expect exactly 2 sections (one per item with body text)
        assert len(sections) == 2

    def test_heading_content_preserved(self):
        from src.ingestion.narrative_parser import _extract_headings_and_sections
        text = "ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS\n\nRevenues grew."
        sections = _extract_headings_and_sections(text)
        assert sections
        headings = [s[0] for s in sections]
        assert any("7" in h or "Management" in h for h in headings)

    def test_body_text_preserved(self):
        from src.ingestion.narrative_parser import _extract_headings_and_sections
        body = "Acme Corp operates in 42 countries."
        text = f"ITEM 1. BUSINESS\n\n{body}"
        sections = _extract_headings_and_sections(text)
        combined_body = " ".join(s[1] for s in sections)
        assert body in combined_body

    def test_text_before_any_heading(self):
        """Text that appears before the first Item heading uses an empty heading."""
        from src.ingestion.narrative_parser import _extract_headings_and_sections
        text = "Preamble text here.\n\nITEM 1. BUSINESS\n\nBusiness body."
        sections = _extract_headings_and_sections(text)
        first_headings = [s[0] for s in sections if "Preamble" in s[1]]
        assert any(h == "" for h in first_headings)

    def test_no_headings_returns_one_section(self):
        from src.ingestion.narrative_parser import _extract_headings_and_sections
        text = "This filing has no structured headings whatsoever."
        sections = _extract_headings_and_sections(text)
        assert len(sections) == 1
        assert sections[0][0] == ""  # empty heading
        assert "no structured headings" in sections[0][1]


# ---------------------------------------------------------------------------
# 8. Unstructured mock – ensure graceful fallback on import error
# ---------------------------------------------------------------------------

class TestUnstructuredFallback:
    """When unstructured raises, the regex fallback must be used."""

    def test_partition_failure_falls_back_to_regex(self):
        from src.ingestion import narrative_parser as np_mod

        original_avail = np_mod._UNSTRUCTURED_AVAILABLE

        with patch.object(np_mod, "_UNSTRUCTURED_AVAILABLE", True):
            with patch.object(
                np_mod,
                "partition_text",
                side_effect=RuntimeError("mock unstructured failure"),
                create=True,
            ):
                # _elements_to_sections should catch the error and use regex fallback
                sections = np_mod._elements_to_sections(SIMPLE_FILING)
        assert isinstance(sections, list)
        # The regex fallback must have produced results
        assert len(sections) > 0

    def test_parse_filing_survives_total_element_extraction_failure(self):
        """Even if _elements_to_sections raises entirely, parse_filing must not crash."""
        from src.ingestion import narrative_parser as np_mod

        with patch.object(np_mod, "_elements_to_sections", side_effect=Exception("boom")):
            # parse_filing has a try/except that catches this
            records = np_mod.parse_filing(SIMPLE_FILING, "crash.txt", "10-K")
        # Should still return a list (possibly from last-resort fallback)
        assert isinstance(records, list)
        assert len(records) > 0


# ---------------------------------------------------------------------------
# 9. Token counter / encoder helpers
# ---------------------------------------------------------------------------

class TestTokenHelpers:
    """Low-level token helpers must work in isolation."""

    def test_count_tokens_non_zero_for_nonempty(self):
        from src.ingestion.narrative_parser import _count_tokens
        count = _count_tokens("Hello world", None)
        assert count > 0

    def test_count_tokens_zero_or_one_for_single_word(self):
        from src.ingestion.narrative_parser import _count_tokens
        count = _count_tokens("Hello", None)
        assert count >= 1

    def test_count_tokens_with_encoder(self):
        """If tiktoken is available, _count_tokens uses it."""
        from src.ingestion.narrative_parser import _count_tokens, _get_encoder
        encoder = _get_encoder()
        if encoder is None:
            pytest.skip("tiktoken not installed")
        count = _count_tokens("Hello world", encoder)
        assert count >= 1

    def test_encode_decode_roundtrip_with_encoder(self):
        from src.ingestion.narrative_parser import _encode, _decode, _get_encoder
        encoder = _get_encoder()
        if encoder is None:
            pytest.skip("tiktoken not installed")
        text = "SEC filing narrative text for roundtrip testing."
        ids = _encode(text, encoder)
        recovered = _decode(ids, text, encoder)
        assert recovered == text

    def test_encode_fallback_returns_index_list(self):
        from src.ingestion.narrative_parser import _encode
        ids = _encode("one two three", None)
        assert ids == [0, 1, 2]

    def test_decode_fallback_returns_words(self):
        from src.ingestion.narrative_parser import _decode
        text = "one two three"
        ids = [0, 1, 2]
        result = _decode(ids, text, None)
        assert result == "one two three"


# ---------------------------------------------------------------------------
# 10. Integration-style tests on realistic synthetic filings
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end tests using realistic multi-section filing text."""

    def test_multi_section_produces_chunks_for_each_section(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(MULTI_SECTION_FILING, "globex-10k.txt", "10-K")
        headings = {r["section_heading"] for r in records}
        # We expect at least Item 1, Item 7 to appear
        assert any("1" in h for h in headings)
        assert any("7" in h for h in headings)

    def test_all_sections_have_corresponding_chunks(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(MULTI_SECTION_FILING, "test.txt", "10-K")
        # Four items → at least 4 chunks (one per section)
        assert len(records) >= 4

    def test_filing_type_custom_string(self):
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(SIMPLE_FILING, "test.txt", "8-K")
        assert all(r["filing_type"] == "8-K" for r in records)

    def test_10q_filing_processed(self):
        ten_q = """\
ITEM 1. FINANCIAL STATEMENTS

The unaudited interim financial statements are presented herein.

ITEM 2. MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION

The following discussion is for the quarter ended September 30 2023.
Revenues for Q3 2023 were $1.1B, up 12% versus Q3 2022.
"""
        from src.ingestion.narrative_parser import parse_filing
        records = parse_filing(ten_q, "q3-2023-10q.txt", "10-Q")
        assert records
        assert all(r["filing_type"] == "10-Q" for r in records)
        headings = {r["section_heading"] for r in records}
        assert any("1" in h or "Financial" in h for h in headings)

    def test_chunk_text_coverage(self):
        """All meaningful words in the source appear somewhere in the chunks."""
        from src.ingestion.narrative_parser import parse_filing
        key_phrases = ["Acme Corp", "Cloud Platform", "revenues", "18%"]
        records = parse_filing(SIMPLE_FILING, "test.txt", "10-K")
        all_text = " ".join(r["text"] for r in records)
        for phrase in key_phrases:
            assert phrase in all_text, f"Key phrase {phrase!r} missing from chunks"

    def test_parse_config_custom_values_respected(self):
        from src.ingestion.narrative_parser import parse_filing, ParseConfig
        text = " ".join(["filler"] * 1000)
        default_recs = parse_filing(text, "t.txt", "10-K", config=ParseConfig(512, 50))
        small_recs = parse_filing(text, "t.txt", "10-K", config=ParseConfig(50, 5))
        # Smaller chunks → more records
        assert len(small_recs) > len(default_recs)
