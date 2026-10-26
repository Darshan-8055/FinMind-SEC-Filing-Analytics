
"""
narrative_parser.py
===================
Phase-1 ingestion module for the FinMind project.

Responsibilities
----------------
* Parse SEC 10-K / 10-Q narrative text (plain-text or HTML) via Unstructured.
* Detect and preserve section headings (Item 1, Item 7, etc.).
* Split text into token-aware chunks with configurable overlap.
* Attach metadata to every chunk for retrieval and citation.

Public API
----------
    parse_filing(text, filename, filing_type) -> list[dict]

Each chunk record contains:
    text, filename, filing_type, section_heading, chunk_index
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: tiktoken
# ---------------------------------------------------------------------------
try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TIKTOKEN_AVAILABLE = False
    logger.warning(
        "tiktoken is not installed. Falling back to whitespace-based tokens."
    )

# ---------------------------------------------------------------------------
# Optional dependency: Unstructured
# ---------------------------------------------------------------------------
try:
    from unstructured.partition.text import partition_text

    _UNSTRUCTURED_AVAILABLE = True
except ImportError:  # pragma: no cover
    _UNSTRUCTURED_AVAILABLE = False
    logger.warning(
        "unstructured is not installed. Falling back to regex-based extraction."
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CHUNK_SIZE: int = 512
DEFAULT_CHUNK_OVERLAP: int = 50
_TIKTOKEN_ENCODING = "cl100k_base"

_HEADING_RE = re.compile(
    r"^\s*(?:PART\s+[IVX]+\s+)?ITEM\s+\d+[A-Z]?\b.*$",
    re.IGNORECASE | re.MULTILINE,
)

_NOTABLE_ITEMS: dict[str, str] = {
    "1": "Item 1 - Business",
    "1a": "Item 1A - Risk Factors",
    "2": "Item 2 - Properties",
    "3": "Item 3 - Legal Proceedings",
    "5": "Item 5 - Market for Registrant's Common Equity",
    "6": "Item 6 - Selected Financial Data",
    "7": "Item 7 - Management's Discussion and Analysis",
    "7a": "Item 7A - Quantitative and Qualitative Disclosures about Market Risk",
    "8": "Item 8 - Financial Statements",
    "9": "Item 9 - Changes in and Disagreements with Accountants",
    "9a": "Item 9A - Controls and Procedures",
    "13": "Item 13 - Certain Relationships and Related Transactions",
    "14": "Item 14 - Principal Accounting Fees and Services",
    "15": "Item 15 - Exhibits",
}


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------
def _get_encoder() -> Any:
    """Return the tiktoken encoding, or None if unavailable."""
    if not _TIKTOKEN_AVAILABLE:
        return None

    try:
        return tiktoken.get_encoding(_TIKTOKEN_ENCODING)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not load tokenizer %r: %s", _TIKTOKEN_ENCODING, exc)
        return None


def _encode(text: str, encoder: Any) -> list[int]:
    """Encode text to token IDs, with a word-based fallback."""
    if encoder is not None:
        return encoder.encode(text)
    return list(range(len(text.split())))


def _decode(token_ids: list[int], text: str, encoder: Any) -> str:
    """Decode token IDs, with a word-based fallback."""
    if encoder is not None:
        return encoder.decode(token_ids)

    words = text.split()
    return " ".join(words[i] for i in token_ids if i < len(words))


def _count_tokens(text: str, encoder: Any) -> int:
    """Count tokens using the configured encoder."""
    if encoder is not None:
        return len(encoder.encode(text))
    return max(1, int(len(text.split()) * 1.3))


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------
def _normalise_heading(raw: str) -> str:
    """Return a canonical label for a recognized SEC Item heading."""
    raw = raw.strip()
    match = re.search(r"ITEM\s+(\d+[A-Z]?)", raw, re.IGNORECASE)

    if match:
        key = match.group(1).lower()
        if key in _NOTABLE_ITEMS:
            return _NOTABLE_ITEMS[key]

    return raw


def _extract_headings_and_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (heading, body) pairs using SEC heading patterns."""
    sections: list[tuple[str, str]] = []
    last_end = 0
    current_heading = ""

    for match in _HEADING_RE.finditer(text):
        body = text[last_end:match.start()]

        if body.strip():
            sections.append((current_heading, body))

        current_heading = _normalise_heading(match.group(0))
        last_end = match.end()

    tail = text[last_end:]
    if tail.strip():
        sections.append((current_heading, tail))

    return sections


# ---------------------------------------------------------------------------
# Token-aware chunking
# ---------------------------------------------------------------------------
def _chunk_text(
    text: str,
    encoder: Any,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping token windows."""
    if not text or not text.strip():
        return []

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})."
        )

    token_ids = _encode(text, encoder)
    if not token_ids:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        chunk = _decode(token_ids[start:end], text, encoder)

        if chunk.strip():
            chunks.append(chunk)

        start += step

    return chunks


# ---------------------------------------------------------------------------
# Unstructured-backed element extraction
# ---------------------------------------------------------------------------
def _elements_to_sections(raw_text: str) -> list[tuple[str, str]]:
    """
    Extract text elements and group them into (heading, body) pairs.

    Only SEC-style ITEM/PART headings are treated as section headings.
    Unstructured can classify ordinary body sentences as Title elements,
    so element type alone must not determine whether text is a heading.
    """
    if not _UNSTRUCTURED_AVAILABLE:
        return _extract_headings_and_sections(raw_text)

    try:
        elements = partition_text(text=raw_text)
    except Exception as exc:
        logger.warning(
            "Unstructured partition_text failed (%s). Using regex fallback.",
            exc,
        )
        return _extract_headings_and_sections(raw_text)

    sections: list[tuple[str, str]] = []
    current_heading = ""
    buffer: list[str] = []

    for element in elements:
        element_text = str(element).strip()
        if not element_text:
            continue

        # Critical fix: don't classify every Unstructured Title as a heading.
        is_sec_heading = bool(_HEADING_RE.match(element_text))

        if is_sec_heading:
            if buffer:
                sections.append(
                    (current_heading, "\n\n".join(buffer))
                )
                buffer = []

            current_heading = _normalise_heading(element_text)
        else:
            # Preserve Title, NarrativeText, and other element text as content.
            buffer.append(element_text)

    if buffer:
        sections.append((current_heading, "\n\n".join(buffer)))

    if not sections and current_heading:
        sections.append((current_heading, ""))

    return sections


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------
@dataclass
class ParseConfig:
    """Chunking configuration passed to parse_filing."""
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP


def parse_filing(
    text: str,
    filename: str,
    filing_type: str,
    config: ParseConfig | None = None,
) -> list[dict]:
    """Parse filing text into chunk records with section metadata."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__!r}")

    if config is None:
        config = ParseConfig()

    if not text or not text.strip():
        logger.info(
            "parse_filing: empty text received for %r; returning [].",
            filename,
        )
        return []

    encoder = _get_encoder()

    try:
        sections = _elements_to_sections(text)
    except Exception as exc:
        logger.error(
            "parse_filing: unexpected error extracting sections from %r: %s",
            filename,
            exc,
            exc_info=True,
        )
        sections = [("", text)]

    records: list[dict] = []
    chunk_index = 0

    for heading, body in sections:
        if not body.strip():
            continue

        chunks = _chunk_text(
            body,
            encoder,
            config.chunk_size,
            config.chunk_overlap,
        )

        for chunk_text in chunks:
            if not chunk_text.strip():
                continue

            records.append(
                {
                    "text": chunk_text,
                    "filename": filename,
                    "filing_type": filing_type,
                    "section_heading": heading,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

    logger.info(
        "parse_filing: %r -> %d chunks across %d sections.",
        filename,
        len(records),
        len(sections),
    )
    return records
