"""
narrative_parser.py
===================
Phase-1 ingestion module for the FinMind project.

Responsibilities
----------------
* Parse SEC 10-K / 10-Q narrative text (plain-text or HTML) via Unstructured.
* Detect and preserve section headings (Item 1, Item 7, MD&A, etc.).
* Split the cleaned text into 512-token chunks with a 50-token overlap using a
  tiktoken-backed, tokenizer-aware splitter – no blind character counting.
* Attach structured metadata to every chunk so the future hybrid-retrieval
  module can filter, rank, and cite results without re-parsing the source.

Public API
----------
    parse_filing(text, filename, filing_type) -> list[dict]

Each returned dict (a "chunk record") has this shape::

    {
        "text":            str,   # the actual chunk text
        "filename":        str,   # e.g. "apple-10k-2023.txt"
        "filing_type":     str,   # "10-K" | "10-Q"
        "section_heading": str,   # detected heading or empty string
        "chunk_index":     int,   # 0-based position within this filing
    }
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: tiktoken
# ---------------------------------------------------------------------------
try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover – tested via mock in unit tests
    _TIKTOKEN_AVAILABLE = False
    logger.warning(
        "tiktoken is not installed. "
        "Falling back to a simple whitespace-based token counter. "
        "Install tiktoken for accurate token counts: pip install tiktoken"
    )

# ---------------------------------------------------------------------------
# Optional dependency: Unstructured
# ---------------------------------------------------------------------------
try:
    from unstructured.partition.text import partition_text
    from unstructured.documents.elements import Title, NarrativeText, Text

    _UNSTRUCTURED_AVAILABLE = True
except ImportError:  # pragma: no cover
    _UNSTRUCTURED_AVAILABLE = False
    logger.warning(
        "unstructured is not installed. "
        "Falling back to a regex-based text splitter. "
        "Install it with: pip install 'unstructured[local-inference]'"
    )

# ---------------------------------------------------------------------------
# Constants / configuration
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE: int = 512    # tokens per chunk
DEFAULT_CHUNK_OVERLAP: int = 50  # token overlap between consecutive chunks

# Tiktoken encoding to use.  "cl100k_base" is the GPT-4 / text-embedding-3
# tokeniser and is a sensible default for SEC filing text.
_TIKTOKEN_ENCODING: str = "cl100k_base"

# Regex that identifies SEC section headings in plain text.
# Matches patterns like:
#   ITEM 1. BUSINESS
#   Item 1 – Business
#   Item 1A.  Risk Factors
#   PART II  ITEM 8  Financial Statements
_HEADING_RE = re.compile(
    r"^\s*(?:PART\s+[IVX]+\s+)?ITEM\s+\d+[A-Z]?\b.*$",
    re.IGNORECASE | re.MULTILINE,
)

# Heading labels we especially care about (for normalisation / display).
_NOTABLE_ITEMS: dict[str, str] = {
    "1":   "Item 1 - Business",
    "1a":  "Item 1A - Risk Factors",
    "2":   "Item 2 - Properties",
    "3":   "Item 3 - Legal Proceedings",
    "5":   "Item 5 - Market for Registrant's Common Equity",
    "6":   "Item 6 - Selected Financial Data",
    "7":   "Item 7 - Management's Discussion and Analysis",
    "7a":  "Item 7A - Quantitative and Qualitative Disclosures about Market Risk",
    "8":   "Item 8 - Financial Statements",
    "9":   "Item 9 - Changes in and Disagreements with Accountants",
    "9a":  "Item 9A - Controls and Procedures",
    "13":  "Item 13 - Certain Relationships and Related Transactions",
    "14":  "Item 14 - Principal Accounting Fees and Services",
    "15":  "Item 15 - Exhibits",
}


# ---------------------------------------------------------------------------
# Tokeniser helpers
# ---------------------------------------------------------------------------

def _get_encoder():
    """Return a tiktoken Encoding object, or None if not available."""
    if not _TIKTOKEN_AVAILABLE:
        return None
    try:
        return tiktoken.get_encoding(_TIKTOKEN_ENCODING)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not load tiktoken encoding %r: %s", _TIKTOKEN_ENCODING, exc)
        return None


def _count_tokens(text: str, encoder) -> int:
    """Count tokens in *text* using *encoder*, or approximate via words."""
    if encoder is not None:
        return len(encoder.encode(text))
    # Fallback: a word is ~1.3 tokens on average; round up for safety.
    return max(1, int(len(text.split()) * 1.3))


def _encode(text: str, encoder) -> list[int]:
    """Return token IDs for *text*."""
    if encoder is not None:
        return encoder.encode(text)
    # Fallback: encode each word as its own fake "token id" (index).
    return list(range(len(text.split())))


def _decode(token_ids: list[int], text: str, encoder) -> str:
    """Decode *token_ids* back to a string."""
    if encoder is not None:
        return encoder.decode(token_ids)
    # Fallback: recover words from the original text by index.
    words = text.split()
    return " ".join(words[i] for i in token_ids if i < len(words))


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

def _normalise_heading(raw: str) -> str:
    """
    Turn a raw heading match (e.g. "  ITEM 7A.  Quantitative …") into a
    canonical display string from *_NOTABLE_ITEMS*, or return the stripped
    raw text if not recognised.
    """
    raw = raw.strip()
    # Extract the item number/letter, e.g. "7A" from "Item 7A. …"
    m = re.search(r"ITEM\s+(\d+[A-Z]?)", raw, re.IGNORECASE)
    if m:
        key = m.group(1).lower()
        if key in _NOTABLE_ITEMS:
            return _NOTABLE_ITEMS[key]
    return raw


def _extract_headings_and_sections(text: str) -> list[tuple[str, str]]:
    """
    Split *text* into (heading, body) pairs using _HEADING_RE.

    The first pair will have an empty heading if the text begins before any
    section marker.
    """
    sections: list[tuple[str, str]] = []
    last_end = 0
    current_heading = ""

    for match in _HEADING_RE.finditer(text):
        body = text[last_end: match.start()]
        if body.strip():
            sections.append((current_heading, body))
        current_heading = _normalise_heading(match.group(0))
        last_end = match.end()

    # Remainder after the last heading.
    tail = text[last_end:]
    if tail.strip():
        sections.append((current_heading, tail))

    return sections


# ---------------------------------------------------------------------------
# Token-aware chunking
# ---------------------------------------------------------------------------

def _chunk_text(
    text: str,
    encoder,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split *text* into overlapping token windows.

    Parameters
    ----------
    text:
        The input string to split.
    encoder:
        A tiktoken Encoding (or None for the word-index fallback).
    chunk_size:
        Maximum tokens per chunk.
    chunk_overlap:
        Number of tokens from the end of one chunk that are repeated at the
        start of the next.

    Returns
    -------
    list[str]
        Decoded chunk strings.  An empty input returns [].
    """
    if not text or not text.strip():
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})."
        )

    token_ids = _encode(text, encoder)
    if not token_ids:
        return []

    chunks: list[str] = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))
        window = token_ids[start:end]
        chunk_str = _decode(window, text, encoder)
        if chunk_str.strip():
            chunks.append(chunk_str)
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Unstructured-backed element extraction
# ---------------------------------------------------------------------------

def _elements_to_sections(raw_text: str) -> list[tuple[str, str]]:
    """
    Use Unstructured to extract elements, then group them into
    (heading, body_text) pairs.

    Falls back to the regex splitter when Unstructured is unavailable or
    raises an exception.
    """
    if not _UNSTRUCTURED_AVAILABLE:
        return _extract_headings_and_sections(raw_text)

    try:
        elements = partition_text(text=raw_text)
    except Exception as exc:
        logger.warning("Unstructured partition_text failed (%s). Using regex fallback.", exc)
        return _extract_headings_and_sections(raw_text)

    sections: list[tuple[str, str]] = []
    current_heading = ""
    buffer: list[str] = []

    for el in elements:
        el_text = str(el).strip()
        if not el_text:
            continue

        # Unstructured marks headings with the Title element type.
        is_title = isinstance(el, Title)

        # Also catch SEC-style headings that regex finds but Unstructured
        # might classify as NarrativeText if the file has no explicit markup.
        is_heading_re = bool(_HEADING_RE.match(el_text))

        if is_title or is_heading_re:
            # Flush the current buffer as a body block.
            if buffer:
                sections.append((current_heading, "\n\n".join(buffer)))
                buffer = []
            current_heading = _normalise_heading(el_text)
        else:
            buffer.append(el_text)

    # Flush tail.
    if buffer:
        sections.append((current_heading, "\n\n".join(buffer)))

    # Edge case: no sections built (e.g. only headings, no body).
    if not sections and current_heading:
        sections.append((current_heading, ""))

    return sections


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

@dataclass
class ParseConfig:
    """Chunking configuration passed to :func:`parse_filing`."""
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP


def parse_filing(
    text: str,
    filename: str,
    filing_type: str,
    config: ParseConfig | None = None,
) -> list[dict]:
    """
    Parse a raw SEC filing text into a list of annotated chunk records.

    Parameters
    ----------
    text:
        Raw narrative text of the filing (plain-text or lightly-marked HTML).
    filename:
        Source filename, e.g. ``"apple-10k-2023.txt"``.
    filing_type:
        One of ``"10-K"`` or ``"10-Q"`` (or any string – it is stored
        verbatim as metadata).
    config:
        Optional :class:`ParseConfig`; defaults to 512-token chunks with
        50-token overlap.

    Returns
    -------
    list[dict]
        A (possibly empty) list of chunk dicts with keys:
        ``text``, ``filename``, ``filing_type``, ``section_heading``,
        ``chunk_index``.

    Raises
    ------
    TypeError
        If *text* is not a ``str``.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__!r}")

    if config is None:
        config = ParseConfig()

    if not text or not text.strip():
        logger.info("parse_filing: empty text received for %r – returning [].", filename)
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
        # Last-resort fallback: treat whole text as one anonymous section.
        sections = [("", text)]

    records: list[dict] = []
    chunk_index = 0

    for heading, body in sections:
        if not body.strip():
            continue

        chunks = _chunk_text(body, encoder, config.chunk_size, config.chunk_overlap)

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
        "parse_filing: %r → %d chunks across %d sections.",
        filename,
        len(records),
        len(sections),
    )
    return records
