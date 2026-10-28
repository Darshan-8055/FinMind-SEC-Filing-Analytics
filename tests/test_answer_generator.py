"""Tests for src/agents/answer_generator.py.

All tests use injected fake callables. No network requests are made.
"""

import pytest

from src.agents.answer_generator import (
    AnswerGenerator,
    ClaimVerifier,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    _acknowledges_insufficient_evidence,
    _cited_labels,
    _sanitise_answer,
    _validate_verifier_result,
    build_evidence_prompt,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_CHUNK = {
    "chunk_id": "AAPL:2025-10-31:aapl-10k.htm:0",
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "form_type": "10-K",
    "filing_date": "2025-10-31",
    "period_end": "2025-09-27",
    "section": "Risk Factors",
    "text": "The company faces risks related to competition.",
    "source_url": "https://www.sec.gov/Archives/example",
}

SECOND_CHUNK = {
    "chunk_id": "AAPL:2025-10-31:aapl-10k.htm:1",
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "form_type": "10-K",
    "filing_date": "2025-10-31",
    "period_end": "2025-09-27",
    "section": "Financial Results",
    "text": "Revenue increased significantly in fiscal 2025.",
    "source_url": "https://www.sec.gov/Archives/example2",
}

# Ten chunks used to test exact label matching (S1 vs S10).
TEN_CHUNKS = [
    {
        "chunk_id": f"C{i}",
        "ticker": "X",
        "filing_date": "2025-01-01",
        "section": "S",
        "text": f"Passage {i}.",
        "source_url": f"https://example.com/{i}",
    }
    for i in range(1, 11)
]


# ---------------------------------------------------------------------------
# build_evidence_prompt — structure and validation
# ---------------------------------------------------------------------------

class TestBuildEvidencePrompt:

    def test_prompt_includes_question_and_evidence(self):
        prompt, sources = build_evidence_prompt(
            "What risks does the company face?",
            [SAMPLE_CHUNK],
        )
        assert "What risks does the company face?" in prompt
        assert SAMPLE_CHUNK["text"] in prompt
        assert "[S1]" in prompt
        assert sources["S1"]["chunk_id"] == SAMPLE_CHUNK["chunk_id"]

    def test_source_registry_contains_required_fields(self):
        _, sources = build_evidence_prompt("Test?", [SAMPLE_CHUNK])
        s1 = sources["S1"]
        assert s1["chunk_id"] == SAMPLE_CHUNK["chunk_id"]
        assert s1["source_url"] == SAMPLE_CHUNK["source_url"]
        assert s1["section"] == SAMPLE_CHUNK["section"]
        assert s1["ticker"] == SAMPLE_CHUNK["ticker"]
        assert s1["filing_date"] == SAMPLE_CHUNK["filing_date"]

    def test_multiple_chunks_get_sequential_labels(self):
        _, sources = build_evidence_prompt("Q?", [SAMPLE_CHUNK, SECOND_CHUNK])
        assert set(sources.keys()) == {"S1", "S2"}
        assert sources["S1"]["chunk_id"] == SAMPLE_CHUNK["chunk_id"]
        assert sources["S2"]["chunk_id"] == SECOND_CHUNK["chunk_id"]

    def test_passage_text_wrapped_in_delimiters(self):
        """Passage text must be isolated from the instruction block."""
        prompt, _ = build_evidence_prompt("Q?", [SAMPLE_CHUNK])
        assert "<<<" in prompt
        assert ">>>" in prompt
        # The passage text itself must be between the delimiters.
        assert f"<<<\n{SAMPLE_CHUNK['text']}\n>>>" in prompt

    def test_prompt_instructs_lm_not_to_invent_labels(self):
        prompt, _ = build_evidence_prompt("Q?", [SAMPLE_CHUNK])
        assert "Do NOT create citation labels" in prompt or \
               "not create citation labels" in prompt.lower() or \
               "Do not create citations" in prompt

    def test_prompt_instructs_lm_to_handle_conflicts(self):
        prompt, _ = build_evidence_prompt("Q?", [SAMPLE_CHUNK, SECOND_CHUNK])
        lower = prompt.lower()
        assert "conflict" in lower

    def test_prompt_rejects_chunk_without_source_url(self):
        chunk = {**SAMPLE_CHUNK, "source_url": ""}
        with pytest.raises(ValueError, match="source_url"):
            build_evidence_prompt("What are the risks?", [chunk])

    def test_prompt_rejects_chunk_without_chunk_id(self):
        chunk = {**SAMPLE_CHUNK, "chunk_id": ""}
        with pytest.raises(ValueError, match="chunk_id"):
            build_evidence_prompt("Q?", [chunk])

    def test_prompt_rejects_chunk_without_text(self):
        chunk = {**SAMPLE_CHUNK, "text": "   "}
        with pytest.raises(ValueError, match="text"):
            build_evidence_prompt("Q?", [chunk])

    def test_prompt_rejects_empty_chunk_list(self):
        with pytest.raises(ValueError):
            build_evidence_prompt("Q?", [])

    def test_prompt_rejects_empty_question(self):
        with pytest.raises(ValueError):
            build_evidence_prompt("   ", [SAMPLE_CHUNK])

    def test_missing_optional_fields_use_defaults(self):
        minimal_chunk = {
            "chunk_id": "C1",
            "text": "Some text.",
            "source_url": "https://example.com",
        }
        prompt, sources = build_evidence_prompt("Q?", [minimal_chunk])
        assert sources["S1"]["section"] == "Unknown section"
        assert sources["S1"]["ticker"] == "Unknown"
        assert sources["S1"]["filing_date"] == "Unknown"


# ---------------------------------------------------------------------------
# _cited_labels — exact label matching
# ---------------------------------------------------------------------------

class TestCitedLabels:

    def _sources(self, n: int) -> dict:
        return {f"S{i}": {} for i in range(1, n + 1)}

    def test_single_citation_matched(self):
        assert _cited_labels("See [S1] for details.", self._sources(3)) == ["S1"]

    def test_s1_not_matched_inside_s10(self):
        """[S1] must not fire on text containing only [S10]."""
        labels = _cited_labels("See [S10] for details.", self._sources(10))
        assert "S1" not in labels

    def test_s1_not_matched_inside_s11(self):
        labels = _cited_labels("See [S11] for details.", self._sources(11))
        assert "S1" not in labels

    def test_s1_matched_when_followed_by_non_digit(self):
        labels = _cited_labels("See [S1] and [S10].", self._sources(10))
        assert "S1" in labels
        assert "S10" in labels

    def test_s1_matched_at_end_of_string(self):
        labels = _cited_labels("Risk is present [S1]", self._sources(3))
        assert labels == ["S1"]

    def test_multiple_citations_matched_in_order(self):
        sources = self._sources(3)
        labels = _cited_labels("[S3] outperforms [S1].", sources)
        assert set(labels) == {"S1", "S3"}

    def test_unknown_label_not_in_sources_ignored(self):
        labels = _cited_labels("[S99] is not a real label.", self._sources(3))
        assert "S99" not in labels

    def test_duplicate_occurrences_deduplicated(self):
        sources = self._sources(3)
        labels = _cited_labels("[S1] supports claim, and [S1] again.", sources)
        assert labels.count("S1") == 1

    def test_empty_generated_text_returns_empty(self):
        assert _cited_labels("", self._sources(3)) == []


# ---------------------------------------------------------------------------
# _acknowledges_insufficient_evidence — no false positives
# ---------------------------------------------------------------------------

class TestAcknowledgesInsufficientEvidence:

    def test_detects_explicit_phrase(self):
        assert _acknowledges_insufficient_evidence(
            "The available evidence is insufficient to answer this question."
        )

    def test_detects_cannot_answer(self):
        assert _acknowledges_insufficient_evidence(
            "I cannot answer this question based on the provided passages."
        )

    def test_detects_passages_do_not(self):
        assert _acknowledges_insufficient_evidence(
            "The passages do not contain the requested information."
        )

    def test_case_insensitive(self):
        assert _acknowledges_insufficient_evidence(
            "INSUFFICIENT EVIDENCE was found in the data."
        )

    def test_realistic_sufficient_answer_not_flagged(self):
        answer = (
            "According to [S1], the company reported strong revenue growth "
            "of 12% year-over-year in fiscal 2025. Competition risk is noted "
            "in [S2], which states that new market entrants could erode margins."
        )
        assert not _acknowledges_insufficient_evidence(answer)

    def test_answer_mentioning_cannot_in_different_context_not_flagged(self):
        """'cannot' in a factual statement must not trigger the flag."""
        answer = (
            "The company reported that it cannot guarantee future dividends [S1], "
            "but cash flow from operations increased 15% [S2]."
        )
        assert not _acknowledges_insufficient_evidence(answer)

    def test_answer_with_not_in_different_context_not_flagged(self):
        answer = (
            "Revenue did not decline during fiscal 2025, according to [S1]. "
            "The filing notes competition is not expected to diminish [S2]."
        )
        assert not _acknowledges_insufficient_evidence(answer)


# ---------------------------------------------------------------------------
# AnswerGenerator.answer — full integration scenarios
# ---------------------------------------------------------------------------

class TestAnswerGeneratorAnswer:

    def _make_generator(self, chunks, response, verifier=None):
        """Helper: build an AnswerGenerator with injected retrieve/generate."""
        def retrieve(question, *, top_k, filters):
            return chunks

        def generate(prompt):
            return response

        return AnswerGenerator(retrieve, generate, verifier=verifier)

    # --- existing valid citation behavior ---

    def test_answer_contains_only_known_citations(self):
        gen = self._make_generator(
            [SAMPLE_CHUNK],
            "Competition is a risk disclosed in the filing [S1]. [S99]",
        )
        result = gen.answer("What are the risks?")

        assert result["insufficient_evidence"] is False
        assert len(result["citations"]) == 1
        assert result["citations"][0]["label"] == "S1"
        assert result["citations"][0]["chunk_id"] == SAMPLE_CHUNK["chunk_id"]
        assert result["citations"][0]["source_url"] == SAMPLE_CHUNK["source_url"]

    def test_result_keys_present(self):
        gen = self._make_generator([SAMPLE_CHUNK], "Answer [S1].")
        result = gen.answer("Q?")
        assert {"question", "answer", "citations", "insufficient_evidence"} \
               <= result.keys()

    def test_question_preserved_in_result(self):
        gen = self._make_generator([SAMPLE_CHUNK], "Answer [S1].")
        result = gen.answer("What are the risks?")
        assert result["question"] == "What are the risks?"

    # --- no retrieved evidence ---

    def test_no_retrieved_chunks_returns_insufficient_evidence(self):
        def retrieve(question, *, top_k, filters):
            return []

        def generate(prompt):
            raise AssertionError("LLM must not be called without evidence")

        gen = AnswerGenerator(retrieve, generate)
        result = gen.answer("What was the company's revenue?")

        assert result["answer"] == INSUFFICIENT_EVIDENCE_MESSAGE
        assert result["citations"] == []
        assert result["insufficient_evidence"] is True

    # --- empty / malformed generated output ---

    def test_empty_string_from_llm_returns_insufficient_evidence(self):
        gen = self._make_generator([SAMPLE_CHUNK], "")
        result = gen.answer("Q?")
        assert result["insufficient_evidence"] is True
        assert result["citations"] == []

    def test_whitespace_only_from_llm_returns_insufficient_evidence(self):
        gen = self._make_generator([SAMPLE_CHUNK], "   \n  ")
        result = gen.answer("Q?")
        assert result["insufficient_evidence"] is True

    def test_none_from_llm_returns_insufficient_evidence(self):
        """None is not a valid string; must not raise."""
        gen = self._make_generator([SAMPLE_CHUNK], None)  # type: ignore[arg-type]
        result = gen.answer("Q?")
        assert result["insufficient_evidence"] is True

    # --- explicit insufficient-evidence responses ---

    def test_llm_acknowledges_insufficient_evidence(self):
        gen = self._make_generator(
            [SAMPLE_CHUNK],
            "I cannot answer this question based on the provided passages.",
        )
        result = gen.answer("Q?")
        assert result["insufficient_evidence"] is True
        assert result["citations"] == []

    def test_llm_says_passages_do_not_support(self):
        gen = self._make_generator(
            [SAMPLE_CHUNK],
            "The passages do not contain the requested information.",
        )
        result = gen.answer("Q?")
        assert result["insufficient_evidence"] is True

    def test_realistic_answer_with_citations_not_flagged_as_insufficient(self):
        answer = (
            "According to [S1], the company reported a 12% increase in revenue. "
            "Competition risk is highlighted in [S2]."
        )
        gen = self._make_generator([SAMPLE_CHUNK, SECOND_CHUNK], answer)
        result = gen.answer("What was revenue growth?")
        assert result["insufficient_evidence"] is False
        assert len(result["citations"]) == 2

    # --- answers without recognised citations ---

    def test_answer_with_no_citations_flagged_as_insufficient(self):
        gen = self._make_generator(
            [SAMPLE_CHUNK],
            "The company faces many risks in its industry.",
        )
        result = gen.answer("Q?")
        assert result["insufficient_evidence"] is True
        assert result["citations"] == []

    # --- exact label matching: [S1] vs [S10] ---

    def test_s1_not_confused_with_s10(self):
        """Retrieving 10 chunks: [S10] must not match S1."""
        gen = self._make_generator(
            TEN_CHUNKS,
            "Passage ten is relevant [S10].",
        )
        result = gen.answer("Q?")
        labels = [c["label"] for c in result["citations"]]
        assert "S10" in labels
        assert "S1" not in labels

    def test_s1_matched_separately_from_s10(self):
        gen = self._make_generator(
            TEN_CHUNKS,
            "Both [S1] and [S10] are relevant.",
        )
        result = gen.answer("Q?")
        labels = {c["label"] for c in result["citations"]}
        assert "S1" in labels
        assert "S10" in labels

    # --- duplicate citation handling ---

    def test_duplicate_citation_labels_deduplicated(self):
        gen = self._make_generator(
            [SAMPLE_CHUNK],
            "[S1] supports the claim and [S1] is noted again.",
        )
        result = gen.answer("Q?")
        assert len(result["citations"]) == 1

    # --- multiple valid citations ---

    def test_multiple_valid_citations(self):
        gen = self._make_generator(
            [SAMPLE_CHUNK, SECOND_CHUNK],
            "Risk is described in [S1]. Revenue is in [S2].",
        )
        result = gen.answer("Q?")
        labels = {c["label"] for c in result["citations"]}
        assert labels == {"S1", "S2"}
        assert result["insufficient_evidence"] is False

    def test_citation_metadata_from_registry_not_llm(self):
        """Citation metadata must come from the trusted registry."""
        gen = self._make_generator(
            [SAMPLE_CHUNK],
            "See [S1] for risks.",
        )
        result = gen.answer("Q?")
        c = result["citations"][0]
        assert c["chunk_id"] == SAMPLE_CHUNK["chunk_id"]
        assert c["source_url"] == SAMPLE_CHUNK["source_url"]
        assert c["section"] == SAMPLE_CHUNK["section"]
        assert c["ticker"] == SAMPLE_CHUNK["ticker"]
        assert c["filing_date"] == SAMPLE_CHUNK["filing_date"]

    # --- invalid chunk metadata ---

    def test_chunk_missing_chunk_id_raises(self):
        bad_chunk = {**SAMPLE_CHUNK, "chunk_id": ""}

        def retrieve(question, *, top_k, filters):
            return [bad_chunk]

        def generate(prompt):
            return "[S1] answer"

        gen = AnswerGenerator(retrieve, generate)
        with pytest.raises(ValueError, match="chunk_id"):
            gen.answer("Q?")

    def test_chunk_missing_source_url_raises(self):
        bad_chunk = {**SAMPLE_CHUNK, "source_url": None}

        def retrieve(question, *, top_k, filters):
            return [bad_chunk]

        gen = AnswerGenerator(retrieve, lambda p: "[S1] answer")
        with pytest.raises(ValueError, match="source_url"):
            gen.answer("Q?")

    def test_chunk_missing_text_raises(self):
        bad_chunk = {**SAMPLE_CHUNK, "text": ""}

        def retrieve(question, *, top_k, filters):
            return [bad_chunk]

        gen = AnswerGenerator(retrieve, lambda p: "[S1] answer")
        with pytest.raises(ValueError, match="text"):
            gen.answer("Q?")

    # --- conflicting evidence instruction ---

    def test_prompt_contains_conflict_instruction(self):
        """The prompt must tell the LLM to explain conflicting passages."""
        observed_prompt = {}

        def retrieve(q, *, top_k, filters):
            return [SAMPLE_CHUNK, SECOND_CHUNK]

        def generate(prompt):
            observed_prompt["value"] = prompt
            return "[S1] and [S2] conflict."

        gen = AnswerGenerator(retrieve, generate)
        gen.answer("Q?")
        assert "conflict" in observed_prompt["value"].lower()

    # --- top_k and filter passthrough ---

    def test_retrieval_receives_top_k_and_filters(self):
        observed = {}

        def retrieve(question, *, top_k, filters):
            observed["question"] = question
            observed["top_k"] = top_k
            observed["filters"] = filters
            return [SAMPLE_CHUNK]

        gen = AnswerGenerator(
            retrieve,
            lambda prompt: "The passage discusses competition [S1].",
        )

        gen.answer(
            "What risks are discussed?",
            top_k=3,
            filters={"ticker": "AAPL"},
        )

        assert observed["top_k"] == 3
        assert observed["filters"] == {"ticker": "AAPL"}
        assert observed["question"] == "What risks are discussed?"

    # --- argument validation ---

    def test_rejects_empty_question(self):
        gen = self._make_generator([SAMPLE_CHUNK], "[S1] answer")
        with pytest.raises(ValueError):
            gen.answer("   ")

    def test_rejects_invalid_top_k(self):
        gen = self._make_generator([SAMPLE_CHUNK], "[S1] answer")
        with pytest.raises(ValueError):
            gen.answer("Q?", top_k=0)


# ---------------------------------------------------------------------------
# ClaimVerifier — dependency injection and safe fallback
# ---------------------------------------------------------------------------

class TestClaimVerifier:

    def _chunk(self):
        return [SAMPLE_CHUNK]

    def test_verifier_success(self):
        def verify_fn(question, answer, chunks):
            return {
                "verified": True,
                "unsupported_claims": [],
                "conflicts": [],
            }

        verifier = ClaimVerifier(verify_fn)
        result = verifier.verify("Q?", "Answer [S1].", self._chunk())
        assert result["verified"] is True
        assert result["verification_available"] is True

    def test_verifier_finds_unsupported_claim(self):
        def verify_fn(question, answer, chunks):
            return {
                "verified": False,
                "unsupported_claims": ["Revenue grew 50%"],
                "conflicts": [],
            }

        verifier = ClaimVerifier(verify_fn)
        result = verifier.verify("Q?", "Revenue grew 50% [S1].", self._chunk())
        assert result["verified"] is False
        assert "Revenue grew 50%" in result["unsupported_claims"]

    def test_verifier_not_available_returns_safe_fallback(self):
        def verify_fn(question, answer, chunks):
            raise NotImplementedError

        verifier = ClaimVerifier(verify_fn)
        result = verifier.verify("Q?", "Answer.", self._chunk())
        assert result["verified"] is False
        assert result["verification_available"] is False
        assert "error" in result

    def test_verifier_unexpected_error_returns_safe_fallback(self):
        def verify_fn(question, answer, chunks):
            raise RuntimeError("Model crashed")

        verifier = ClaimVerifier(verify_fn)
        result = verifier.verify("Q?", "Answer.", self._chunk())
        assert result["verified"] is False
        assert result["verification_available"] is False
        assert "Model crashed" in result["error"]

    def test_verifier_result_never_silently_verified(self):
        """A fallback result must never claim verified=True."""
        def verify_fn(question, answer, chunks):
            raise NotImplementedError

        verifier = ClaimVerifier(verify_fn)
        result = verifier.verify("Q?", "Answer.", self._chunk())
        assert result["verified"] is not True

    def test_verifier_absent_key_defaults_added(self):
        """verify_fn may omit optional keys; ClaimVerifier fills them."""
        def verify_fn(question, answer, chunks):
            return {"verified": True}

        verifier = ClaimVerifier(verify_fn)
        result = verifier.verify("Q?", "Answer.", self._chunk())
        assert "unsupported_claims" in result
        assert "conflicts" in result
        assert result["verification_available"] is True


# ---------------------------------------------------------------------------
# AnswerGenerator with ClaimVerifier injected
# ---------------------------------------------------------------------------

class TestAnswerGeneratorWithVerifier:

    def test_verification_included_in_result_when_verifier_supplied(self):
        def verify_fn(q, a, chunks):
            return {"verified": True, "unsupported_claims": [], "conflicts": []}

        gen = AnswerGenerator(
            retrieve=lambda q, *, top_k, filters: [SAMPLE_CHUNK],
            generate=lambda p: "Risks noted in filing [S1].",
            verifier=ClaimVerifier(verify_fn),
        )
        result = gen.answer("Q?")
        assert "verification" in result
        assert result["verification"]["verified"] is True

    def test_verification_key_absent_when_no_verifier(self):
        gen = AnswerGenerator(
            retrieve=lambda q, *, top_k, filters: [SAMPLE_CHUNK],
            generate=lambda p: "Risks noted in filing [S1].",
        )
        result = gen.answer("Q?")
        assert "verification" not in result

    def test_verifier_failure_does_not_crash_answer(self):
        def verify_fn(q, a, chunks):
            raise RuntimeError("Verifier unavailable")

        gen = AnswerGenerator(
            retrieve=lambda q, *, top_k, filters: [SAMPLE_CHUNK],
            generate=lambda p: "Risks noted in filing [S1].",
            verifier=ClaimVerifier(verify_fn),
        )
        result = gen.answer("Q?")
        # Answer still returned; verification contains fallback
        assert result["answer"] == "Risks noted in filing [S1]."
        assert result["verification"]["verified"] is False

    def test_insufficient_evidence_result_has_no_verification_run(self):
        """Verifier should not run when there is no evidence to verify."""
        calls = []

        def verify_fn(q, a, chunks):
            calls.append(1)
            return {"verified": True, "unsupported_claims": [], "conflicts": []}

        gen = AnswerGenerator(
            retrieve=lambda q, *, top_k, filters: [],
            generate=lambda p: "Answer",
            verifier=ClaimVerifier(verify_fn),
        )
        result = gen.answer("Q?")
        assert result["insufficient_evidence"] is True
        # Verifier must not be called when no evidence was retrieved
        assert len(calls) == 0


# ---------------------------------------------------------------------------
# Fix 1 — _sanitise_answer: unknown citation markers stripped from prose
# ---------------------------------------------------------------------------

class TestSanitiseAnswer:
    """Tests for _sanitise_answer().

    Verifies that unknown [S<n>] labels are removed from text while valid
    citations and all surrounding prose are preserved unchanged.
    """

    def _sources(self, n: int) -> dict:
        return {f"S{i}": {} for i in range(1, n + 1)}

    # --- unknown labels removed ---

    def test_removes_single_unknown_label(self):
        result = _sanitise_answer("See [S99] for details.", self._sources(1))
        assert "[S99]" not in result
        assert "See" in result
        assert "for details." in result

    def test_removes_multiple_unknown_labels(self):
        result = _sanitise_answer("[S50] and [S99] are fabricated.", self._sources(2))
        assert "[S50]" not in result
        assert "[S99]" not in result

    def test_preserves_prose_around_removed_label(self):
        result = _sanitise_answer("Competition is high [S99].", self._sources(1))
        assert "Competition is high" in result
        assert "[S99]" not in result

    # --- valid labels preserved ---

    def test_keeps_valid_label(self):
        result = _sanitise_answer("Risk is noted in [S1].", self._sources(3))
        assert "[S1]" in result

    def test_keeps_all_valid_labels(self):
        result = _sanitise_answer("[S1] and [S2] agree.", self._sources(3))
        assert "[S1]" in result
        assert "[S2]" in result

    # --- mixed valid and invalid ---

    def test_mixed_keeps_valid_removes_invalid(self):
        result = _sanitise_answer(
            "Competition [S1]. Also see [S99].", self._sources(1)
        )
        assert "[S1]" in result
        assert "[S99]" not in result

    def test_mixed_prose_intact(self):
        result = _sanitise_answer(
            "Revenue grew per [S1] but [S99] is fabricated.", self._sources(1)
        )
        assert "Revenue grew per" in result
        assert "but" in result
        assert "is fabricated." in result

    # --- exact boundary: [S1] vs [S10] ---

    def test_s1_not_removed_when_s10_present(self):
        """[S1] is valid; [S10] may or may not be — both tested independently."""
        sources_one = self._sources(1)          # only S1 is valid
        result = _sanitise_answer("[S10] and [S1].", sources_one)
        assert "[S1]" in result
        assert "[S10]" not in result

    def test_s10_kept_when_registered(self):
        sources_ten = self._sources(10)         # S1 through S10 are valid
        result = _sanitise_answer("[S10] is relevant.", sources_ten)
        assert "[S10]" in result

    # --- repeated labels ---

    def test_repeated_valid_label_both_preserved(self):
        result = _sanitise_answer("[S1] is mentioned and [S1] again.", self._sources(2))
        # Both occurrences should survive.
        assert result.count("[S1]") == 2

    def test_repeated_invalid_label_both_removed(self):
        result = _sanitise_answer("[S99] here and [S99] there.", self._sources(1))
        assert "[S99]" not in result

    # --- edge cases ---

    def test_empty_string_unchanged(self):
        assert _sanitise_answer("", self._sources(3)) == ""

    def test_no_citations_in_text_unchanged(self):
        text = "Plain prose with no citations."
        assert _sanitise_answer(text, self._sources(3)) == text

    def test_only_valid_citations_all_preserved(self):
        text = "[S1] [S2] [S3]"
        result = _sanitise_answer(text, self._sources(3))
        assert result == text


# ---------------------------------------------------------------------------
# Fix 1 — regression: answer text in AnswerGenerator has unknowns stripped
# ---------------------------------------------------------------------------

class TestAnswerTextSanitisation:
    """Regression tests confirming AnswerGenerator strips unknown markers
    from the returned ``answer`` string, not just from the citation list."""

    def _make_generator(self, chunks, response):
        def retrieve(q, *, top_k, filters):
            return chunks

        def generate(prompt):
            return response

        return AnswerGenerator(retrieve, generate)

    def test_unknown_label_absent_from_answer_text(self):
        gen = self._make_generator(
            [SAMPLE_CHUNK],
            "Competition is a risk [S1]. Also fabricated [S99].",
        )
        result = gen.answer("What are the risks?")
        assert "[S99]" not in result["answer"]

    def test_valid_label_present_in_answer_text(self):
        gen = self._make_generator(
            [SAMPLE_CHUNK],
            "Competition is a risk [S1]. Also fabricated [S99].",
        )
        result = gen.answer("What are the risks?")
        assert "[S1]" in result["answer"]

    def test_prose_preserved_after_unknown_label_stripped(self):
        gen = self._make_generator(
            [SAMPLE_CHUNK],
            "Revenue grew [S1]. Unrelated claim noted [S99].",
        )
        result = gen.answer("Q?")
        assert "Revenue grew" in result["answer"]
        assert "Unrelated claim noted" in result["answer"]

    def test_only_unknown_labels_returns_insufficient_evidence(self):
        """If stripping unknowns leaves no valid citations, apply policy."""
        gen = self._make_generator(
            [SAMPLE_CHUNK],
            "Some claim from [S99].",
        )
        result = gen.answer("Q?")
        assert result["insufficient_evidence"] is True
        assert result["citations"] == []

    def test_s1_not_stripped_when_s10_also_present(self):
        """With 10 chunks, [S10] is valid; [S1] must also remain."""
        gen = self._make_generator(
            TEN_CHUNKS,
            "Both [S1] and [S10] are relevant.",
        )
        result = gen.answer("Q?")
        assert "[S1]" in result["answer"]
        assert "[S10]" in result["answer"]


# ---------------------------------------------------------------------------
# Fix 2 — _validate_verifier_result: strict output validation
# ---------------------------------------------------------------------------

class TestValidateVerifierResult:
    """Tests for _validate_verifier_result().

    Confirms that malformed verifier returns are rejected safely and that
    valid results pass through correctly.
    """

    # --- non-dict returns ---

    def test_none_result_is_malformed(self):
        out = _validate_verifier_result(None)
        assert out["verified"] is False
        assert out["verification_available"] is False
        assert "error" in out

    def test_string_result_is_malformed(self):
        out = _validate_verifier_result("verified")
        assert out["verified"] is False
        assert out["verification_available"] is False
        assert "error" in out

    def test_list_result_is_malformed(self):
        out = _validate_verifier_result([{"verified": True}])
        assert out["verified"] is False
        assert out["verification_available"] is False

    def test_integer_result_is_malformed(self):
        out = _validate_verifier_result(42)
        assert out["verified"] is False

    # --- missing or non-bool 'verified' ---

    def test_missing_verified_key_is_malformed(self):
        out = _validate_verifier_result({"unsupported_claims": []})
        assert out["verified"] is False
        assert out["verification_available"] is False

    def test_integer_verified_is_rejected(self):
        """verified=1 must not be accepted as True."""
        out = _validate_verifier_result({"verified": 1})
        assert out["verified"] is False
        assert out["verification_available"] is False

    def test_string_verified_is_rejected(self):
        out = _validate_verifier_result({"verified": "true"})
        assert out["verified"] is False

    def test_none_verified_is_rejected(self):
        out = _validate_verifier_result({"verified": None})
        assert out["verified"] is False

    # --- malformed list fields ---

    def test_unsupported_claims_not_list_normalised_to_empty(self):
        raw = {"verified": False, "unsupported_claims": "all claims", "conflicts": []}
        out = _validate_verifier_result(raw)
        assert out["unsupported_claims"] == []
        assert "normalisation_notes" in out

    def test_conflicts_not_list_normalised_to_empty(self):
        raw = {"verified": True, "unsupported_claims": [], "conflicts": {"a": "b"}}
        out = _validate_verifier_result(raw)
        assert out["conflicts"] == []
        assert "normalisation_notes" in out

    # --- valid results pass through ---

    def test_valid_verified_true_passes_through(self):
        raw = {"verified": True, "unsupported_claims": [], "conflicts": []}
        out = _validate_verifier_result(raw)
        assert out["verified"] is True
        assert out["verification_available"] is True

    def test_valid_verified_false_passes_through(self):
        raw = {
            "verified": False,
            "unsupported_claims": ["Claim A"],
            "conflicts": [],
        }
        out = _validate_verifier_result(raw)
        assert out["verified"] is False
        assert out["unsupported_claims"] == ["Claim A"]

    def test_absent_optional_fields_default_correctly(self):
        raw = {"verified": True}
        out = _validate_verifier_result(raw)
        assert out["unsupported_claims"] == []
        assert out["conflicts"] == []
        assert out["verification_available"] is True


# ---------------------------------------------------------------------------
# Fix 2 — ClaimVerifier hardened against malformed verifier returns
# ---------------------------------------------------------------------------

class TestClaimVerifierHardened:
    """Confirms ClaimVerifier.verify() rejects or safely normalises malformed
    verifier callable return values.

    These are distinct from the exception-path tests in TestClaimVerifier;
    those cover when verify_fn raises, while these cover when it returns
    a syntactically invalid value.
    """

    def _chunks(self):
        return [SAMPLE_CHUNK]

    def test_none_return_from_verifier_is_safe_fallback(self):
        verifier = ClaimVerifier(lambda q, a, c: None)  # type: ignore[return-value]
        out = verifier.verify("Q?", "Answer.", self._chunks())
        assert out["verified"] is False
        assert out["verification_available"] is False

    def test_string_return_from_verifier_is_safe_fallback(self):
        verifier = ClaimVerifier(lambda q, a, c: "verified")  # type: ignore[return-value]
        out = verifier.verify("Q?", "Answer.", self._chunks())
        assert out["verified"] is False

    def test_list_return_from_verifier_is_safe_fallback(self):
        verifier = ClaimVerifier(lambda q, a, c: [])  # type: ignore[return-value]
        out = verifier.verify("Q?", "Answer.", self._chunks())
        assert out["verified"] is False

    def test_integer_verified_not_accepted(self):
        """verified=1 is not a bool; must not be treated as verified=True."""
        verifier = ClaimVerifier(lambda q, a, c: {"verified": 1})
        out = verifier.verify("Q?", "Answer.", self._chunks())
        assert out["verified"] is False
        assert out["verification_available"] is False

    def test_string_verified_not_accepted(self):
        verifier = ClaimVerifier(lambda q, a, c: {"verified": "yes"})
        out = verifier.verify("Q?", "Answer.", self._chunks())
        assert out["verified"] is False

    def test_missing_verified_key_safe_fallback(self):
        verifier = ClaimVerifier(lambda q, a, c: {"unsupported_claims": []})
        out = verifier.verify("Q?", "Answer.", self._chunks())
        assert out["verified"] is False
        assert out["verification_available"] is False

    def test_unsupported_claims_non_list_normalised(self):
        verifier = ClaimVerifier(
            lambda q, a, c: {
                "verified": False,
                "unsupported_claims": "some claim",
                "conflicts": [],
            }
        )
        out = verifier.verify("Q?", "Answer.", self._chunks())
        assert isinstance(out["unsupported_claims"], list)

    def test_conflicts_non_list_normalised(self):
        verifier = ClaimVerifier(
            lambda q, a, c: {
                "verified": True,
                "unsupported_claims": [],
                "conflicts": 42,
            }
        )
        out = verifier.verify("Q?", "Answer.", self._chunks())
        assert isinstance(out["conflicts"], list)

    def test_valid_bool_true_accepted(self):
        verifier = ClaimVerifier(
            lambda q, a, c: {"verified": True, "unsupported_claims": [], "conflicts": []}
        )
        out = verifier.verify("Q?", "Answer.", self._chunks())
        assert out["verified"] is True

    def test_valid_bool_false_accepted(self):
        verifier = ClaimVerifier(
            lambda q, a, c: {
                "verified": False,
                "unsupported_claims": ["Claim X"],
                "conflicts": [],
            }
        )
        out = verifier.verify("Q?", "Answer.", self._chunks())
        assert out["verified"] is False
        assert "Claim X" in out["unsupported_claims"]

    def test_not_implemented_error_still_returns_safe_fallback(self):
        def bad(q, a, c):
            raise NotImplementedError

        verifier = ClaimVerifier(bad)
        out = verifier.verify("Q?", "Answer.", self._chunks())
        assert out["verified"] is False
        assert out["verification_available"] is False

    def test_unexpected_exception_still_returns_safe_fallback(self):
        def bad(q, a, c):
            raise RuntimeError("exploded")

        verifier = ClaimVerifier(bad)
        out = verifier.verify("Q?", "Answer.", self._chunks())
        assert out["verified"] is False
        assert "exploded" in out["error"]
