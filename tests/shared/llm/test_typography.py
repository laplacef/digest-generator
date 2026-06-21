"""Tests for digest_generator/shared/llm/typography.py: post-LLM ASCII normalization."""

from digest_generator.shared.llm.typography import normalize_typography


class TestNormalizeTypography:
    def test_strips_non_breaking_hyphen(self):
        assert normalize_typography("AI‑ready") == "AI-ready"  # noqa: RUF001

    def test_strips_multiple_occurrences(self):

        result = normalize_typography("Policy‑as‑Code is non‑human-friendly")  # noqa: RUF001
        assert result == "Policy-as-Code is non-human-friendly"

    def test_passes_through_clean_ascii(self):
        text = "GPT-5 Meets a Widening Trust Deficit"
        assert normalize_typography(text) is text or normalize_typography(text) == text

    def test_passes_through_other_unicode(self):
        # Em dashes, en dashes, smart quotes are not in the strip table.
        text = "Anthropic — a $900B valuation — “Claude”"
        assert normalize_typography(text) == text

    def test_handles_empty_string(self):
        assert normalize_typography("") == ""
