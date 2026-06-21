"""Tests for digest_generator/core/summary/: LLM-driven ContentSummarizer.

Mocks the Ollama client. Mirrors the digest stage test pattern (see
``tests/core/test_editorial.py``) with fixture-built entries, ``MagicMock``
client, and a ``_mock_response`` helper for the chat shape.
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from digest_generator.core.summary import ContentSummarizer
from digest_generator.core.types import Entry


def _mock_response(content: str) -> MagicMock:
    """Build a mock ollama chat response shape."""
    resp = MagicMock()
    resp.message.content = content
    return resp


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def now():
    return datetime.now(tz=UTC)


@pytest.fixture
def sample_entries(now):
    return [
        Entry(
            title="OpenAI ships GPT-5",
            url="https://example.com/1",
            origin="openai",
            published=now,
            description="OpenAI announced GPT-5 with improved reasoning.",
            content="Full content of article one about GPT-5 release.",
            content_head="Truncated head of article one.",
        ),
        Entry(
            title="Cloudflare patches CVE-2026-0142",
            url="https://example.com/2",
            origin="github",
            published=now,
            description="Cloudflare disclosed a Workers RCE.",
            content="Full content of article two about CVE-2026-0142.",
            content_head="Truncated head of article two.",
        ),
    ]


@pytest.fixture
def summarizer():
    mock_client = MagicMock()
    return ContentSummarizer(client=mock_client, model="test-summarizer-model")


# =============================================================================
# summarize_entries: fan-out and ordering
# =============================================================================


class TestSummarizeEntriesFanOut:
    def test_returns_one_summary_per_entry(self, summarizer, sample_entries):
        summarizer._client.chat.return_value = _mock_response("Extracted fact.")
        result = asyncio.run(summarizer.summarize_entries(sample_entries))
        assert len(result) == 2

    def test_preserves_input_order(self, summarizer, sample_entries):
        summarizer._client.chat.side_effect = [
            _mock_response("Summary A."),
            _mock_response("Summary B."),
        ]
        result = asyncio.run(summarizer.summarize_entries(sample_entries))
        assert result[0].entry is sample_entries[0]
        assert result[1].entry is sample_entries[1]
        assert result[0].summary == "Summary A."
        assert result[1].summary == "Summary B."

    def test_empty_input_returns_empty_list(self, summarizer):
        result = asyncio.run(summarizer.summarize_entries([]))
        assert result == []
        summarizer._client.chat.assert_not_called()

    def test_summary_strips_whitespace(self, summarizer, sample_entries):
        summarizer._client.chat.return_value = _mock_response("  Trimmed.  \n")
        result = asyncio.run(summarizer.summarize_entries(sample_entries[:1]))
        assert result[0].summary == "Trimmed."

    def test_topics_initialized_empty(self, summarizer, sample_entries):
        """Classifier populates topics later, so summarizer leaves them empty."""
        summarizer._client.chat.return_value = _mock_response("Fact.")
        result = asyncio.run(summarizer.summarize_entries(sample_entries[:1]))
        assert result[0].topics == []

    def test_length_matches_summary(self, summarizer, sample_entries):
        summarizer._client.chat.return_value = _mock_response("Twelve chars.")
        result = asyncio.run(summarizer.summarize_entries(sample_entries[:1]))
        assert result[0].length == len(result[0].summary)


# =============================================================================
# summarize_entry: fallback behavior on empty LLM response
# =============================================================================


class TestFallback:
    def test_empty_response_falls_back_to_content_head(self, summarizer, sample_entries):
        summarizer._client.chat.return_value = _mock_response("")
        result = asyncio.run(summarizer.summarize_entry(sample_entries[0]))
        assert result.summary == "Truncated head of article one."

    def test_content_head_truncated_at_1000_chars(self, summarizer, now):
        long_head = "A" * 2500
        entry = Entry(
            title="t",
            url="u",
            origin="openai",
            published=now,
            description="d",
            content="c",
            content_head=long_head,
        )
        summarizer._client.chat.return_value = _mock_response("")
        result = asyncio.run(summarizer.summarize_entry(entry))
        assert len(result.summary) == 1000
        assert result.summary == "A" * 1000

    def test_falls_back_to_description_when_no_content_head(self, summarizer, now):
        entry = Entry(
            title="t",
            url="u",
            origin="openai",
            published=now,
            description="Description-only fallback.",
            content="",
            content_head="",
        )
        summarizer._client.chat.return_value = _mock_response("")
        result = asyncio.run(summarizer.summarize_entry(entry))
        assert result.summary == "Description-only fallback."

    def test_whitespace_only_response_triggers_fallback(self, summarizer, sample_entries):
        summarizer._client.chat.return_value = _mock_response("   \n\t  ")
        result = asyncio.run(summarizer.summarize_entry(sample_entries[0]))
        assert result.summary == "Truncated head of article one."


# =============================================================================
# user-prompt format: XML-tag mirror of writer._format_articles
# =============================================================================


class TestUserPrompt:
    def test_includes_title_url_source_published(self, summarizer, sample_entries):
        summarizer._client.chat.return_value = _mock_response("ok")
        asyncio.run(summarizer.summarize_entries(sample_entries[:1]))
        user_prompt = summarizer._client.chat.call_args.kwargs["messages"][1]["content"]
        assert "<title>OpenAI ships GPT-5</title>" in user_prompt
        assert "<url>https://example.com/1</url>" in user_prompt
        assert "<origin>openai</origin>" in user_prompt
        assert "<published>" in user_prompt

    def test_includes_description_when_present(self, summarizer, sample_entries):
        summarizer._client.chat.return_value = _mock_response("ok")
        asyncio.run(summarizer.summarize_entries(sample_entries[:1]))
        user_prompt = summarizer._client.chat.call_args.kwargs["messages"][1]["content"]
        assert (
            "<description>OpenAI announced GPT-5 with improved reasoning.</description>"
            in user_prompt
        )

    def test_omits_empty_description(self, summarizer, now):
        entry = Entry(
            title="t",
            url="u",
            origin="openai",
            published=now,
            description="",
            content="some content",
            content_head="",
        )
        summarizer._client.chat.return_value = _mock_response("ok")
        asyncio.run(summarizer.summarize_entry(entry))
        user_prompt = summarizer._client.chat.call_args.kwargs["messages"][1]["content"]
        assert "<description>" not in user_prompt

    def test_omits_empty_content(self, summarizer, now):
        entry = Entry(
            title="t",
            url="u",
            origin="openai",
            published=now,
            description="d",
            content="",
            content_head="",
        )
        summarizer._client.chat.return_value = _mock_response("ok")
        asyncio.run(summarizer.summarize_entry(entry))
        user_prompt = summarizer._client.chat.call_args.kwargs["messages"][1]["content"]
        assert "<content>" not in user_prompt

    def test_does_not_send_content_head_separately(self, summarizer, sample_entries):
        """content_head is a derived truncation of content; sending both is redundant."""
        summarizer._client.chat.return_value = _mock_response("ok")
        asyncio.run(summarizer.summarize_entries(sample_entries[:1]))
        user_prompt = summarizer._client.chat.call_args.kwargs["messages"][1]["content"]
        assert "<content_head>" not in user_prompt


# =============================================================================
# DI and configuration
# =============================================================================


class TestSharedSemaphore:
    """The concurrency cap is per-instance, not per-call.

    Cloud Ollama 429s when the per-call ceiling multiplies by parallel feeds.
    Tests pin the lazy-init contract and the cross-call sharing.
    """

    def test_semaphore_lazy_initialized_on_first_call(self, summarizer, sample_entries):
        assert summarizer._semaphore is None
        summarizer._client.chat.return_value = _mock_response("ok")
        asyncio.run(summarizer.summarize_entries(sample_entries[:1]))
        assert summarizer._semaphore is not None

    def test_semaphore_shared_across_summarize_entries_calls(self, summarizer, sample_entries):
        """Two consecutive calls reuse the same semaphore: one global cap, not two."""
        summarizer._client.chat.return_value = _mock_response("ok")
        asyncio.run(summarizer.summarize_entries(sample_entries[:1]))
        first = summarizer._semaphore
        asyncio.run(summarizer.summarize_entries(sample_entries[1:]))
        assert summarizer._semaphore is first

    def test_empty_input_does_not_initialize_semaphore(self, summarizer):
        """No work to do means no semaphore allocation."""
        asyncio.run(summarizer.summarize_entries([]))
        assert summarizer._semaphore is None


class TestConfiguration:
    def test_uses_configured_model(self, summarizer, sample_entries):
        summarizer._client.chat.return_value = _mock_response("ok")
        asyncio.run(summarizer.summarize_entries(sample_entries[:1]))
        assert summarizer._client.chat.call_args.kwargs["model"] == "test-summarizer-model"

    def test_passes_summarizer_temperature(self, summarizer, sample_entries):
        summarizer._client.chat.return_value = _mock_response("ok")
        asyncio.run(summarizer.summarize_entries(sample_entries[:1]))
        options = summarizer._client.chat.call_args.kwargs["options"]
        assert "temperature" in options

    def test_system_prompt_loaded(self, summarizer, sample_entries):
        summarizer._client.chat.return_value = _mock_response("ok")
        asyncio.run(summarizer.summarize_entries(sample_entries[:1]))
        system = summarizer._client.chat.call_args.kwargs["messages"][0]
        assert system["role"] == "system"
        assert "fact-extraction" in system["content"]
        # Style placeholders should be resolved
        assert "{{style:" not in system["content"]
