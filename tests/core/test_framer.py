"""Tests for digest_generator/core/digest/stages/framer.py: DigestFramer title and intro generation."""

from unittest.mock import MagicMock

import pytest

from digest_generator.core.digest.stages.framer import (
    _INTRO_SYSTEM_PROMPT,
    _TITLE_SYSTEM_PROMPT,
    DigestFramer,
    _build_retry_feedback,
    _clean_title,
    _format_period,
    _format_sections,
    _title_issues,
)
from digest_generator.core.digest.types import DigestFraming, SectionDraft
from digest_generator.shared import settings as settings_module


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.message.content = content
    return resp


@pytest.fixture
def sections():
    return [
        SectionDraft(
            name="AI & Machine Learning",
            content="## AI & Machine Learning\n\nOpenAI shipped GPT-5.",
            article_count=20,
        ),
        SectionDraft(
            name="Security",
            content="## Security\n\nCloudflare disclosed CVE-2026-0142.",
            article_count=7,
        ),
    ]


@pytest.fixture
def framer():
    mock_client = MagicMock()
    return DigestFramer(client=mock_client, model="test-framer-model")


# =============================================================================
# frame: end-to-end
# =============================================================================


class TestFrame:
    def test_returns_digest_framing(self, framer, sections):
        framer._client.chat.side_effect = [
            _mock_response("GPT-5 Meets a Widening Trust Deficit"),
            _mock_response(
                "OpenAI shipped GPT-5 this week alongside Cloudflare's disclosure of CVE-2026-0142."
            ),
        ]
        result = framer.frame(sections, date_range=("2026-03-10", "2026-03-17"))
        assert isinstance(result, DigestFraming)
        assert result.title == "GPT-5 Meets a Widening Trust Deficit"
        assert result.intro.startswith("OpenAI shipped")

    def test_makes_two_llm_calls(self, framer, sections):
        framer._client.chat.side_effect = [
            _mock_response("Title Here"),
            _mock_response("Intro here."),
        ]
        framer.frame(sections)
        assert framer._client.chat.call_count == 2

    def test_title_call_uses_title_prompt(self, framer, sections):
        framer._client.chat.side_effect = [
            _mock_response("Title Here"),
            _mock_response("Intro here."),
        ]
        framer.frame(sections)
        first_call = framer._client.chat.call_args_list[0]
        assert first_call.kwargs["messages"][0]["content"] == _TITLE_SYSTEM_PROMPT

    def test_intro_call_uses_intro_prompt(self, framer, sections):
        framer._client.chat.side_effect = [
            _mock_response("Title Here"),
            _mock_response("Intro here."),
        ]
        framer.frame(sections)
        second_call = framer._client.chat.call_args_list[1]
        assert second_call.kwargs["messages"][0]["content"] == _INTRO_SYSTEM_PROMPT

    def test_intro_prompt_includes_title(self, framer, sections):
        framer._client.chat.side_effect = [
            _mock_response("Unique Title 12345"),
            _mock_response("Intro here."),
        ]
        framer.frame(sections)
        second_call = framer._client.chat.call_args_list[1]
        assert "Unique Title 12345" in second_call.kwargs["messages"][1]["content"]

    def test_date_range_threaded_through(self, framer, sections):
        framer._client.chat.side_effect = [
            _mock_response("Title"),
            _mock_response("Intro."),
        ]
        framer.frame(sections, date_range=("2026-03-10", "2026-03-17"))
        for call in framer._client.chat.call_args_list:
            user_prompt = call.kwargs["messages"][1]["content"]
            assert "2026-03-10" in user_prompt
            assert "2026-03-17" in user_prompt

    def test_strips_non_breaking_hyphens_from_title_and_intro(self, framer, sections):

        framer._client.chat.side_effect = [
            _mock_response("Cloudflare‑Stripe Protocol Tightens Agent Controls"),  # noqa: RUF001
            _mock_response(
                "Cloudflare and Stripe shipped an AI‑ready protocol; "  # noqa: RUF001
                "Auth0 urged policy‑as‑code."  # noqa: RUF001
            ),
        ]
        result = framer.frame(sections)
        assert "‑" not in result.title  # noqa: RUF001  # verifying U+2011 stripped
        assert "‑" not in result.intro  # noqa: RUF001  # verifying U+2011 stripped
        assert "Cloudflare-Stripe Protocol" in result.title
        assert "AI-ready protocol" in result.intro
        assert "policy-as-code" in result.intro


# =============================================================================
# model fallback
# =============================================================================


class TestModelFallback:
    def test_explicit_model_wins(self):
        editor = DigestFramer(client=MagicMock(), model="explicit-model")
        assert editor.model == "explicit-model"

    def test_framer_model_setting_used_when_set(self, monkeypatch):
        monkeypatch.setattr(settings_module.settings, "framer_model", "framer-only")
        monkeypatch.setattr(settings_module.settings, "writer_model", "writer-only")
        framer = DigestFramer(client=MagicMock())
        assert framer.model == "framer-only"

    def test_falls_back_to_writer_model(self, monkeypatch):
        monkeypatch.setattr(settings_module.settings, "framer_model", None)
        monkeypatch.setattr(settings_module.settings, "writer_model", "writer-only")
        framer = DigestFramer(client=MagicMock())
        assert framer.model == "writer-only"


# =============================================================================
# helpers
# =============================================================================


class TestCleanTitle:
    def test_strips_whitespace(self):
        assert _clean_title("  Hello World  \n") == "Hello World"

    def test_strips_double_quotes(self):
        assert _clean_title('"Hello World"') == "Hello World"

    def test_strips_single_quotes(self):
        assert _clean_title("'Hello World'") == "Hello World"

    def test_preserves_internal_quotes(self):
        assert _clean_title('A "quoted" phrase') == 'A "quoted" phrase'


class TestFormatPeriod:
    def test_with_date_range(self):
        assert _format_period(("2026-03-10", "2026-03-17")) == "covering 2026-03-10 to 2026-03-17"

    def test_without_date_range_uses_today(self):
        period = _format_period(None)
        assert period.startswith("week of ")


class TestFormatSections:
    def test_wraps_each_section(self, sections):
        lines = _format_sections(sections)
        text = "\n".join(lines)
        assert text.startswith("<sections>")
        assert text.endswith("</sections>")
        assert text.count("<section>") == 2
        assert "<name>AI & Machine Learning</name>" in text
        assert "<article-count>20</article-count>" in text
        assert "## Security" in text


# =============================================================================
# title retry-with-feedback
# =============================================================================


class TestTitleIssues:
    def test_clean_title_has_no_issues(self):
        assert _title_issues("GPT-5 Meets a Widening Trust Deficit") == []

    def test_detects_ai_prefix(self):
        issues = _title_issues("AI Agents Drive Enterprise Workflows")
        assert len(issues) == 1
        assert "AI " in issues[0]

    def test_detects_colon_split(self):
        issues = _title_issues("Security: A Quarter in Review")
        assert len(issues) == 1
        assert "colon" in issues[0]

    def test_ai_colon_only_flagged_as_colon(self):
        # "AI: Something" only matches ^\w+:\s; the ^AI\s pattern requires
        # whitespace immediately after "AI", not a colon. The two patterns
        # are mutually exclusive on the same prefix.
        issues = _title_issues("AI: Something Happened")
        assert len(issues) == 1
        assert "colon" in issues[0]

    def test_internal_colon_not_flagged(self):
        # Colons mid-title (e.g. quotes) don't trigger the ^\w+:\s pattern
        assert _title_issues('Sam Altman on AGI: "imminent"') == []

    def test_detects_brand_plus_announcement_verb(self):
        # Single-word brand + verb: "OpenAI Releases GPT-5"
        issues = _title_issues("OpenAI Releases GPT-5 With Vision Tools")
        assert len(issues) == 1
        assert "vendor marketing" in issues[0]

    def test_detects_two_word_brand_plus_verb(self):
        # Multi-word product + verb: "Claude Mythos Finds 271 Zero-Days"
        issues = _title_issues("Claude Mythos Finds 271 Zero-Days, Shaking Cyber Defenses")
        assert len(issues) == 1
        assert "vendor marketing" in issues[0]

    def test_detects_rolls_out_two_word_verb(self):
        # The two-word "Rolls Out" should match across the whitespace.
        issues = _title_issues("Microsoft Rolls Out Copilot Studio Updates")
        assert len(issues) == 1
        assert "vendor marketing" in issues[0]

    def test_brand_led_skips_non_announcement_verb(self):
        # "Capital Concentrated Around Infrastructure" is not in verb list.
        assert _title_issues("Capital Consolidation Collides with Geopolitical Fractures") == []

    def test_brand_led_skips_hyphenated_first_word(self):
        # Hyphenated first words break the brand-pattern match (no leading
        # whitespace word boundary), preserving good titles like
        # "Zero-Day Discovery Collides with Agentic Production Risks".
        assert _title_issues("Zero-Day Discovery Collides with Agentic Production Risks") == []


class TestBuildRetryFeedback:
    def test_includes_previous_attempt_verbatim(self):
        feedback = _build_retry_feedback("AI Drives Things", ["starts with 'AI '"])
        assert '"AI Drives Things"' in feedback

    def test_combines_multiple_issues_with_and(self):
        feedback = _build_retry_feedback("AI: Stuff", ["starts with 'AI '", "uses the colon"])
        assert "starts with 'AI ' and uses the colon" in feedback

    def test_wrapped_in_retry_feedback_tags(self):
        feedback = _build_retry_feedback("Bad Title", ["bad reason"])
        assert feedback.startswith("<retry-feedback>")
        assert feedback.endswith("</retry-feedback>")


class TestRetryWithFeedback:
    def test_no_retry_when_title_clean(self, framer, sections):
        framer._client.chat.side_effect = [
            _mock_response("GPT-5 Meets a Widening Trust Deficit"),
            _mock_response("Intro paragraph here."),
        ]
        result = framer.frame(sections)
        assert framer._client.chat.call_count == 2
        assert result.title == "GPT-5 Meets a Widening Trust Deficit"

    def test_retries_on_ai_prefix(self, framer, sections):
        framer._client.chat.side_effect = [
            _mock_response("AI Agents Drive Enterprise Workflows"),
            _mock_response("Sovereign Models Rise as Trust Falls"),
            _mock_response("Intro paragraph here."),
        ]
        result = framer.frame(sections)
        assert framer._client.chat.call_count == 3
        assert result.title == "Sovereign Models Rise as Trust Falls"

    def test_retries_on_colon_split(self, framer, sections):
        framer._client.chat.side_effect = [
            _mock_response("Security: Quarter in Review"),
            _mock_response("Cloudflare Discloses Critical CVE"),
            _mock_response("Intro paragraph here."),
        ]
        result = framer.frame(sections)
        assert framer._client.chat.call_count == 3
        assert result.title == "Cloudflare Discloses Critical CVE"

    def test_retry_user_prompt_includes_feedback(self, framer, sections):
        framer._client.chat.side_effect = [
            _mock_response("AI Reshapes Everything"),
            _mock_response("Better Title Here"),
            _mock_response("Intro."),
        ]
        framer.frame(sections)
        retry_call = framer._client.chat.call_args_list[1]
        retry_prompt = retry_call.kwargs["messages"][1]["content"]
        assert "<retry-feedback>" in retry_prompt
        assert '"AI Reshapes Everything"' in retry_prompt
        assert "AI " in retry_prompt  # the reason explanation

    def test_retried_title_threaded_to_intro(self, framer, sections):
        framer._client.chat.side_effect = [
            _mock_response("AI Wins Big"),
            _mock_response("Specific Event That Mattered"),
            _mock_response("Intro paragraph."),
        ]
        framer.frame(sections)
        intro_call = framer._client.chat.call_args_list[2]
        assert "Specific Event That Mattered" in intro_call.kwargs["messages"][1]["content"]

    def test_retried_title_still_bad_is_accepted(self, framer, sections):
        framer._client.chat.side_effect = [
            _mock_response("AI Does Stuff"),
            _mock_response("AI Still Does Stuff"),
            _mock_response("Intro."),
        ]
        result = framer.frame(sections)
        assert framer._client.chat.call_count == 3
        # Retry-once-only: second bad title is accepted, no third title call
        assert result.title == "AI Still Does Stuff"

    def test_no_retry_extra_call_when_title_passes(self, framer, sections):
        # Belt-and-braces: non-bad title produces exactly title + intro, no extras
        framer._client.chat.side_effect = [
            _mock_response("Capital Consolidation Meets Geopolitical Fractures"),
            _mock_response("Intro."),
        ]
        framer.frame(sections)
        assert framer._client.chat.call_count == 2
