"""Tests for digest_generator/core/digest/stages/composer.py: DigestComposer mechanical assembly."""

import pytest

from digest_generator.core.digest.stages.composer import DigestComposer
from digest_generator.core.digest.types import DigestFraming, DigestResult, SectionDraft, WatchItem


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
def framing():
    return DigestFraming(
        title="GPT-5 Meets a Widening Trust Deficit",
        intro="OpenAI shipped GPT-5 this week alongside a Cloudflare CVE disclosure.",
    )


@pytest.fixture
def watch():
    return [
        WatchItem(heading="Open-weight models close the gap", body="Mistral and DeepSeek."),
        WatchItem(heading="Supply-chain attacks intensify", body="CVE disclosures rising."),
    ]


@pytest.fixture
def composer():
    return DigestComposer()


# =============================================================================
# compose: happy path
# =============================================================================


class TestCompose:
    def test_returns_digest_result(self, composer, sections, framing, watch):
        result = composer.compose(sections, framing, watch, date_range=("2026-03-10", "2026-03-17"))
        assert isinstance(result, DigestResult)

    def test_content_omits_h1_title(self, composer, sections, framing, watch):
        """Title lives in frontmatter (built by io.build_digest_markdown);
        the body must never start with an H1, since every consumer renders
        the title from frontmatter and would otherwise show it twice."""
        result = composer.compose(sections, framing, watch)
        assert not any(line.startswith("# ") for line in result.content.splitlines())

    def test_includes_overview_heading_and_intro(self, composer, sections, framing, watch):
        result = composer.compose(sections, framing, watch)
        chunks = result.content.split("\n\n", 2)
        assert chunks[0] == "## Overview"
        assert chunks[1].startswith("OpenAI shipped GPT-5")

    def test_overview_heading_precedes_first_section(self, composer, sections, framing, watch):
        result = composer.compose(sections, framing, watch)
        overview_idx = result.content.index("## Overview")
        first_section_idx = result.content.index("## AI & Machine Learning")
        assert overview_idx < first_section_idx

    def test_overview_heading_omitted_when_intro_empty(self, composer, sections, watch):
        framing_no_intro = DigestFraming(title="Test Title", intro="")
        result = composer.compose(sections, framing_no_intro, watch)
        assert "## Overview" not in result.content

    def test_preserves_section_content_verbatim(self, composer, sections, framing, watch):
        result = composer.compose(sections, framing, watch)
        for section in sections:
            assert section.content.strip() in result.content

    def test_sections_appear_in_input_order(self, composer, sections, framing, watch):
        result = composer.compose(sections, framing, watch)
        ai_idx = result.content.index("## AI & Machine Learning")
        sec_idx = result.content.index("## Security")
        assert ai_idx < sec_idx

    def test_watch_section_appended_at_end(self, composer, sections, framing, watch):
        result = composer.compose(sections, framing, watch)
        watch_idx = result.content.index("## What to Watch")
        last_section_idx = result.content.index("## Security")
        assert last_section_idx < watch_idx

    def test_watch_items_render_as_h3(self, composer, sections, framing, watch):
        result = composer.compose(sections, framing, watch)
        assert "### Open-weight models close the gap" in result.content
        assert "### Supply-chain attacks intensify" in result.content

    def test_watch_item_body_follows_heading(self, composer, sections, framing, watch):
        result = composer.compose(sections, framing, watch)
        assert "### Open-weight models close the gap\n\nMistral and DeepSeek." in result.content

    def test_no_what_to_watch_when_items_empty(self, composer, sections, framing):
        result = composer.compose(sections, framing, [])
        assert "## What to Watch" not in result.content

    def test_title_propagated(self, composer, sections, framing, watch):
        result = composer.compose(sections, framing, watch)
        assert result.title == framing.title


# =============================================================================
# metadata
# =============================================================================


class TestMetadata:
    def test_section_counts_populated(self, composer, sections, framing, watch):
        result = composer.compose(sections, framing, watch)
        assert result.section_counts == {
            "AI & Machine Learning": 20,
            "Security": 7,
        }

    def test_article_count_is_sum_of_sections(self, composer, sections, framing, watch):
        result = composer.compose(sections, framing, watch)
        assert result.article_count == 27

    def test_word_count_reflects_composed_content(self, composer, sections, framing, watch):
        result = composer.compose(sections, framing, watch)
        assert result.word_count == len(result.content.split())

    def test_reading_time_at_least_one_minute(self, composer, sections, framing, watch):
        result = composer.compose(sections, framing, watch)
        assert result.reading_time_minutes >= 1

    def test_date_from_date_range(self, composer, sections, framing, watch):
        result = composer.compose(sections, framing, watch, date_range=("2026-03-10", "2026-03-17"))
        assert result.date == "2026-03-17"

    def test_date_defaults_to_today_when_no_range(self, composer, sections, framing, watch):
        result = composer.compose(sections, framing, watch)
        # Just verify format: YYYY-MM-DD
        assert len(result.date) == 10
        assert result.date[4] == "-"
        assert result.date[7] == "-"


# =============================================================================
# edge cases
# =============================================================================


class TestEdgeCases:
    def test_empty_sections_returns_empty_content(self, composer, framing):
        result = composer.compose([], framing, [])
        assert result.content == ""
        assert result.article_count == 0
        assert result.section_counts == {}
        assert result.title == framing.title

    def test_empty_intro_starts_with_first_section(self, composer, sections, watch):
        """With no intro, the body opens directly on the first section's H2,
        with no H1 and no ``## Overview`` placeholder."""
        framing = DigestFraming(title="T", intro="")
        result = composer.compose(sections, framing, watch)
        assert result.content.startswith("## AI & Machine Learning")
        assert "# T" not in result.content
        assert "## Overview" not in result.content
        assert result.title == "T"

    def test_single_section_and_no_watch(self, composer, framing):
        drafts = [SectionDraft(name="A", content="## A\n\nBody.", article_count=1)]
        result = composer.compose(drafts, framing, [])
        assert "## What to Watch" not in result.content
        assert "## A" in result.content
        assert result.article_count == 1
