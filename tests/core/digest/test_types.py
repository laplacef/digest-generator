"""Tests for digest_generator/core/digest/types.py: digest-internal dataclasses."""

from digest_generator.core.digest.types import DigestResult


class TestDigestResult:
    """Terminal digest output: title, content, frontmatter inputs."""

    def test_construction(self):
        result = DigestResult(
            title="AI Agents Reshape Workflows",
            content="# Digest\n\nBody text.",
            date="2026-03-17",
            word_count=150,
            reading_time_minutes=1,
            article_count=42,
            section_counts={"AI & Machine Learning": 20, "Security": 22},
        )
        assert result.title == "AI Agents Reshape Workflows"
        assert result.article_count == 42
        assert len(result.section_counts) == 2

    def test_default_section_counts(self):
        """section_counts defaults to an empty dict via field(default_factory)."""
        result = DigestResult(
            title="",
            content="",
            date="2026-03-17",
            word_count=0,
            reading_time_minutes=0,
            article_count=0,
        )
        assert result.section_counts == {}

    def test_equality(self):
        a = DigestResult(
            title="Title",
            content="Body",
            date="2026-03-17",
            word_count=100,
            reading_time_minutes=1,
            article_count=10,
        )
        b = DigestResult(
            title="Title",
            content="Body",
            date="2026-03-17",
            word_count=100,
            reading_time_minutes=1,
            article_count=10,
        )
        assert a == b


# `SectionDraft`, `DigestFraming`, `WatchItem` are exercised as fixtures
# throughout the digest stage tests (test_writer / test_editorial /
# test_framer / test_watcher / test_composer) and the io round-trip tests
# (test_io.py). Pure construction tests for them would duplicate that
# coverage with no additional contract checked.
