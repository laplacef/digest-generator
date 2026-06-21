"""Tests for digest_generator/sources/rss/types.py: RSS-specific enums and the Feed dataclass.

Cross-stage types (`Entry`, `Summary`, `ContentType`, `Filter`, `Label`,
`TopicType`) live in `digest_generator.core.types`; their tests are at
`tests/core/test_types.py`.
"""

from digest_generator.core.types import ContentType
from digest_generator.sources.rss.types import (
    BoilerplateMarker,
    Feed,
    SelectorType,
)


class TestBoilerplateMarker:
    """13 lowercase markers used by `_quality_check` to reject scraped boilerplate."""

    def test_count(self):
        assert len(BoilerplateMarker) == 13

    def test_values_are_lowercase(self):
        for marker in BoilerplateMarker:
            assert marker == marker.value.lower()


class TestSelectorType:
    """15 CSS selectors; iteration order is the article-extraction cascade order."""

    def test_count(self):
        assert len(SelectorType) == 15

    def test_first_is_article(self):
        """First selector tried should be the semantic <article> tag."""
        assert next(iter(SelectorType)) == SelectorType.ARTICLE

    def test_last_is_main(self):
        """Last selector (fallback) should be <main>."""
        assert list(SelectorType)[-1] == SelectorType.MAIN


class TestFeed:
    """Per-feed configuration carried in the registry."""

    def test_construction(self):
        feed = Feed(
            name="openai-blog",
            url="https://openai.com/blog/rss",
            content_type=ContentType.AI,
        )
        assert feed.name == "openai-blog"
        assert feed.url == "https://openai.com/blog/rss"
        assert feed.content_type == ContentType.AI

    def test_equality(self):
        feed_a = Feed(
            name="test",
            url="https://example.com/rss",
            content_type=ContentType.ENGINEERING,
        )
        feed_b = Feed(
            name="test",
            url="https://example.com/rss",
            content_type=ContentType.ENGINEERING,
        )
        assert feed_a == feed_b
