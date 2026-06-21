"""Tests for digest_generator/sources/rss/fetcher.py: FeedFetcher logic.

Tests the pure helper methods (_quality_check, _filter_entries) directly,
and tests fetch_entries/_process_candidates with mocked I/O (cloudscraper,
feedparser, asyncio.to_thread, asyncio.sleep).
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from digest_generator.core.types import Filter
from digest_generator.sources.rss.fetcher import FeedFetcher, _truncate_at_word_boundary
from digest_generator.sources.rss.types import Feed

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fetcher():
    """FeedFetcher instance (no scraper state; scraper created per call)."""
    return FeedFetcher()


@pytest.fixture
def sample_feed():
    return Feed(
        name="test-feed",
        url="https://example.com/rss",
        content_type="ai",
    )


def _make_feedparser_entry(
    title="Test Article",
    link="https://example.com/article",
    summary="<p>A test article</p>",
    published_parsed=None,
):
    """Create a fake feedparser entry (SimpleNamespace mimicking FeedParserDict)."""
    if published_parsed is None:
        # Default to 1 day ago
        dt = datetime.now(tz=UTC) - timedelta(days=1)
        published_parsed = dt.utctimetuple()
    return SimpleNamespace(
        title=title,
        link=link,
        summary=summary,
        published_parsed=published_parsed,
        updated_parsed=None,
    )


def _sync_side_effect(fn, *args, **kwargs):
    """Side effect for asyncio.to_thread that runs the function synchronously."""
    return fn(*args, **kwargs)


# =============================================================================
# _quality_check: pure logic
# =============================================================================


class TestQualityCheck:
    def test_accepts_good_content(self, fetcher):
        content = "This is a substantial article about machine learning. " * 20
        assert fetcher._quality_check(content, "Test") is True

    def test_rejects_short_content(self, fetcher):
        assert fetcher._quality_check("Too short.", "Test") is False

    def test_rejects_exactly_at_threshold(self, fetcher):
        """Content shorter than min_content_length (200 chars) is rejected."""
        content = "x" * 199
        assert fetcher._quality_check(content, "Test") is False

    def test_accepts_at_min_length(self, fetcher):
        content = "word " * 50  # 250 chars, no boilerplate
        assert fetcher._quality_check(content, "Test") is True

    def test_rejects_boilerplate_heavy_content(self, fetcher):
        """Content with >5% boilerplate markers is rejected."""
        # Build content where boilerplate markers dominate
        boilerplate = "subscribe cookie newsletter privacy policy " * 10
        assert fetcher._quality_check(boilerplate, "Test") is False

    def test_accepts_low_boilerplate(self, fetcher):
        """Content with minimal boilerplate markers passes."""
        content = "This is a great article about distributed systems. " * 20
        content += "subscribe"  # 1 marker in ~200 words
        assert fetcher._quality_check(content, "Test") is True

    def test_empty_content_rejected(self, fetcher):
        assert fetcher._quality_check("", "Test") is False


# =============================================================================
# _truncate_at_word_boundary: content_head helper
# =============================================================================


class TestTruncateAtWordBoundary:
    def test_returns_empty_for_empty_input(self):
        assert _truncate_at_word_boundary("", 100) == ""

    def test_returns_unchanged_when_shorter_than_limit(self):
        assert _truncate_at_word_boundary("short", 100) == "short"

    def test_cuts_at_word_boundary(self):
        text = "one two three four five six seven"
        out = _truncate_at_word_boundary(text, 15)
        assert out == "one two three"
        assert not out.endswith(" ")

    def test_returns_hard_cut_when_no_whitespace(self):
        long_word = "a" * 100
        assert _truncate_at_word_boundary(long_word, 10) == "a" * 10


# =============================================================================
# _filter_entries: feedparser dict filtering
# =============================================================================


class TestFilterEntries:
    def test_filters_by_date(self, fetcher):
        since = datetime.now(tz=UTC) - timedelta(days=7)
        recent = _make_feedparser_entry(
            title="Recent",
            link="https://example.com/1",
        )
        old_dt = datetime.now(tz=UTC) - timedelta(days=30)
        old = _make_feedparser_entry(
            title="Old",
            link="https://example.com/2",
            published_parsed=old_dt.utctimetuple(),
        )
        parsed_feed = SimpleNamespace(entries=[recent, old])
        result = fetcher._filter_entries(parsed_feed, since=since)
        assert len(result) == 1
        assert result[0].title == "Recent"

    def test_filters_by_until_date(self, fetcher):
        """Entries on or after until are excluded."""
        since = datetime.now(tz=UTC) - timedelta(days=30)
        until = datetime.now(tz=UTC) - timedelta(days=3)
        # 5 days ago, within range
        mid_dt = datetime.now(tz=UTC) - timedelta(days=5)
        mid = _make_feedparser_entry(
            title="Mid",
            link="https://example.com/1",
            published_parsed=mid_dt.utctimetuple(),
        )
        # 1 day ago, after until
        recent = _make_feedparser_entry(
            title="Recent",
            link="https://example.com/2",
        )
        parsed_feed = SimpleNamespace(entries=[mid, recent])
        result = fetcher._filter_entries(parsed_feed, since=since, until=until)
        assert len(result) == 1
        assert result[0].title == "Mid"

    def test_filters_by_date_range(self, fetcher):
        """Only entries within since-until range are included."""
        since = datetime.now(tz=UTC) - timedelta(days=10)
        until = datetime.now(tz=UTC) - timedelta(days=3)
        old_dt = datetime.now(tz=UTC) - timedelta(days=20)
        mid_dt = datetime.now(tz=UTC) - timedelta(days=5)

        old = _make_feedparser_entry(
            title="Old", link="https://example.com/1", published_parsed=old_dt.utctimetuple()
        )
        mid = _make_feedparser_entry(
            title="Mid", link="https://example.com/2", published_parsed=mid_dt.utctimetuple()
        )
        recent = _make_feedparser_entry(title="Recent", link="https://example.com/3")

        parsed_feed = SimpleNamespace(entries=[old, mid, recent])
        result = fetcher._filter_entries(parsed_feed, since=since, until=until)
        assert len(result) == 1
        assert result[0].title == "Mid"

    def test_until_none_means_no_upper_bound(self, fetcher):
        """When until is None, only since is applied."""
        since = datetime.now(tz=UTC) - timedelta(days=7)
        recent = _make_feedparser_entry(title="Recent", link="https://example.com/1")
        parsed_feed = SimpleNamespace(entries=[recent])
        result = fetcher._filter_entries(parsed_feed, since=since, until=None)
        assert len(result) == 1

    def test_deduplicates_by_url(self, fetcher):
        since = datetime.now(tz=UTC) - timedelta(days=7)
        entry1 = _make_feedparser_entry(title="First", link="https://example.com/same")
        entry2 = _make_feedparser_entry(title="Dupe", link="https://example.com/same")
        parsed_feed = SimpleNamespace(entries=[entry1, entry2])
        result = fetcher._filter_entries(parsed_feed, since=since)
        assert len(result) == 1
        assert result[0].title == "First"

    def test_skips_entries_without_title(self, fetcher):
        since = datetime.now(tz=UTC) - timedelta(days=7)
        entry = _make_feedparser_entry(link="https://example.com/1")
        entry.title = None
        parsed_feed = SimpleNamespace(entries=[entry])
        assert fetcher._filter_entries(parsed_feed, since=since) == []

    def test_skips_entries_without_link(self, fetcher):
        since = datetime.now(tz=UTC) - timedelta(days=7)
        entry = _make_feedparser_entry(title="Test")
        entry.link = None
        parsed_feed = SimpleNamespace(entries=[entry])
        assert fetcher._filter_entries(parsed_feed, since=since) == []

    def test_skips_entries_without_date(self, fetcher):
        since = datetime.now(tz=UTC) - timedelta(days=7)
        entry = _make_feedparser_entry()
        entry.published_parsed = None
        entry.updated_parsed = None
        parsed_feed = SimpleNamespace(entries=[entry])
        assert fetcher._filter_entries(parsed_feed, since=since) == []

    def test_uses_updated_parsed_fallback(self, fetcher):
        """If published_parsed is None, updated_parsed should be used."""
        since = datetime.now(tz=UTC) - timedelta(days=7)
        entry = _make_feedparser_entry(link="https://example.com/1")
        entry.published_parsed = None
        entry.updated_parsed = (datetime.now(tz=UTC) - timedelta(days=1)).utctimetuple()
        parsed_feed = SimpleNamespace(entries=[entry])
        result = fetcher._filter_entries(parsed_feed, since=since)
        assert len(result) == 1

    def test_empty_feed(self, fetcher):
        since = datetime.now(tz=UTC) - timedelta(days=7)
        parsed_feed = SimpleNamespace(entries=[])
        assert fetcher._filter_entries(parsed_feed, since=since) == []


# =============================================================================
# fetch_entries: async integration with mocked I/O
# =============================================================================


class TestFetchEntries:
    @pytest.mark.asyncio
    @patch("digest_generator.sources.rss.fetcher.asyncio.sleep", new_callable=AsyncMock)
    @patch("digest_generator.sources.rss.fetcher.asyncio.to_thread", new_callable=AsyncMock)
    @patch("digest_generator.sources.rss.fetcher.feedparser")
    @patch("digest_generator.sources.rss.fetcher.cloudscraper")
    async def test_returns_entries_on_success(
        self, mock_cloudscraper, mock_feedparser, mock_to_thread, mock_sleep, fetcher, sample_feed
    ):
        """Full happy path: fetch feed, filter, process candidates."""
        mock_scraper = MagicMock()
        mock_cloudscraper.create_scraper.return_value = mock_scraper

        # Mock responses
        feed_response = MagicMock()
        feed_response.content = b"<rss>...</rss>"
        article_response = MagicMock()
        article_response.text = (
            "<html><article>Long article content. " + "x" * 300 + "</article></html>"
        )

        # to_thread calls: scraper.get(feed_url), scraper.get(article_url)
        mock_to_thread.side_effect = [feed_response, article_response]

        # Mock feedparser.parse to return one recent entry
        entry = _make_feedparser_entry()
        mock_feedparser.parse.return_value = SimpleNamespace(entries=[entry])

        result = await fetcher.fetch_entries(sample_feed, Filter(days_back=7))
        assert len(result) == 1
        assert result[0].title == "Test Article"
        # origin is the feed slug (feed.name), not the provider.
        assert result[0].origin == sample_feed.name
        assert result[0].source_type == "rss"
        # content_head populated from the extracted article text
        assert result[0].content_head
        assert result[0].content_head.startswith("Long article content")

    @pytest.mark.asyncio
    @patch("digest_generator.sources.rss.fetcher.asyncio.sleep", new_callable=AsyncMock)
    @patch("digest_generator.sources.rss.fetcher.asyncio.to_thread", new_callable=AsyncMock)
    @patch("digest_generator.sources.rss.fetcher.feedparser")
    @patch("digest_generator.sources.rss.fetcher.cloudscraper")
    async def test_falls_back_to_feedparser_on_scraper_failure(
        self, mock_cloudscraper, mock_feedparser, mock_to_thread, mock_sleep, fetcher, sample_feed
    ):
        """If cloudscraper fails for the feed URL, falls back to direct feedparser."""
        mock_cloudscraper.create_scraper.return_value = MagicMock()

        # First to_thread call (scraper.get) fails, second (feedparser.parse) succeeds
        mock_to_thread.side_effect = [
            Exception("Connection error"),
            SimpleNamespace(entries=[]),  # feedparser fallback
        ]

        result = await fetcher.fetch_entries(sample_feed, Filter(days_back=7))
        assert result == []
        # Second to_thread call should be feedparser.parse(feed.url)
        assert mock_to_thread.call_count == 2

    @pytest.mark.asyncio
    @patch("digest_generator.sources.rss.fetcher.asyncio.sleep", new_callable=AsyncMock)
    @patch("digest_generator.sources.rss.fetcher.asyncio.to_thread", new_callable=AsyncMock)
    @patch("digest_generator.sources.rss.fetcher.feedparser")
    @patch("digest_generator.sources.rss.fetcher.cloudscraper")
    async def test_falls_back_to_rss_summary_on_article_fetch_failure(
        self, mock_cloudscraper, mock_feedparser, mock_to_thread, mock_sleep, fetcher, sample_feed
    ):
        """If article page fetch fails, uses RSS summary as content."""
        mock_cloudscraper.create_scraper.return_value = MagicMock()

        feed_response = MagicMock()
        feed_response.content = b"<rss>...</rss>"

        mock_to_thread.side_effect = [
            feed_response,  # Feed fetch succeeds
            Exception("Article fetch failed"),  # Article fetch fails
        ]

        entry = _make_feedparser_entry(summary="x" * 250)
        mock_feedparser.parse.return_value = SimpleNamespace(entries=[entry])

        result = await fetcher.fetch_entries(sample_feed, Filter(days_back=7))
        assert len(result) == 1
        assert result[0].content == "x" * 250

    @pytest.mark.asyncio
    @patch("digest_generator.sources.rss.fetcher.asyncio.sleep", new_callable=AsyncMock)
    @patch("digest_generator.sources.rss.fetcher.asyncio.to_thread", new_callable=AsyncMock)
    @patch("digest_generator.sources.rss.fetcher.feedparser")
    @patch("digest_generator.sources.rss.fetcher.cloudscraper")
    async def test_skips_low_quality_entries(
        self, mock_cloudscraper, mock_feedparser, mock_to_thread, mock_sleep, fetcher, sample_feed
    ):
        """Entries failing quality check are excluded."""
        mock_cloudscraper.create_scraper.return_value = MagicMock()

        feed_response = MagicMock()
        feed_response.content = b"<rss>...</rss>"
        article_response = MagicMock()
        article_response.text = "<html><article>Short</article></html>"

        mock_to_thread.side_effect = [feed_response, article_response]

        entry = _make_feedparser_entry()
        mock_feedparser.parse.return_value = SimpleNamespace(entries=[entry])

        result = await fetcher.fetch_entries(sample_feed, Filter(days_back=7))
        assert result == []

    @pytest.mark.asyncio
    @patch("digest_generator.sources.rss.fetcher.asyncio.sleep", new_callable=AsyncMock)
    @patch("digest_generator.sources.rss.fetcher.asyncio.to_thread", new_callable=AsyncMock)
    @patch("digest_generator.sources.rss.fetcher.feedparser")
    @patch("digest_generator.sources.rss.fetcher.cloudscraper")
    async def test_cleans_html_from_description(
        self, mock_cloudscraper, mock_feedparser, mock_to_thread, mock_sleep, fetcher, sample_feed
    ):
        """HTML tags should be stripped from the RSS description."""
        mock_cloudscraper.create_scraper.return_value = MagicMock()

        feed_response = MagicMock()
        feed_response.content = b"<rss>...</rss>"
        article_response = MagicMock()
        article_response.text = (
            "<html><article>Long article content. " + "x" * 300 + "</article></html>"
        )

        mock_to_thread.side_effect = [feed_response, article_response]

        entry = _make_feedparser_entry(summary="<p>Clean <b>this</b> up</p>")
        mock_feedparser.parse.return_value = SimpleNamespace(entries=[entry])

        result = await fetcher.fetch_entries(sample_feed, Filter(days_back=7))
        assert len(result) == 1
        assert "<" not in result[0].description
        assert "Clean this up" in result[0].description

    @pytest.mark.asyncio
    @patch("digest_generator.sources.rss.fetcher.asyncio.sleep", new_callable=AsyncMock)
    @patch("digest_generator.sources.rss.fetcher.asyncio.to_thread", new_callable=AsyncMock)
    @patch("digest_generator.sources.rss.fetcher.feedparser")
    @patch("digest_generator.sources.rss.fetcher.cloudscraper")
    async def test_respects_filter_limit(
        self, mock_cloudscraper, mock_feedparser, mock_to_thread, mock_sleep, fetcher, sample_feed
    ):
        """Filter.limit should truncate candidates before processing."""
        mock_cloudscraper.create_scraper.return_value = MagicMock()

        feed_response = MagicMock()
        feed_response.content = b"<rss>...</rss>"
        article_response = MagicMock()
        article_response.text = (
            "<html><article>Long article content. " + "x" * 300 + "</article></html>"
        )

        # Two articles available, but limit=1
        entry1 = _make_feedparser_entry(title="First", link="https://example.com/1")
        entry2 = _make_feedparser_entry(title="Second", link="https://example.com/2")
        mock_feedparser.parse.return_value = SimpleNamespace(entries=[entry1, entry2])

        mock_to_thread.side_effect = [feed_response, article_response]

        result = await fetcher.fetch_entries(sample_feed, Filter(days_back=7, limit=1))
        assert len(result) == 1
        assert result[0].title == "First"

    @pytest.mark.asyncio
    @patch("digest_generator.sources.rss.fetcher.asyncio.sleep", new_callable=AsyncMock)
    @patch("digest_generator.sources.rss.fetcher.asyncio.to_thread", new_callable=AsyncMock)
    @patch("digest_generator.sources.rss.fetcher.feedparser")
    @patch("digest_generator.sources.rss.fetcher.cloudscraper")
    async def test_limit_none_returns_all(
        self, mock_cloudscraper, mock_feedparser, mock_to_thread, mock_sleep, fetcher, sample_feed
    ):
        """Filter with limit=None should return all matching entries."""
        mock_cloudscraper.create_scraper.return_value = MagicMock()

        feed_response = MagicMock()
        feed_response.content = b"<rss>...</rss>"
        article_response = MagicMock()
        article_response.text = (
            "<html><article>Long article content. " + "x" * 300 + "</article></html>"
        )

        entry1 = _make_feedparser_entry(title="First", link="https://example.com/1")
        entry2 = _make_feedparser_entry(title="Second", link="https://example.com/2")
        mock_feedparser.parse.return_value = SimpleNamespace(entries=[entry1, entry2])

        mock_to_thread.side_effect = [feed_response, article_response, article_response]

        result = await fetcher.fetch_entries(sample_feed, Filter(days_back=7, limit=None))
        assert len(result) == 2

    @pytest.mark.asyncio
    @patch("digest_generator.sources.rss.fetcher.asyncio.sleep", new_callable=AsyncMock)
    @patch("digest_generator.sources.rss.fetcher.asyncio.to_thread", new_callable=AsyncMock)
    @patch("digest_generator.sources.rss.fetcher.feedparser")
    @patch("digest_generator.sources.rss.fetcher.cloudscraper")
    async def test_respects_filter_since(
        self, mock_cloudscraper, mock_feedparser, mock_to_thread, mock_sleep, fetcher, sample_feed
    ):
        """Filter with explicit since should use it instead of computing from days_back."""
        mock_cloudscraper.create_scraper.return_value = MagicMock()

        feed_response = MagicMock()
        feed_response.content = b"<rss>...</rss>"
        article_response = MagicMock()
        article_response.text = (
            "<html><article>Long article content. " + "x" * 300 + "</article></html>"
        )

        # Entry from 5 days ago
        dt_5d = datetime.now(tz=UTC) - timedelta(days=5)
        entry = _make_feedparser_entry(
            title="Mid-range",
            link="https://example.com/1",
            published_parsed=dt_5d.utctimetuple(),
        )
        mock_feedparser.parse.return_value = SimpleNamespace(entries=[entry])

        mock_to_thread.side_effect = [feed_response, article_response]

        # since=3 days ago should exclude the 5-day-old entry
        since = datetime.now(tz=UTC) - timedelta(days=3)
        result = await fetcher.fetch_entries(sample_feed, Filter(since=since))
        assert result == []
