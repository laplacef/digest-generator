"""RSS feed fetching, article extraction, and content quality filtering.

Uses cloudscraper for Cloudflare bypass with fallback to direct feedparser.
Extracts full article text via CSS selectors, falling back to RSS summary.
"""

import asyncio
import calendar
from datetime import UTC, datetime, timedelta

import cloudscraper
import feedparser
from bs4 import BeautifulSoup

from digest_generator.core.types import Entry, Filter
from digest_generator.shared.logging import current_span, log_stage, logger
from digest_generator.shared.settings import settings
from digest_generator.sources.rss.types import BoilerplateMarker, Feed, SelectorType


class FeedFetcher:
    """Fetches and parses articles from RSS feeds, extracting full page content.

    Uses cloudscraper to bypass Cloudflare protection on both feed URLs and
    article pages. Falls back to RSS summary when full extraction fails.

    A new cloudscraper session is created per ``fetch_entries()`` call for
    thread safety when running concurrently via ``asyncio.to_thread()``.
    """

    async def fetch_entries(self, feed: Feed, filter: Filter) -> list[Entry]:
        """Fetch and process recent entries from an RSS feed.

        Fetches the feed (cloudscraper with feedparser fallback), filters entries
        by date and uniqueness, then extracts full article content for each.

        Args:
            feed: RSS feed to fetch from.
            filter: Controls date range and entry limits.

        Returns:
            Processed entries with full article content and cleaned descriptions.
        """
        with log_stage("fetcher", feed=feed.name) as span:
            scraper = cloudscraper.create_scraper()

            # Fetch the RSS feed with cloudscraper
            try:
                response = await asyncio.to_thread(
                    scraper.get, feed.url, timeout=settings.fetch_timeout
                )
                response.raise_for_status()
                parsed_feed = feedparser.parse(response.content)
            except Exception as e:
                logger.warning(
                    "Cloudscraper failed for {}, falling back to feedparser: {}", feed.url, e
                )
                # Fallback to direct feedparser if cloudscraper fails
                parsed_feed = await asyncio.to_thread(feedparser.parse, feed.url)

            since = filter.since or (datetime.now(UTC) - timedelta(days=filter.days_back))
            candidate_entries: list[feedparser.FeedParserDict] = self._filter_entries(
                parsed_feed, since=since, until=filter.until
            )

            if filter.limit is not None:
                candidate_entries = candidate_entries[: filter.limit]

            processed_entries: list[Entry] = await self._process_candidates(
                feed=feed, candidate_entries=candidate_entries, scraper=scraper
            )

            span.set(
                candidates=len(candidate_entries),
                entries=len(processed_entries),
            )
            return processed_entries

    async def _process_candidates(
        self,
        feed: Feed,
        candidate_entries: list[feedparser.FeedParserDict],
        scraper: cloudscraper.CloudScraper,
    ) -> list[Entry]:
        """Fetch full article content for each candidate and build ``Entry`` objects.

        For each candidate: scrapes the article page, extracts text via CSS
        selectors (falling back to RSS summary), runs a quality check, and
        cleans the RSS description of HTML tags. Sleeps between fetches.

        Args:
            feed: The parent feed (provides provider and metadata).
            candidate_entries: Pre-filtered feedparser entries to process.
            scraper: Cloudscraper session for HTTP requests.

        Returns:
            Entries that passed the quality check, with full content extracted.
        """
        processed_entries: list[Entry] = []
        for index, candidate in enumerate(candidate_entries):
            logger.debug("Parsing entry {} {}", index, candidate.title)

            # Parse the published date for this entry
            dt_struct = getattr(candidate, "published_parsed", None) or getattr(
                candidate, "updated_parsed", None
            )
            if dt_struct is None:
                continue
            published_datetime = datetime.fromtimestamp(calendar.timegm(dt_struct), tz=UTC)

            span = current_span()

            # Fetch the article content
            extraction_mode = "full_content"
            try:
                response = await asyncio.to_thread(
                    scraper.get, candidate.link, timeout=settings.fetch_timeout
                )
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # Extract article content by trying common article selectors
                article_content = ""
                for selector in SelectorType:
                    elem = soup.select_one(selector)
                    if elem:
                        article_content = elem.get_text(" ", strip=True)
                        break

                # Fallback to RSS summary if article extraction fails
                if not article_content:
                    article_content = getattr(candidate, "summary", "")
                    extraction_mode = "rss_fallback_no_selector"
                    logger.warning("No article content extracted, using RSS summary")

            except Exception as e:
                logger.warning("Failed to fetch article, using RSS summary: {}", e)
                article_content = getattr(candidate, "summary", "")
                extraction_mode = "rss_fallback_http_error"

            # Skip entries with low-quality content
            if not self._quality_check(content=article_content, title=candidate.title):
                if span is not None:
                    span.add(quality_skipped=1)
                await asyncio.sleep(settings.fetch_rate_limit)
                continue

            if span is not None:
                span.add(**{extraction_mode: 1})

            # Clean the description by removing HTML tags
            raw_description = getattr(candidate, "summary", "")
            description = BeautifulSoup(raw_description, "html.parser").get_text(" ", strip=True)

            entry = Entry(
                title=candidate.title,
                url=candidate.link,
                origin=feed.name,
                published=published_datetime,
                description=description,
                content=article_content,
                content_head=_truncate_at_word_boundary(
                    article_content, settings.content_head_max_chars
                ),
                source_type="rss",
                content_type=feed.content_type,
                fetched_at=datetime.now(UTC),
            )
            processed_entries.append(entry)

            # Sleep to avoid rate limiting
            await asyncio.sleep(settings.fetch_rate_limit)

        return processed_entries

    def _filter_entries(
        self,
        parsed_feed: feedparser.FeedParserDict,
        *,
        since: datetime,
        until: datetime | None = None,
    ) -> list[feedparser.FeedParserDict]:
        """Filter feed entries by date range, deduplication, and required fields.

        Skips entries that are missing title/link/date, have duplicate URLs,
        or fall outside the date range.

        Args:
            parsed_feed: Parsed RSS feed from feedparser.
            since: Exclusive lower bound; entries on or before this date are excluded.
            until: Exclusive upper bound; entries on or after this date are excluded.
                ``None`` means no upper bound.

        Returns:
            Deduplicated entries published within the date range.
        """
        seen: set[str] = set()
        candidate_entries = []
        for parsed_entry in parsed_feed.entries:
            title = getattr(parsed_entry, "title", None)
            link = getattr(parsed_entry, "link", None)
            dt_struct = getattr(parsed_entry, "published_parsed", None) or getattr(
                parsed_entry, "updated_parsed", None
            )
            if not title or not link or not dt_struct:
                continue
            if link in seen:
                continue
            seen.add(link)

            published_datetime = datetime.fromtimestamp(calendar.timegm(dt_struct), tz=UTC)
            if published_datetime <= since:
                continue
            if until is not None and published_datetime >= until:
                continue

            candidate_entries.append(parsed_entry)

        return candidate_entries

    def _quality_check(self, content: str, title: str) -> bool:
        """Check whether extracted content is substantial enough to summarize.

        Rejects content shorter than 200 characters (video pages, paywalls) or
        with a boilerplate marker ratio exceeding 5% (cookie banners, nav leakage).

        Args:
            content: Extracted article text.
            title: Article title (used for debug logging).

        Returns:
            ``True`` if the content passes quality checks.
        """
        boilerplate_markers = [marker.value for marker in BoilerplateMarker]

        if len(content) < settings.min_content_length:
            logger.debug("Skipping '{}': content too short ({} chars)", title, len(content))
            return False

        content_lower = content.lower()
        words = content_lower.split()
        if words:
            hits = sum(content_lower.count(m) for m in boilerplate_markers)
            if hits / len(words) > settings.max_boilerplate_ratio:
                logger.debug("Skipping '{}': too much boilerplate", title)
                return False

        return True


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """Return ``text`` trimmed to ``max_chars``, cut at the last word boundary.

    Returns an empty string when ``text`` is empty; returns ``text`` unchanged
    when it's already short enough or has no whitespace to split on.
    """
    if not text or len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space <= 0:
        return truncated
    return truncated[:last_space]
