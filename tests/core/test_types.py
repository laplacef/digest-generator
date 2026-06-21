"""Tests for digest_generator/core/types.py: cross-stage domain vocabulary.

`Entry`, `Summary`, `ContentType`, `Filter`, `Label`, `TopicType` are
the types every pipeline stage shares.
"""

import json
from datetime import UTC, datetime, timedelta

import pytest

from digest_generator.core.types import ContentType, Entry, Filter, Label, TopicType

# =============================================================================
# Enum tests
# =============================================================================


class TestTopicType:
    """29 zero-shot classification labels emitted by the topic stage."""

    def test_values_are_strings(self):
        """StrEnum values should be usable as plain strings."""
        assert TopicType.LLM == "large-language-models"
        assert TopicType.AGENTS == "agents"
        assert TopicType.CYBERSECURITY == "cybersecurity"

    def test_count(self):
        """Guard against accidental additions/deletions."""
        assert len(TopicType) == 29

    def test_json_serializable(self):
        """StrEnum values should work directly in JSON without .value."""
        data = {"topic": TopicType.LLM}
        assert '"large-language-models"' in json.dumps(data)


class TestContentType:
    """Five content types used as the digest section grouping."""

    def test_values(self):
        assert ContentType.AI == "ai"
        assert ContentType.ENGINEERING == "engineering"
        assert ContentType.INFRASTRUCTURE == "infrastructure"
        assert ContentType.SECURITY == "security"
        assert ContentType.BUSINESS == "business"

    def test_count(self):
        assert len(ContentType) == 5

    def test_display_name_ai(self):
        """AI has a special display name."""
        assert ContentType.AI.display_name == "AI & Machine Learning"

    def test_display_name_others(self):
        """Non-AI types capitalize the enum name."""
        assert ContentType.ENGINEERING.display_name == "Engineering"
        assert ContentType.SECURITY.display_name == "Security"
        assert ContentType.BUSINESS.display_name == "Business"
        assert ContentType.INFRASTRUCTURE.display_name == "Infrastructure"

    def test_enum_order_defines_digest_sections(self):
        """ContentType iteration order is the digest section order."""
        order = list(ContentType)
        assert order == [
            ContentType.AI,
            ContentType.ENGINEERING,
            ContentType.INFRASTRUCTURE,
            ContentType.SECURITY,
            ContentType.BUSINESS,
        ]


# =============================================================================
# Filter dataclass + Filter.resolve
# =============================================================================


class TestFilter:
    """Construction, defaults, equality."""

    def test_default_days_back(self):
        """Filter() with no args should default to 7 days."""
        f = Filter()
        assert f.days_back == 7

    def test_custom_days_back(self):
        f = Filter(days_back=14)
        assert f.days_back == 14

    def test_new_fields_default_to_none(self):
        f = Filter()
        assert f.since is None
        assert f.until is None
        assert f.limit is None

    def test_custom_since_until(self):
        since = datetime(2026, 3, 1, tzinfo=UTC)
        until = datetime(2026, 3, 15, tzinfo=UTC)
        f = Filter(since=since, until=until)
        assert f.since == since
        assert f.until == until

    def test_custom_limit(self):
        f = Filter(limit=5)
        assert f.limit == 5

    def test_equality(self):
        assert Filter(days_back=7) == Filter(days_back=7)
        assert Filter(days_back=7) != Filter(days_back=14)

        since = datetime(2026, 3, 1, tzinfo=UTC)
        assert Filter(since=since) == Filter(since=since)
        assert Filter(limit=5) != Filter(limit=10)


class TestFilterResolve:
    """Resolving relative (days_back) to absolute (since/until) timestamps."""

    def test_resolve_from_days_back(self):
        """Without since/until, resolve computes both from days_back."""
        before = datetime.now(UTC)
        f = Filter.resolve(days_back=7)
        after = datetime.now(UTC)

        assert f.days_back == 7
        assert f.since is not None
        assert f.until is not None
        expected_since = before - timedelta(days=7)
        assert abs((f.since - expected_since).total_seconds()) < 1
        assert abs((f.until - after).total_seconds()) < 1

    def test_resolve_with_explicit_since(self):
        """With only since, until defaults to now."""
        since = datetime(2026, 3, 1, tzinfo=UTC)
        f = Filter.resolve(since=since)
        after = datetime.now(UTC)

        assert f.since == since
        assert f.until is not None
        assert abs((f.until - after).total_seconds()) < 1

    def test_resolve_with_explicit_until(self):
        """With only until, since computed from days_back."""
        until = datetime.now(UTC) + timedelta(days=1)
        before = datetime.now(UTC)
        f = Filter.resolve(until=until, days_back=7)

        assert f.until == until
        assert f.since is not None
        expected_since = before - timedelta(days=7)
        assert abs((f.since - expected_since).total_seconds()) < 1

    def test_resolve_with_both(self):
        """With both since and until, use them directly."""
        since = datetime(2026, 3, 1, tzinfo=UTC)
        until = datetime(2026, 3, 15, tzinfo=UTC)
        f = Filter.resolve(since=since, until=until)

        assert f.since == since
        assert f.until == until

    def test_resolve_since_after_until_raises(self):
        since = datetime(2026, 3, 15, tzinfo=UTC)
        until = datetime(2026, 3, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="must be before"):
            Filter.resolve(since=since, until=until)

    def test_resolve_since_equals_until_raises(self):
        dt = datetime(2026, 3, 15, tzinfo=UTC)
        with pytest.raises(ValueError, match="must be before"):
            Filter.resolve(since=dt, until=dt)

    def test_resolve_limit_zero_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            Filter.resolve(limit=0)

    def test_resolve_negative_limit_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            Filter.resolve(limit=-1)

    def test_resolve_positive_limit(self):
        f = Filter.resolve(limit=5)
        assert f.limit == 5


# =============================================================================
# Label
# =============================================================================


class TestLabel:
    """Generic label dataclass with `value` and `confidence`. Used by every label stage."""

    def test_construction(self):
        label = Label(value=TopicType.LLM, confidence=0.95)
        assert label.value == TopicType.LLM
        assert label.confidence == 0.95

    def test_stores_enum(self):
        """Label.value should preserve the enum identity (StrEnum) it was passed."""
        label = Label(value=TopicType.SAFETY, confidence=0.8)
        assert isinstance(label.value, TopicType)


# =============================================================================
# Entry
# =============================================================================


class TestEntry:
    """Cross-stage article record. Identity (source/source_type/content_type/
    fetched_at) plus payload (title/url/published/description/content/
    content_head). The identity-metadata fields drive the cache layout.
    """

    def _required_kwargs(self, now):
        return {
            "title": "t",
            "url": "https://example.com/x",
            "origin": "ai-magazine",
            "published": now,
            "description": "d",
            "content": "c",
        }

    def test_minimal_construction(self):
        """Only the seven core fields are required; the identity
        fields default safely so existing constructor sites keep working."""
        now = datetime(2026, 5, 11, tzinfo=UTC)
        entry = Entry(**self._required_kwargs(now))
        assert entry.title == "t"
        assert entry.origin == "ai-magazine"

    def test_new_identity_field_defaults(self):
        """source_type defaults to "rss"; content_type and fetched_at default
        to None until the fetcher (or test fixture) populates them."""
        now = datetime(2026, 5, 11, tzinfo=UTC)
        entry = Entry(**self._required_kwargs(now))
        assert entry.source_type == "rss"
        assert entry.content_type is None
        assert entry.fetched_at is None
        assert entry.content_head == ""

    def test_explicit_identity_fields(self):
        """Identity fields round-trip through the constructor."""
        now = datetime(2026, 5, 11, tzinfo=UTC)
        fetched = datetime(2026, 5, 11, 12, 0, tzinfo=UTC)
        entry = Entry(
            **self._required_kwargs(now),
            source_type="rss",
            content_type=ContentType.AI,
            fetched_at=fetched,
        )
        assert entry.source_type == "rss"
        assert entry.content_type is ContentType.AI
        assert entry.fetched_at == fetched
