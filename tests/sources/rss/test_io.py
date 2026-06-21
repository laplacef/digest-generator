"""Tests for digest_generator/sources/rss/io.py: per-feed Entry persistence."""

import json
from datetime import UTC, datetime

import pytest

from digest_generator.core.types import ContentType, Entry
from digest_generator.sources.rss.io import (
    fetched_path,
    iter_fetched,
    load_entries,
    save_entries,
)


@pytest.fixture
def now():
    return datetime.now(tz=UTC)


def _entry(
    now,
    *,
    title="GPT-5",
    url="https://example.com/1",
    content_head="head",
    source_type="rss",
    content_type=ContentType.AI,
    fetched_at=None,
):
    return Entry(
        title=title,
        url=url,
        origin="openai-news",
        published=now,
        description="d",
        content="full content",
        content_head=content_head,
        source_type=source_type,
        content_type=content_type,
        fetched_at=fetched_at if fetched_at is not None else now,
    )


class TestSaveEntries:
    def test_writes_json_at_canonical_path(self, tmp_path, now):
        out = save_entries(tmp_path, "openai-news", [_entry(now)])
        assert out == fetched_path(tmp_path, "openai-news")
        assert out.exists()
        assert out.parent == tmp_path / "source-fetched"

    def test_payload_schema(self, tmp_path, now):
        save_entries(tmp_path, "openai-news", [_entry(now)])
        article = json.loads(fetched_path(tmp_path, "openai-news").read_text())[0]
        for key in ("title", "url", "origin", "published", "description", "content"):
            assert key in article
        assert article["title"] == "GPT-5"
        assert article["origin"] == "openai-news"
        assert "source" not in article

    def test_omits_content_head_when_empty(self, tmp_path, now):
        save_entries(tmp_path, "openai-news", [_entry(now, content_head="")])
        article = json.loads(fetched_path(tmp_path, "openai-news").read_text())[0]
        assert "content_head" not in article

    def test_includes_content_head_when_present(self, tmp_path, now):
        save_entries(tmp_path, "openai-news", [_entry(now, content_head="excerpt")])
        article = json.loads(fetched_path(tmp_path, "openai-news").read_text())[0]
        assert article["content_head"] == "excerpt"

    def test_writes_new_identity_fields(self, tmp_path, now):
        """source_type / content_type / fetched_at are written when populated."""
        save_entries(tmp_path, "openai-news", [_entry(now)])
        article = json.loads(fetched_path(tmp_path, "openai-news").read_text())[0]
        assert article["source_type"] == "rss"
        assert article["content_type"] == "ai"
        assert article["fetched_at"] == now.isoformat()

    def test_atomic_write_no_lingering_tmp(self, tmp_path, now):
        save_entries(tmp_path, "openai-news", [_entry(now)])
        tmps = list((tmp_path / "source-fetched").glob("*.tmp"))
        assert tmps == []


class TestLoadEntries:
    def test_returns_none_on_missing_file(self, tmp_path):
        assert load_entries(tmp_path, "missing") is None

    def test_round_trip_returns_entry_dataclasses(self, tmp_path, now):
        save_entries(tmp_path, "openai-news", [_entry(now)])
        entries = load_entries(tmp_path, "openai-news")
        assert entries is not None
        assert len(entries) == 1
        assert isinstance(entries[0], Entry)
        assert entries[0].title == "GPT-5"
        assert entries[0].published == now

    def test_round_trip_preserves_content_head_default(self, tmp_path, now):
        save_entries(tmp_path, "openai-news", [_entry(now, content_head="")])
        entries = load_entries(tmp_path, "openai-news")
        assert entries is not None
        assert entries[0].content_head == ""

    def test_round_trip_preserves_new_identity_fields(self, tmp_path, now):
        """source_type / content_type / fetched_at survive the JSON round-trip."""
        save_entries(tmp_path, "openai-news", [_entry(now)])
        entries = load_entries(tmp_path, "openai-news")
        assert entries is not None
        assert entries[0].source_type == "rss"
        assert entries[0].content_type is ContentType.AI
        assert entries[0].fetched_at == now

    def test_raises_on_non_list_payload(self, tmp_path):
        target = fetched_path(tmp_path, "broken")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"oops": "not a list"}))
        with pytest.raises(ValueError, match="not a JSON list"):
            load_entries(tmp_path, "broken")


class TestIterFetched:
    def test_empty_run_dir_yields_nothing(self, tmp_path):
        assert list(iter_fetched(tmp_path)) == []

    def test_yields_per_content_type_per_feed(self, tmp_path, now):
        save_entries(
            tmp_path,
            "feed-a",
            [_entry(now, content_type=ContentType.AI)],
        )
        save_entries(
            tmp_path,
            "feed-b",
            [_entry(now, content_type=ContentType.SECURITY)],
        )
        results = list(iter_fetched(tmp_path))
        types = {ct for ct, _, _ in results}
        feeds = {feed for _, feed, _ in results}
        assert types == {ContentType.AI, ContentType.SECURITY}
        assert feeds == {"feed-a", "feed-b"}

    def test_returns_entry_dataclasses(self, tmp_path, now):
        save_entries(tmp_path, "openai-news", [_entry(now)])
        results = list(iter_fetched(tmp_path))
        _, _, entries = results[0]
        assert all(isinstance(e, Entry) for e in entries)

    def test_skips_malformed_json(self, tmp_path, now):
        save_entries(tmp_path, "good-feed", [_entry(now)])
        bad = fetched_path(tmp_path, "bad-feed")
        bad.write_text("{{{ not json")
        feeds = {feed for _, feed, _ in iter_fetched(tmp_path)}
        assert feeds == {"good-feed"}

    def test_skips_malformed_entry(self, tmp_path, now):
        save_entries(tmp_path, "good-feed", [_entry(now)])
        bad = fetched_path(tmp_path, "bad-feed")
        bad.write_text(json.dumps([{"missing": "required-fields"}]))
        feeds = {feed for _, feed, _ in iter_fetched(tmp_path)}
        assert feeds == {"good-feed"}

    def test_skips_entries_without_content_type(self, tmp_path, now):
        """iter_fetched skips files whose entries lack a populated
        ``content_type``, which is required because content_type is sourced
        from each record rather than encoded in the directory path."""
        path = fetched_path(tmp_path, "broken")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                [
                    {
                        "title": "t",
                        "url": "u",
                        "origin": "s",
                        "source_type": "rss",
                        "published": now.isoformat(),
                        "description": "d",
                        "content": "c",
                    }
                ]
            )
        )
        results = list(iter_fetched(tmp_path))
        assert results == []
