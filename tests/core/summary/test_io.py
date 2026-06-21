"""Tests for digest_generator/core/summary/io.py: per-feed summary persistence."""

import json
from datetime import UTC, datetime

import pytest

from digest_generator.core.summary.io import (
    iter_summarized,
    load_summarized,
    save_summarized,
    summarized_path,
)
from digest_generator.core.types import ContentType, Entry, Summary


@pytest.fixture
def now():
    return datetime.now(tz=UTC)


def _summary(
    now,
    *,
    title="GPT-5",
    url="https://example.com/1",
    content_head="head",
    source_type="rss",
    content_type=None,
    fetched_at=None,
):
    entry = Entry(
        title=title,
        url=url,
        origin="openai-news",
        published=now,
        description="d",
        content="c",
        content_head=content_head,
        source_type=source_type,
        content_type=content_type,
        fetched_at=fetched_at,
    )
    return Summary(entry=entry, summary="extracted fact", length=14, topics=[])


class TestSaveSummarized:
    def test_writes_json_at_canonical_path(self, tmp_path, now):
        out = save_summarized(tmp_path, "openai-news", [_summary(now)])
        assert out == summarized_path(tmp_path, "openai-news")
        assert out.exists()
        assert out.parent == tmp_path / "source-summarized"

    def test_payload_omits_topics(self, tmp_path, now):
        """Summarizer-stage schema omits ``topics``; they live in
        ``source-labeled/`` and join back via ``api._load_digest_input``.
        """
        save_summarized(tmp_path, "openai-news", [_summary(now)])
        article = json.loads(summarized_path(tmp_path, "openai-news").read_text())[0]
        assert "topics" not in article

    def test_carries_summarizer_fields(self, tmp_path, now):
        save_summarized(tmp_path, "openai-news", [_summary(now)])
        article = json.loads(summarized_path(tmp_path, "openai-news").read_text())[0]
        for key in (
            "title",
            "url",
            "origin",
            "source_type",
            "published",
            "description",
            "summary",
            "summary_length",
        ):
            assert key in article
        assert article["origin"] == "openai-news"
        assert article["summary"] == "extracted fact"
        assert article["summary_length"] == 14
        assert article["source_type"] == "rss"
        # The "feed" and "source" keys do not exist in the schema.
        assert "feed" not in article
        assert "source" not in article

    def test_carries_optional_identity_fields(self, tmp_path, now):
        """``content_type`` and ``fetched_at`` propagate from Entry."""
        save_summarized(
            tmp_path,
            "openai-news",
            [_summary(now, content_type=ContentType.AI, fetched_at=now)],
        )
        article = json.loads(summarized_path(tmp_path, "openai-news").read_text())[0]
        assert article["content_type"] == "ai"
        assert article["fetched_at"] == now.isoformat()

    def test_omits_optional_identity_fields_when_unset(self, tmp_path, now):
        save_summarized(tmp_path, "openai-news", [_summary(now)])
        article = json.loads(summarized_path(tmp_path, "openai-news").read_text())[0]
        assert "content_type" not in article
        assert "fetched_at" not in article

    def test_omits_content_head_when_empty(self, tmp_path, now):
        save_summarized(tmp_path, "openai-news", [_summary(now, content_head="")])
        article = json.loads(summarized_path(tmp_path, "openai-news").read_text())[0]
        assert "content_head" not in article

    def test_includes_content_head_when_present(self, tmp_path, now):
        save_summarized(tmp_path, "openai-news", [_summary(now, content_head="excerpt")])
        article = json.loads(summarized_path(tmp_path, "openai-news").read_text())[0]
        assert article["content_head"] == "excerpt"

    def test_atomic_write_no_lingering_tmp(self, tmp_path, now):
        save_summarized(tmp_path, "openai-news", [_summary(now)])
        tmps = list((tmp_path / "source-summarized").glob("*.tmp"))
        assert tmps == []


class TestLoadSummarized:
    def test_returns_none_on_missing_file(self, tmp_path):
        assert load_summarized(tmp_path, "missing") is None

    def test_round_trip(self, tmp_path, now):
        save_summarized(tmp_path, "openai-news", [_summary(now)])
        data = load_summarized(tmp_path, "openai-news")
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["title"] == "GPT-5"

    def test_raises_on_non_list_payload(self, tmp_path):
        target = summarized_path(tmp_path, "broken")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"oops": "not a list"}))
        with pytest.raises(ValueError, match="not a JSON list"):
            load_summarized(tmp_path, "broken")


class TestIterSummarized:
    def test_empty_run_dir_yields_nothing(self, tmp_path):
        assert list(iter_summarized(tmp_path)) == []

    def test_yields_per_feed(self, tmp_path, now):
        save_summarized(tmp_path, "feed-a", [_summary(now)])
        save_summarized(tmp_path, "feed-b", [_summary(now)])
        results = list(iter_summarized(tmp_path))
        feeds = {feed for _, feed, _ in results}
        assert feeds == {"feed-a", "feed-b"}

    def test_yields_multiple_source_types(self, tmp_path, now):
        """``source_type`` comes from each record's data, so different
        feeds can carry different ``source_type`` values without
        per-source-type subdirs."""
        save_summarized(tmp_path, "openai-news", [_summary(now)])
        save_summarized(tmp_path, "topstories", [_summary(now, source_type="hn")])
        sources = {source for source, _, _ in iter_summarized(tmp_path)}
        assert sources == {"rss", "hn"}

    def test_defaults_source_type_to_rss_when_missing(self, tmp_path):
        """Records that lack ``source_type`` yield ``"rss"`` as a default."""
        path = summarized_path(tmp_path, "feed-a")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                [
                    {
                        "title": "t",
                        "url": "u",
                        "origin": "feed-a",
                        "published": "2026-05-11T00:00:00+00:00",
                        "description": "d",
                        "summary": "s",
                        "summary_length": 1,
                    }
                ]
            )
        )
        results = list(iter_summarized(tmp_path))
        assert results[0][0] == "rss"

    def test_skips_malformed_json(self, tmp_path, now):
        save_summarized(tmp_path, "good-feed", [_summary(now)])
        bad = summarized_path(tmp_path, "bad-feed")
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("{{{ not json")
        feeds = {feed for _, feed, _ in iter_summarized(tmp_path)}
        assert feeds == {"good-feed"}
