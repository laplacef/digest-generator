"""Tests for digest_generator/core/label/io.py: per-feed labeled persistence.

The label io writes a single artifact per feed:
``source-labeled/<feed>.json`` containing URL-keyed label lists.
"""

import json

import pytest

from digest_generator.core.label.io import (
    iter_labeled,
    labeled_path,
    load_labeled,
    save_labeled,
)
from digest_generator.core.types import Label, TopicType


class TestSaveLabeled:
    def test_writes_json_at_canonical_path(self, tmp_path):
        out = save_labeled(
            tmp_path,
            "openai-news",
            urls=["https://example.com/1"],
            labels_per_entry=[[Label(value=TopicType.LLM, confidence=0.92)]],
        )
        assert out == labeled_path(tmp_path, "openai-news")
        assert out.exists()
        assert out.parent == tmp_path / "source-labeled"

    def test_payload_url_keyed(self, tmp_path):
        save_labeled(
            tmp_path,
            "openai-news",
            urls=["https://example.com/1", "https://example.com/2"],
            labels_per_entry=[
                [Label(value=TopicType.LLM, confidence=0.92)],
                [
                    Label(value=TopicType.CYBERSECURITY, confidence=0.88),
                    Label(value=TopicType.VULNERABILITY, confidence=0.71),
                ],
            ],
        )
        data = json.loads(labeled_path(tmp_path, "openai-news").read_text())
        assert isinstance(data, list)
        assert {item["url"] for item in data} == {
            "https://example.com/1",
            "https://example.com/2",
        }
        item1 = next(i for i in data if i["url"] == "https://example.com/1")
        assert item1["labels"] == [{"value": "large-language-models", "confidence": 0.92}]

    def test_raises_on_length_mismatch(self, tmp_path):
        with pytest.raises(ValueError, match="len\\(urls\\)=2"):
            save_labeled(
                tmp_path,
                "openai-news",
                urls=["a", "b"],
                labels_per_entry=[[Label(value=TopicType.LLM, confidence=0.5)]],
            )

    def test_atomic_write_no_lingering_tmp(self, tmp_path):
        save_labeled(
            tmp_path,
            "openai-news",
            urls=["a"],
            labels_per_entry=[[Label(value=TopicType.LLM, confidence=0.5)]],
        )
        tmps = list((tmp_path / "source-labeled").glob("*.tmp"))
        assert tmps == []


class TestLoadLabeled:
    def test_returns_none_on_missing_file(self, tmp_path):
        assert load_labeled(tmp_path, "missing") is None

    def test_round_trip_url_keyed(self, tmp_path):
        save_labeled(
            tmp_path,
            "openai-news",
            urls=["https://example.com/1"],
            labels_per_entry=[[Label(value=TopicType.LLM, confidence=0.92)]],
        )
        result = load_labeled(tmp_path, "openai-news")
        assert result is not None
        assert "https://example.com/1" in result
        assert result["https://example.com/1"][0].value == TopicType.LLM
        assert result["https://example.com/1"][0].confidence == 0.92

    def test_raises_on_non_list_payload(self, tmp_path):
        target = labeled_path(tmp_path, "broken")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({"oops": "not a list"}))
        with pytest.raises(ValueError, match="not a JSON list"):
            load_labeled(tmp_path, "broken")


class TestIterLabeled:
    def test_empty_run_dir_yields_nothing(self, tmp_path):
        assert list(iter_labeled(tmp_path)) == []

    def test_yields_per_feed(self, tmp_path):
        save_labeled(
            tmp_path,
            "feed-a",
            urls=["a"],
            labels_per_entry=[[Label(value=TopicType.LLM, confidence=0.5)]],
        )
        save_labeled(
            tmp_path,
            "feed-b",
            urls=["b"],
            labels_per_entry=[[Label(value=TopicType.SAFETY, confidence=0.5)]],
        )
        feeds = {feed for _, feed, _ in iter_labeled(tmp_path)}
        assert feeds == {"feed-a", "feed-b"}

    def test_yields_rss_source_type_placeholder(self, tmp_path):
        """The sidecar shape doesn't carry source_type per record;
        iter_labeled yields ``"rss"`` as a placeholder until a
        multi-source manifest pattern lands.
        """
        save_labeled(
            tmp_path,
            "feed-a",
            urls=["a"],
            labels_per_entry=[[Label(value=TopicType.LLM, confidence=0.5)]],
        )
        results = list(iter_labeled(tmp_path))
        assert results[0][0] == "rss"

    def test_skips_malformed_entry(self, tmp_path):
        save_labeled(
            tmp_path,
            "good-feed",
            urls=["a"],
            labels_per_entry=[[Label(value=TopicType.LLM, confidence=0.5)]],
        )
        bad = labeled_path(tmp_path, "bad-feed")
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text(json.dumps([{"missing": "url-key"}]))
        feeds = {feed for _, feed, _ in iter_labeled(tmp_path)}
        assert feeds == {"good-feed"}
