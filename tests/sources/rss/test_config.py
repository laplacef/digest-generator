"""Tests for digest_generator/sources/rss/config.py: feeds.yaml discovery + loading.

Feeds come from a user-supplied feeds.yaml. These tests cover the happy
path, every validation error, and the discovery search order.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from digest_generator.core.types import ContentType
from digest_generator.shared.settings import settings
from digest_generator.sources.rss import config as config_module
from digest_generator.sources.rss.config import (
    FeedsConfigError,
    candidate_paths,
    discover_feeds_file,
    load_configured_feeds,
    load_feeds,
)

_VALID = """\
feeds:
  - name: openai-news
    url: https://openai.com/news/rss.xml
    category: ai
  - name: krebs
    url: https://krebsonsecurity.com/feed/
    category: security
"""


def _write(tmp_path, text, name="feeds.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def isolated_discovery(tmp_path, monkeypatch):
    """Cut discovery off from the host environment.

    Points the user-level config dir and cwd at empty temp locations and
    clears the settings fallbacks so a real
    ``~/.config/digest-generator/feeds.yaml`` on the host can't make
    absence tests pass by accident.
    """
    monkeypatch.setattr(
        config_module, "_USER_CONFIG_DIR", tmp_path / "no-home" / "digest-generator"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "feeds_file", None)
    monkeypatch.setattr(settings, "digest_config", None)


class TestLoadFeeds:
    """load_feeds: parse + validate a feeds.yaml into list[Feed]."""

    def test_valid_file(self, tmp_path):
        feeds = load_feeds(_write(tmp_path, _VALID))
        assert [f.name for f in feeds] == ["openai-news", "krebs"]
        assert feeds[0].content_type == ContentType.AI
        assert feeds[1].content_type == ContentType.SECURITY
        assert feeds[0].url == "https://openai.com/news/rss.xml"

    def test_preserves_document_order(self, tmp_path):
        text = "feeds:\n" + "".join(
            f"  - name: f{i}\n    url: https://e/{i}\n    category: ai\n" for i in range(5)
        )
        feeds = load_feeds(_write(tmp_path, text))
        assert [f.name for f in feeds] == [f"f{i}" for i in range(5)]

    def test_unknown_top_level_keys_ignored(self, tmp_path):
        text = "version: 2\ncategories: [a, b]\n" + _VALID
        feeds = load_feeds(_write(tmp_path, text))
        assert len(feeds) == 2

    def test_empty_file(self, tmp_path):
        with pytest.raises(FeedsConfigError, match="empty"):
            load_feeds(_write(tmp_path, ""))

    def test_empty_feeds_list(self, tmp_path):
        with pytest.raises(FeedsConfigError):
            load_feeds(_write(tmp_path, "feeds: []\n"))

    def test_unknown_category(self, tmp_path):
        text = "feeds:\n  - name: x\n    url: https://e\n    category: politics\n"
        with pytest.raises(FeedsConfigError) as exc:
            load_feeds(_write(tmp_path, text))
        assert "category" in str(exc.value)

    def test_missing_required_field(self, tmp_path):
        text = "feeds:\n  - name: x\n    category: ai\n"  # no url
        with pytest.raises(FeedsConfigError):
            load_feeds(_write(tmp_path, text))

    def test_extra_feed_field_rejected(self, tmp_path):
        text = "feeds:\n  - name: x\n    url: https://e\n    category: ai\n    provider: openai\n"
        with pytest.raises(FeedsConfigError):
            load_feeds(_write(tmp_path, text))

    def test_duplicate_names(self, tmp_path):
        text = (
            "feeds:\n"
            "  - name: dup\n    url: https://a\n    category: ai\n"
            "  - name: dup\n    url: https://b\n    category: security\n"
        )
        with pytest.raises(FeedsConfigError, match="Duplicate"):
            load_feeds(_write(tmp_path, text))

    def test_malformed_yaml(self, tmp_path):
        with pytest.raises(FeedsConfigError, match="Invalid YAML"):
            load_feeds(_write(tmp_path, "feeds: [unterminated\n"))


class TestDiscovery:
    """candidate_paths / discover_feeds_file: the search order."""

    def test_explicit_file_first(self, tmp_path):
        paths = candidate_paths(feeds_file="/explicit/feeds.yaml", config_dir="/cfg")
        assert paths[0].name == "feeds.yaml"
        assert str(paths[0]) == "/explicit/feeds.yaml"
        # config dir comes next, then project-local, then user-level
        assert paths[1] == Path("/cfg/feeds.yaml")

    def test_user_level_is_last(self, tmp_path):
        paths = candidate_paths()
        assert paths[-1].parts[-3:] == (".config", "digest-generator", "feeds.yaml")

    def test_discover_returns_first_existing(self, tmp_path):
        existing = _write(tmp_path, _VALID)
        found = discover_feeds_file(feeds_file=str(existing))
        assert found == existing

    def test_discover_returns_none_when_absent(self, tmp_path, isolated_discovery):
        found = discover_feeds_file(feeds_file=str(tmp_path / "nope.yaml"))
        assert found is None


class TestLoadConfiguredFeeds:
    """load_configured_feeds: discovery + load + clear not-found error."""

    def test_loads_via_explicit_file(self, tmp_path):
        path = _write(tmp_path, _VALID)
        feeds = load_configured_feeds(feeds_file=str(path))
        assert len(feeds) == 2

    def test_loads_via_config_dir(self, tmp_path):
        _write(tmp_path, _VALID)
        feeds = load_configured_feeds(config_dir=str(tmp_path))
        assert len(feeds) == 2

    def test_not_found_error_points_at_example(self, tmp_path, isolated_discovery):
        with pytest.raises(FeedsConfigError) as exc:
            load_configured_feeds(feeds_file=str(tmp_path / "absent.yaml"))
        assert "feeds.example.yaml" in str(exc.value)
