"""Tests for digest_generator/sources/rss/config.py: feeds.yaml resolution + loading.

Feeds and categories come from a user-supplied feeds.yaml. These tests cover
the happy path, every validation error, and single-path resolution (no search,
no working-directory lookup, no fallthrough past a missing selected file).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from digest_generator.shared import settings as settings_module
from digest_generator.shared.settings import settings
from digest_generator.sources.rss.config import (
    FeedsConfigError,
    example_feeds_text,
    load_categories,
    load_config,
    load_configured,
    load_configured_categories,
    load_configured_feeds,
    load_feeds,
    resolve_feeds_path,
    write_starter_feeds,
)

_CATEGORIES = """\
categories:
  - id: ai
    title: "AI & Machine Learning"
  - id: security
    title: "Security"
"""

_VALID = (
    _CATEGORIES
    + """\
feeds:
  - name: openai-news
    url: https://openai.com/news/rss.xml
    category: ai
  - name: krebs
    url: https://krebsonsecurity.com/feed/
    category: security
"""
)


def _write(tmp_path, text, name="feeds.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def user_config_dir(tmp_path, monkeypatch):
    """Point the user-level config dir and cwd at empty temp locations.

    The autouse ``bundled_prompts`` fixture already clears the settings
    fallbacks; this also moves cwd so a stray ``./digest-generator/`` can't
    leak in.
    """
    path = tmp_path / "home" / "digest-generator"
    monkeypatch.setattr(settings_module, "USER_CONFIG_DIR", path)
    monkeypatch.chdir(tmp_path)
    return path


class TestLoadConfig:
    """load_config: parse + validate categories and feeds together."""

    def test_valid_file(self, tmp_path):
        categories, feeds = load_config(_write(tmp_path, _VALID))
        assert categories.ids == ("ai", "security")
        assert categories.title("ai") == "AI & Machine Learning"
        assert [f.name for f in feeds] == ["openai-news", "krebs"]
        assert feeds[0].content_type == "ai"
        assert feeds[1].content_type == "security"
        assert feeds[0].url == "https://openai.com/news/rss.xml"

    def test_category_order_preserved(self, tmp_path):
        text = (
            "categories:\n"
            "  - id: security\n    title: Security\n"
            "  - id: ai\n    title: AI\n"
            "feeds:\n  - name: f\n    url: https://e\n    category: ai\n"
        )
        categories, _ = load_config(_write(tmp_path, text))
        assert categories.ids == ("security", "ai")

    def test_preserves_feed_document_order(self, tmp_path):
        text = (
            _CATEGORIES
            + "feeds:\n"
            + "".join(
                f"  - name: f{i}\n    url: https://e/{i}\n    category: ai\n" for i in range(5)
            )
        )
        feeds = load_feeds(_write(tmp_path, text))
        assert [f.name for f in feeds] == [f"f{i}" for i in range(5)]

    def test_unknown_top_level_keys_ignored(self, tmp_path):
        feeds = load_feeds(_write(tmp_path, "version: 2\n" + _VALID))
        assert len(feeds) == 2

    def test_empty_file(self, tmp_path):
        with pytest.raises(FeedsConfigError, match="empty"):
            load_config(_write(tmp_path, ""))

    def test_missing_categories(self, tmp_path):
        text = "feeds:\n  - name: x\n    url: https://e\n    category: ai\n"
        with pytest.raises(FeedsConfigError):
            load_config(_write(tmp_path, text))

    def test_empty_feeds_list(self, tmp_path):
        with pytest.raises(FeedsConfigError):
            load_config(_write(tmp_path, _CATEGORIES + "feeds: []\n"))

    def test_feed_names_unknown_category(self, tmp_path):
        text = _CATEGORIES + "feeds:\n  - name: x\n    url: https://e\n    category: politics\n"
        with pytest.raises(FeedsConfigError) as exc:
            load_config(_write(tmp_path, text))
        assert "politics" in str(exc.value)

    def test_missing_required_feed_field(self, tmp_path):
        text = _CATEGORIES + "feeds:\n  - name: x\n    category: ai\n"  # no url
        with pytest.raises(FeedsConfigError):
            load_config(_write(tmp_path, text))

    def test_extra_feed_field_rejected(self, tmp_path):
        text = _CATEGORIES + (
            "feeds:\n  - name: x\n    url: https://e\n    category: ai\n    provider: openai\n"
        )
        with pytest.raises(FeedsConfigError):
            load_config(_write(tmp_path, text))

    def test_duplicate_feed_names(self, tmp_path):
        text = _CATEGORIES + (
            "feeds:\n"
            "  - name: dup\n    url: https://a\n    category: ai\n"
            "  - name: dup\n    url: https://b\n    category: security\n"
        )
        with pytest.raises(FeedsConfigError, match="Duplicate feed"):
            load_config(_write(tmp_path, text))

    def test_duplicate_category_ids(self, tmp_path):
        text = (
            "categories:\n"
            "  - id: ai\n    title: AI\n"
            "  - id: ai\n    title: Also AI\n"
            "feeds:\n  - name: f\n    url: https://e\n    category: ai\n"
        )
        with pytest.raises(FeedsConfigError, match="Duplicate category"):
            load_config(_write(tmp_path, text))

    def test_malformed_yaml(self, tmp_path):
        with pytest.raises(FeedsConfigError, match="Invalid YAML"):
            load_config(_write(tmp_path, "categories: [unterminated\n"))

    def test_load_categories_helper(self, tmp_path):
        categories = load_categories(_write(tmp_path, _VALID))
        assert categories.ids == ("ai", "security")


class TestResolveFeedsPath:
    """resolve_feeds_path: one selected file, never a search."""

    def test_explicit_file_wins(self):
        path = resolve_feeds_path(feeds_file="/explicit/feeds.yaml", config_dir="/cfg")
        assert path == Path("/explicit/feeds.yaml")

    def test_config_dir(self):
        assert resolve_feeds_path(config_dir="/cfg") == Path("/cfg/feeds.yaml")

    def test_settings_fallbacks(self, monkeypatch):
        monkeypatch.setattr(settings, "digest_config", "/cfg")
        assert resolve_feeds_path() == Path("/cfg/feeds.yaml")
        monkeypatch.setattr(settings, "feeds_file", "/env/feeds.yaml")
        assert resolve_feeds_path() == Path("/env/feeds.yaml")

    def test_user_config_dir_by_default(self, user_config_dir):
        assert resolve_feeds_path() == user_config_dir / "feeds.yaml"


class TestLoadConfigured:
    """load_configured and its wrappers: resolve + load."""

    def test_loads_via_explicit_file(self, tmp_path):
        path = _write(tmp_path, _VALID)
        assert len(load_configured_feeds(feeds_file=str(path))) == 2

    def test_loads_categories_via_explicit_file(self, tmp_path):
        path = _write(tmp_path, _VALID)
        assert load_configured_categories(feeds_file=str(path)).ids == ("ai", "security")

    def test_returns_categories_and_feeds(self, tmp_path):
        categories, feeds = load_configured(feeds_file=str(_write(tmp_path, _VALID)))
        assert categories.ids == ("ai", "security")
        assert [f.name for f in feeds] == ["openai-news", "krebs"]

    def test_loads_via_config_dir(self, tmp_path):
        _write(tmp_path, _VALID)
        assert len(load_configured_feeds(config_dir=str(tmp_path))) == 2

    def test_loads_user_config_by_default(self, user_config_dir):
        user_config_dir.mkdir(parents=True)
        _write(user_config_dir, _VALID)
        assert len(load_configured_feeds()) == 2

    def test_not_found_error_names_path_and_suggests_init(self, tmp_path, user_config_dir):
        absent = tmp_path / "absent.yaml"
        with pytest.raises(FeedsConfigError) as exc:
            load_configured_feeds(feeds_file=str(absent))
        assert str(absent) in str(exc.value)
        assert "digest-generator init" in str(exc.value)

    def test_missing_explicit_file_does_not_fall_through(self, tmp_path, user_config_dir):
        # A valid default exists, but the selected file wins even when absent.
        user_config_dir.mkdir(parents=True)
        _write(user_config_dir, _VALID)
        with pytest.raises(FeedsConfigError):
            load_configured_feeds(feeds_file=str(tmp_path / "typo.yaml"))

    def test_working_directory_ignored(self, tmp_path, user_config_dir):
        local = tmp_path / "digest-generator"
        local.mkdir()
        _write(local, _VALID)
        with pytest.raises(FeedsConfigError):
            load_configured_feeds()


class TestStarterFeeds:
    """example_feeds_text / write_starter_feeds: the `init` scaffolding."""

    def test_example_text_is_valid_config(self):
        text = example_feeds_text()
        assert "categories:" in text
        assert "feeds:" in text

    def test_writes_to_explicit_file_and_loads(self, tmp_path):
        target = tmp_path / "feeds.yaml"
        written = write_starter_feeds(feeds_file=str(target))
        assert written == target
        # The written starter round-trips through the loader.
        categories, feeds = load_config(written)
        assert len(categories) >= 1
        assert len(feeds) >= 1

    def test_writes_into_config_dir(self, tmp_path):
        written = write_starter_feeds(config_dir=str(tmp_path))
        assert written == tmp_path / "feeds.yaml"
        assert written.is_file()

    def test_default_target_is_user_config(self, user_config_dir):
        written = write_starter_feeds()
        assert written == user_config_dir / "feeds.yaml"
        assert written.is_file()

    def test_no_clobber_without_force(self, tmp_path):
        target = tmp_path / "feeds.yaml"
        write_starter_feeds(feeds_file=str(target))
        with pytest.raises(FeedsConfigError, match="already exists"):
            write_starter_feeds(feeds_file=str(target))

    def test_force_overwrites(self, tmp_path):
        target = tmp_path / "feeds.yaml"
        target.write_text("stale", encoding="utf-8")
        write_starter_feeds(feeds_file=str(target), force=True)
        assert "categories:" in target.read_text(encoding="utf-8")
