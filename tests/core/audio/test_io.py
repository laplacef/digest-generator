"""Tests for digest_generator/core/audio/io.py: cache key and path helpers."""

import pytest

import digest_generator.core.audio.io as io_mod
from digest_generator.core.audio.io import (
    audio_dir,
    cache_key_path,
    compute_cache_key,
    find_digest_md,
    opus_path_for_digest,
    read_cache_key,
    write_cache_key,
)


class TestPaths:
    def test_audio_dir(self, tmp_path):
        assert audio_dir(tmp_path) == tmp_path / "audio"

    def test_cache_key_path(self, tmp_path):
        assert cache_key_path(tmp_path) == tmp_path / "audio" / "cache_key.txt"

    def test_opus_path_for_digest(self, tmp_path):
        md = tmp_path / "2026-05-11-weekly-ai-digest.md"
        md.touch()
        opus = opus_path_for_digest(tmp_path, md)
        assert opus == tmp_path / "audio" / "2026-05-11-weekly-ai-digest.opus"


class TestComputeCacheKey:
    def test_deterministic(self):
        k1 = compute_cache_key(b"# Hello", "voice-a", 24)
        k2 = compute_cache_key(b"# Hello", "voice-a", 24)
        assert k1 == k2

    def test_changes_on_md(self):
        k1 = compute_cache_key(b"# Hello", "v", 24)
        k2 = compute_cache_key(b"# Hello world", "v", 24)
        assert k1 != k2

    def test_changes_on_voice(self):
        k1 = compute_cache_key(b"# x", "voice-a", 24)
        k2 = compute_cache_key(b"# x", "voice-b", 24)
        assert k1 != k2

    def test_changes_on_bitrate(self):
        k1 = compute_cache_key(b"# x", "v", 24)
        k2 = compute_cache_key(b"# x", "v", 32)
        assert k1 != k2

    def test_changes_on_sentence_silence(self):
        k1 = compute_cache_key(b"# x", "v", 24, sentence_silence_s=0.4)
        k2 = compute_cache_key(b"# x", "v", 24, sentence_silence_s=0.5)
        assert k1 != k2

    def test_sentence_silence_none_vs_zero_distinct(self):
        # None (use Piper default) and 0.0 (no pause) are different intents;
        # the hash should distinguish them.
        k1 = compute_cache_key(b"# x", "v", 24, sentence_silence_s=None)
        k2 = compute_cache_key(b"# x", "v", 24, sentence_silence_s=0.0)
        assert k1 != k2

    def test_changes_when_narration_version_changes(self, monkeypatch):
        """Bumping NARRATION_VERSION must invalidate the cache key."""
        original_key = compute_cache_key(b"# x", "v", 24)
        monkeypatch.setattr(io_mod, "NARRATION_VERSION", "v999")
        bumped_key = compute_cache_key(b"# x", "v", 24)
        assert original_key != bumped_key

    def test_separator_prevents_aliasing(self):
        # Without a separator, concatenation could alias these:
        #   md="abc" + voice="def" + bitrate="24"
        #   md="abcdef" + voice=""    + bitrate="24"
        # The NUL separator guards against that.
        k1 = compute_cache_key(b"abc", "def", 24)
        k2 = compute_cache_key(b"abcdef", "", 24)
        assert k1 != k2


class TestReadWriteCacheKey:
    def test_read_missing_returns_none(self, tmp_path):
        assert read_cache_key(tmp_path) is None

    def test_round_trip(self, tmp_path):
        write_cache_key(tmp_path, "abc123")
        assert read_cache_key(tmp_path) == "abc123"

    def test_write_creates_audio_dir(self, tmp_path):
        write_cache_key(tmp_path, "key")
        assert audio_dir(tmp_path).exists()

    def test_empty_file_treated_as_missing(self, tmp_path):
        path = cache_key_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        assert read_cache_key(tmp_path) is None

    def test_strips_trailing_whitespace(self, tmp_path):
        path = cache_key_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("abc\n")
        assert read_cache_key(tmp_path) == "abc"


class TestFindDigestMd:
    def test_single_md_returned(self, tmp_path):
        md = tmp_path / "2026-05-11-digest.md"
        md.touch()
        # Sub-files shouldn't confuse the search.
        (tmp_path / "audio").mkdir()
        (tmp_path / "section-drafts").mkdir()
        (tmp_path / "section-drafts" / "models.md").touch()
        assert find_digest_md(tmp_path) == md

    def test_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no digest markdown"):
            find_digest_md(tmp_path)

    def test_multiple_raises(self, tmp_path):
        (tmp_path / "a.md").touch()
        (tmp_path / "b.md").touch()
        with pytest.raises(ValueError, match=r"multiple \.md deliverables"):
            find_digest_md(tmp_path)

    def test_only_root_md_considered(self, tmp_path):
        """Subdirectory .md files don't confuse the search."""
        deliverable = tmp_path / "2026-05-11-digest.md"
        deliverable.touch()
        nested = tmp_path / "section-drafts"
        nested.mkdir()
        (nested / "models.md").touch()
        (nested / "research.md").touch()
        # find_digest_md only looks at run_dir's immediate children.
        assert find_digest_md(tmp_path) == deliverable
