"""Tests for digest_generator/core/digest/orchestrator.py cache helpers."""

from digest_generator.core.digest.orchestrator import _cached_or_fresh_framing
from digest_generator.core.digest.types import DigestFraming


class TestCachedOrFreshFraming:
    def test_caches_a_titled_framing(self, tmp_path):
        framing = _cached_or_fresh_framing(tmp_path, lambda: DigestFraming("Title", "Intro"))
        assert framing.title == "Title"
        assert (tmp_path / "assembly" / "framing.json").is_file()

    def test_never_caches_an_empty_title(self, tmp_path):
        _cached_or_fresh_framing(tmp_path, lambda: DigestFraming("", "Intro"))
        assert not (tmp_path / "assembly" / "framing.json").exists()
