"""Tests for digest_generator/shared/runtime/dirs.py: create_run_dir()."""

from unittest.mock import patch

from digest_generator.shared.runtime.dirs import create_run_dir


class TestCreateRunDir:
    """Mints a unique, timestamped run-output directory."""

    def test_creates_run_dir(self, tmp_path):
        run_dir = create_run_dir(output_dir=tmp_path)
        assert run_dir.exists()
        assert run_dir.is_dir()
        assert run_dir.parent == tmp_path

    def test_does_not_create_stage_subdirs(self, tmp_path):
        """Stage subdirs (source-fetched/, source-summarized/, source-labeled/) are created lazily by io.py."""
        run_dir = create_run_dir(output_dir=tmp_path)
        for stage_dir in ("source-fetched", "source-summarized", "source-labeled"):
            assert not (run_dir / stage_dir).exists()

    def test_directory_name_is_timestamped(self, tmp_path):
        run_dir = create_run_dir(output_dir=tmp_path)
        # Format: YYYY-MM-DD-HHmmss-xxxx (timestamp + 4-char hex suffix)
        name = run_dir.name
        assert len(name) == 22  # 2026-03-16-143022-a3f1
        assert name[4] == "-"
        assert name[7] == "-"
        assert name[10] == "-"
        assert name[17] == "-"

    def test_concurrent_runs_are_unique(self, tmp_path):
        """Two runs in the same second get distinct directories via random suffix."""
        run_dir1 = create_run_dir(output_dir=tmp_path)
        run_dir2 = create_run_dir(output_dir=tmp_path)
        assert run_dir1 != run_dir2
        assert run_dir1.exists()
        assert run_dir2.exists()

    def test_uses_settings_default(self, tmp_path):
        """When no output_dir given, uses settings.output_dir."""
        with patch("digest_generator.shared.runtime.dirs.settings") as mock_settings:
            mock_settings.output_dir = str(tmp_path)
            run_dir = create_run_dir()
            assert run_dir.parent == tmp_path
