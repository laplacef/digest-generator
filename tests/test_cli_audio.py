"""CLI tests for the audio surface: `digest_generator audio` and `--audio` flags.

Scoped narrowly to the audio commands. The deeper render path is exercised
through ``tests/core/audio/`` and ``tests/test_api.py``; here we just verify
the typer wiring: command discovery, argument validation, the
``--audio`` flag on ``run`` and ``digest``, and that the audio
subcommand reaches ``api.render_audio`` with the right run_dir.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from digest_generator.cli import app

runner = CliRunner()


def _make_run_dir(tmp_path: Path) -> Path:
    """A run_dir with a single digest .md so find_digest_md passes."""
    (tmp_path / "2026-05-11-weekly.md").write_text("# Weekly\n\nBody.\n")
    return tmp_path


class TestAudioCommand:
    """`digest_generator audio <run_dir>`: standalone audio render."""

    def test_command_exists(self):
        result = runner.invoke(app, ["audio", "--help"])
        assert result.exit_code == 0
        assert "Render the digest" in result.stdout

    def test_invokes_render_audio(self, tmp_path):
        _make_run_dir(tmp_path)
        opus_target = tmp_path / "audio" / "2026-05-11-weekly.opus"

        with patch("digest_generator.api.render_audio") as mock_render:
            mock_render.return_value = opus_target
            result = runner.invoke(app, ["audio", str(tmp_path)])

        assert result.exit_code == 0, result.stdout
        mock_render.assert_called_once_with(run_dir=tmp_path, bitrate_kbps=None)
        assert str(opus_target) in result.stdout

    def test_does_not_call_auto_login(self, tmp_path):
        """Audio path uses the public piper-voices repo, so no HF Hub auth is needed.

        Guards against an auto_login() call on the audio path, which can fail
        on a transient network blip (such as an SSL handshake timeout) even
        though no authentication is actually required.
        """
        _make_run_dir(tmp_path)
        with (
            patch("digest_generator.api.render_audio") as mock_render,
            patch("digest_generator.shared.hf_hub.auto_login") as mock_login,
        ):
            mock_render.return_value = tmp_path / "audio" / "x.opus"
            runner.invoke(app, ["audio", str(tmp_path)])

        mock_login.assert_not_called()

    def test_passes_bitrate_override(self, tmp_path):
        _make_run_dir(tmp_path)
        with patch("digest_generator.api.render_audio") as mock_render:
            mock_render.return_value = tmp_path / "audio" / "x.opus"
            result = runner.invoke(app, ["audio", str(tmp_path), "--bitrate-kbps", "48"])

        assert result.exit_code == 0
        mock_render.assert_called_once_with(run_dir=tmp_path, bitrate_kbps=48)

    def test_missing_run_dir_exits_nonzero(self, tmp_path):
        result = runner.invoke(app, ["audio", str(tmp_path / "does-not-exist")])
        assert result.exit_code == 1
        assert "not a directory" in result.stdout

    def test_run_dir_without_md_exits_nonzero(self, tmp_path):
        result = runner.invoke(app, ["audio", str(tmp_path)])
        assert result.exit_code == 1
        assert "no digest markdown" in result.stdout

    def test_render_failure_exits_nonzero(self, tmp_path):
        _make_run_dir(tmp_path)
        with patch("digest_generator.api.render_audio", side_effect=RuntimeError("piper crashed")):
            result = runner.invoke(app, ["audio", str(tmp_path)])
        assert result.exit_code == 1


class TestDigestAudioFlag:
    """`digest_generator digest <run_dir> --audio`: chained audio after digest."""

    def _make_summarized(self, tmp_path: Path) -> None:
        """Minimal scaffolding so the digest command passes its preflight check."""
        (tmp_path / "source-summarized").mkdir()

    def test_audio_flag_triggers_render(self, tmp_path):
        self._make_summarized(tmp_path)
        with (
            patch("digest_generator.api.digest") as mock_digest,
            patch("digest_generator.api.render_audio") as mock_render,
            patch(
                "digest_generator.core.digest.io.build_digest_filename",
                return_value="2026-05-11.md",
            ),
            patch("digest_generator.core.digest.io.build_digest_markdown", return_value="# x"),
        ):
            mock_digest.return_value.title = "Weekly"
            mock_digest.return_value.content = "# x"
            mock_render.return_value = tmp_path / "audio" / "x.opus"

            result = runner.invoke(app, ["digest", str(tmp_path), "--audio"])

        assert result.exit_code == 0, result.stdout
        mock_render.assert_called_once_with(run_dir=tmp_path)

    def test_no_audio_flag_skips_render(self, tmp_path):
        self._make_summarized(tmp_path)
        with (
            patch("digest_generator.api.digest") as mock_digest,
            patch("digest_generator.api.render_audio") as mock_render,
            patch(
                "digest_generator.core.digest.io.build_digest_filename",
                return_value="2026-05-11.md",
            ),
            patch("digest_generator.core.digest.io.build_digest_markdown", return_value="# x"),
        ):
            mock_digest.return_value.title = "Weekly"
            mock_digest.return_value.content = "# x"

            result = runner.invoke(app, ["digest", str(tmp_path)])

        assert result.exit_code == 0
        mock_render.assert_not_called()


class TestRunAudioFlag:
    """`digest_generator run --audio`: full pipeline plus audio at the end."""

    def test_audio_with_no_digest_rejected(self):
        result = runner.invoke(app, ["run", "--audio", "--no-digest", "-c", "ai"])
        assert result.exit_code == 1
        assert "--audio requires the digest stage" in result.stdout
