"""Tests for digest_generator/cli.py helpers that run before settings load."""

import os

import pytest
from typer.testing import CliRunner

from digest_generator.cli import _apply_env_overrides, app

_KEYS = ("DEVICE", "OUTPUT_DIR", "FEEDS_FILE", "DIGEST_CONFIG")


@pytest.fixture
def clean_env(monkeypatch):
    """Record each key so monkeypatch restores the real environment afterward."""
    for key in _KEYS:
        monkeypatch.setenv(key, "sentinel")
        monkeypatch.delenv(key)


class TestApplyEnvOverrides:
    def test_exports_config_selection(self, clean_env):
        _apply_env_overrides(None, None, feeds_file="/f/feeds.yaml", config_dir="/cfg")
        assert os.environ["FEEDS_FILE"] == "/f/feeds.yaml"
        assert os.environ["DIGEST_CONFIG"] == "/cfg"

    def test_unset_flags_leave_env_alone(self, clean_env):
        _apply_env_overrides(None, None)
        assert not any(key in os.environ for key in _KEYS)

    def test_exports_device_and_output_dir(self, clean_env):
        _apply_env_overrides("cuda", "out")
        assert os.environ["DEVICE"] == "cuda"
        assert os.environ["OUTPUT_DIR"] == "out"


_COMMANDS = ["init", "run", "fetch", "summarize", "label", "digest", "audio", "lint", "feeds"]


class TestHelpText:
    """Help output is plain terminal text: no RST markup, arrows, or module names."""

    @pytest.mark.parametrize("command", _COMMANDS)
    def test_help_has_no_markup(self, command):
        result = CliRunner().invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert "``" not in result.output
        assert "->" not in result.output

    def test_run_help_names_the_console_script(self):
        output = " ".join(CliRunner().invoke(app, ["run", "--help"]).output.split())
        assert "digest_generator digest" not in output
        assert "digest-generator digest" in output

    def test_bitrate_help_cites_the_env_var(self):
        output = CliRunner().invoke(app, ["audio", "--help"]).output
        assert "AUDIO_BITRATE_KBPS" in output
        assert "settings." not in output
