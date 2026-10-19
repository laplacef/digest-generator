"""Tests for digest_generator/cli.py helpers that run before settings load."""

import os

import pytest

from digest_generator.cli import _apply_env_overrides

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
