"""Tests for digest_generator/core/prompt_loader.py: override-vs-bundled resolution.

The loader prefers a user override directory and falls back to the bundled
template per name, so a user can override any subset of prompts.
"""

import pytest

from digest_generator.core import prompt_loader
from digest_generator.core.prompt_loader import resolve_prompt
from digest_generator.shared.settings import settings


@pytest.fixture
def bundled(tmp_path):
    """A bundled templates dir with two templates."""
    d = tmp_path / "bundled"
    d.mkdir()
    (d / "alpha.md").write_text("bundled alpha", encoding="utf-8")
    (d / "beta.md").write_text("bundled beta {{style:filler_adjectives}}", encoding="utf-8")
    return d


@pytest.fixture
def clear_override(monkeypatch, tmp_path):
    """Point every override-dir source at empty temp locations.

    So a real prompts/ on the host can't make bundled-fallback tests pass by
    accident.
    """
    monkeypatch.setattr(settings, "prompts_dir", None)
    monkeypatch.setattr(settings, "digest_config", None)
    monkeypatch.setattr(prompt_loader, "_USER_CONFIG_DIR", tmp_path / "no-home")
    monkeypatch.chdir(tmp_path)


class TestResolvePrompt:
    def test_bundled_when_no_override(self, bundled, clear_override):
        assert resolve_prompt("alpha", bundled_dir=bundled) == "bundled alpha"

    def test_override_wins(self, bundled, monkeypatch, tmp_path):
        override = tmp_path / "ov"
        override.mkdir()
        (override / "alpha.md").write_text("override alpha", encoding="utf-8")
        monkeypatch.setattr(settings, "prompts_dir", str(override))
        assert resolve_prompt("alpha", bundled_dir=bundled) == "override alpha"

    def test_partial_override_falls_back_per_name(self, bundled, monkeypatch, tmp_path):
        override = tmp_path / "ov"
        override.mkdir()
        (override / "alpha.md").write_text("override alpha", encoding="utf-8")
        monkeypatch.setattr(settings, "prompts_dir", str(override))
        # alpha overridden, beta falls back to bundled
        assert resolve_prompt("alpha", bundled_dir=bundled) == "override alpha"
        assert resolve_prompt("beta", bundled_dir=bundled).startswith("bundled beta")

    def test_style_expanded_in_override(self, bundled, monkeypatch, tmp_path):
        override = tmp_path / "ov"
        override.mkdir()
        (override / "beta.md").write_text("custom {{style:filler_adjectives}}", encoding="utf-8")
        monkeypatch.setattr(settings, "prompts_dir", str(override))
        out = resolve_prompt("beta", bundled_dir=bundled)
        assert "{{style:" not in out
        assert '"critical,"' in out

    def test_config_dir_prompts_subdir(self, bundled, monkeypatch, tmp_path):
        config = tmp_path / "cfg"
        (config / "prompts").mkdir(parents=True)
        (config / "prompts" / "alpha.md").write_text("cfg alpha", encoding="utf-8")
        monkeypatch.setattr(settings, "prompts_dir", None)
        monkeypatch.setattr(settings, "digest_config", str(config))
        assert resolve_prompt("alpha", bundled_dir=bundled) == "cfg alpha"


class TestOverrideDirs:
    def test_prompts_dir_first(self, monkeypatch):
        monkeypatch.setattr(settings, "prompts_dir", "/explicit/prompts")
        monkeypatch.setattr(settings, "digest_config", "/cfg")
        dirs = prompt_loader.override_dirs()
        assert str(dirs[0]) == "/explicit/prompts"
        assert dirs[1].parts[-2:] == ("cfg", "prompts")

    def test_user_level_last(self, monkeypatch):
        monkeypatch.setattr(settings, "prompts_dir", None)
        monkeypatch.setattr(settings, "digest_config", None)
        dirs = prompt_loader.override_dirs()
        # Last candidate is <user config dir>/prompts (the module constant,
        # which the autouse fixture repoints away from the real host path).
        assert dirs[-1] == prompt_loader._USER_CONFIG_DIR / "prompts"
        assert dirs[-1].name == "prompts"
