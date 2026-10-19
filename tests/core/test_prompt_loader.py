"""Tests for digest_generator/core/prompt_loader.py: override-vs-bundled resolution.

The loader reads one user override directory and falls back to the bundled
template per name, so a user can override any subset of prompts. It never
layers directories and never reads the working directory.
"""

from pathlib import Path

import pytest

from digest_generator.core import prompt_loader
from digest_generator.core.digest import prompts
from digest_generator.core.digest.stages import writer
from digest_generator.core.prompt_loader import resolve_prompt
from digest_generator.core.style import expand_style_placeholders
from digest_generator.shared import settings as settings_module
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
    monkeypatch.setattr(settings_module, "USER_CONFIG_DIR", tmp_path / "no-home")
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

    def test_working_directory_ignored(self, bundled, clear_override, tmp_path):
        local = tmp_path / "digest-generator" / "prompts"
        local.mkdir(parents=True)
        (local / "alpha.md").write_text("cwd alpha", encoding="utf-8")
        assert resolve_prompt("alpha", bundled_dir=bundled) == "bundled alpha"

    def test_no_layering_across_directories(self, bundled, monkeypatch, tmp_path):
        # PROMPTS_DIR lacks beta and the config dir has it; beta must come from
        # the bundled baseline rather than a second override directory.
        override = tmp_path / "ov"
        override.mkdir()
        (override / "alpha.md").write_text("override alpha", encoding="utf-8")
        config = tmp_path / "cfg"
        (config / "prompts").mkdir(parents=True)
        (config / "prompts" / "beta.md").write_text("cfg beta", encoding="utf-8")
        monkeypatch.setattr(settings, "prompts_dir", str(override))
        monkeypatch.setattr(settings, "digest_config", str(config))
        assert resolve_prompt("alpha", bundled_dir=bundled) == "override alpha"
        assert resolve_prompt("beta", bundled_dir=bundled).startswith("bundled beta")


class TestOverrideDir:
    def test_prompts_dir_wins(self, monkeypatch):
        monkeypatch.setattr(settings, "prompts_dir", "/explicit/prompts")
        monkeypatch.setattr(settings, "digest_config", "/cfg")
        assert prompt_loader.override_dir() == Path("/explicit/prompts")

    def test_config_dir_prompts(self, monkeypatch):
        monkeypatch.setattr(settings, "digest_config", "/cfg")
        assert prompt_loader.override_dir() == Path("/cfg/prompts")

    def test_user_config_dir_by_default(self):
        # The autouse fixture repoints USER_CONFIG_DIR away from the host path.
        assert prompt_loader.override_dir() == settings_module.USER_CONFIG_DIR / "prompts"


class TestImportTimeIsolation:
    def test_stage_constants_hold_bundled_baselines(self):
        # Stage modules load templates at import; conftest's empty PROMPTS_DIR
        # must keep host overrides out of those module constants.
        bundled = (prompts._TEMPLATES_DIR / "section_system.md").read_text(encoding="utf-8")
        assert expand_style_placeholders(bundled) == writer._SECTION_SYSTEM_PROMPT
