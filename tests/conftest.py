"""Shared pytest fixtures.

Two autouse fixtures keep tests hermetic against host config:

- ``default_categories`` injects a known category set into ``category_registry``
  so stages resolve sections without reading a real ``feeds.yaml``.
- ``bundled_prompts`` points the prompt-override search away from the host so
  ``load_prompt`` resolves the bundled baselines, matching CI (a developer's
  ``~/.config/digest-generator/prompts/`` must not shadow them under test).

Tests that exercise loading or a specific set/override it locally.
"""

import pytest

from digest_generator.core import prompt_loader
from digest_generator.core.categories import Category, CategorySet, category_registry
from digest_generator.shared.settings import settings

_DEFAULT_TEST_CATEGORIES = CategorySet(
    [
        Category("ai", "AI & Machine Learning"),
        Category("engineering", "Engineering"),
        Category("infrastructure", "Infrastructure"),
        Category("security", "Security"),
        Category("business", "Business"),
    ]
)


@pytest.fixture(autouse=True)
def default_categories():
    """Inject the canonical test category set, reset after each test."""
    category_registry.set(_DEFAULT_TEST_CATEGORIES)
    yield
    category_registry.reset()


@pytest.fixture(autouse=True)
def bundled_prompts(monkeypatch, tmp_path):
    """Resolve bundled prompt baselines, not a host override dir."""
    monkeypatch.setattr(settings, "prompts_dir", None)
    monkeypatch.setattr(settings, "digest_config", None)
    monkeypatch.setattr(prompt_loader, "_USER_CONFIG_DIR", tmp_path / "no-home")
