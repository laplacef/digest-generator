"""Shared pytest fixtures.

Tests stay hermetic against host config in two layers:

- Stage modules load their prompt templates at import, before any fixture
  runs, so ``PROMPTS_DIR`` points at an empty directory before the package is
  imported. Every template captured at collection is the bundled baseline.
- Two autouse fixtures cover calls made during a test. ``default_categories``
  injects a known category set into ``category_registry`` so stages resolve
  sections without reading a real ``feeds.yaml``, and ``bundled_prompts``
  points the config dir away from the host so ``load_prompt`` resolves the
  bundled baselines, matching CI.

Tests that exercise loading or a specific set/override it locally.
"""

import os
import tempfile

os.environ["PROMPTS_DIR"] = tempfile.mkdtemp(prefix="digest-generator-test-prompts-")

import pytest

from digest_generator.core.categories import Category, CategorySet, category_registry
from digest_generator.shared import settings as settings_module
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
    """Resolve bundled prompt baselines and feeds from an empty config dir."""
    monkeypatch.setattr(settings, "prompts_dir", None)
    monkeypatch.setattr(settings, "digest_config", None)
    monkeypatch.setattr(settings, "feeds_file", None)
    monkeypatch.setattr(settings_module, "USER_CONFIG_DIR", tmp_path / "no-home")
