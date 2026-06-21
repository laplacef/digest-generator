"""Shared pytest fixtures.

The autouse ``default_categories`` fixture injects a known category set into
``category_registry`` for every test, so stages that default to the registry
resolve sections without reading a real ``feeds.yaml`` from disk. Tests that
exercise loading or a specific category set override it locally.
"""

import pytest

from digest_generator.core.categories import Category, CategorySet, category_registry

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
