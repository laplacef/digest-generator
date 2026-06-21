"""Tests for digest_generator/core/categories.py: Category, CategorySet, registry.

Categories are user-defined, so these tests use arbitrary ids (not the
project's historical sections) to prove the set is not hardcoded.
"""

from unittest.mock import MagicMock

import pytest

from digest_generator.core.categories import (
    Category,
    CategoryRegistry,
    CategorySet,
)
from digest_generator.core.digest.stages.writer import SectionWriter


@pytest.fixture
def arbitrary():
    """A category set with ids that never existed as a hardcoded enum."""
    return CategorySet(
        [
            Category("robotics", "Robotics"),
            Category("quantum", "Quantum Computing"),
            Category("bio", "Biotech"),
        ]
    )


class TestCategorySet:
    def test_ids_in_order(self, arbitrary):
        assert arbitrary.ids == ("robotics", "quantum", "bio")

    def test_iteration_is_section_order(self, arbitrary):
        assert [c.id for c in arbitrary] == ["robotics", "quantum", "bio"]

    def test_title_lookup(self, arbitrary):
        assert arbitrary.title("quantum") == "Quantum Computing"

    def test_title_passthrough_on_unknown(self, arbitrary):
        # Defensive: an unknown id renders as itself, never raises.
        assert arbitrary.title("nope") == "nope"

    def test_id_set_membership(self, arbitrary):
        assert arbitrary.id_set() == frozenset({"robotics", "quantum", "bio"})
        assert "robotics" in arbitrary
        assert "ai" not in arbitrary

    def test_len(self, arbitrary):
        assert len(arbitrary) == 3

    def test_empty_set(self):
        empty = CategorySet([])
        assert empty.ids == ()
        assert len(empty) == 0
        assert empty.title("x") == "x"


class TestCategoryRegistry:
    def test_set_and_active(self, arbitrary):
        registry = CategoryRegistry()
        registry.set(arbitrary)
        assert registry.active is arbitrary

    def test_reset_clears_injected_set(self, arbitrary):
        registry = CategoryRegistry()
        registry.set(arbitrary)
        registry.reset()
        # After reset the cache is empty; active would lazy-load from config.
        assert registry._cached is None

    def test_active_lazy_loads_from_config(self, monkeypatch):
        sentinel = CategorySet([Category("x", "X")])
        registry = CategoryRegistry()
        called = {"n": 0}

        def fake_load():
            called["n"] += 1
            return sentinel

        monkeypatch.setattr(
            "digest_generator.sources.rss.config.load_configured_categories", fake_load
        )
        assert registry.active is sentinel
        # Second access is cached, no reload.
        assert registry.active is sentinel
        assert called["n"] == 1


class TestArbitraryCategoriesThroughWriter:
    """The writer groups and orders by whatever categories are configured."""

    def test_grouping_and_section_order_follow_arbitrary_set(self, arbitrary):
        writer = SectionWriter(client=MagicMock(), categories=arbitrary)
        results = {
            "feed": [
                {"title": "Q", "url": "https://q", "content_type": "quantum"},
                {"title": "R", "url": "https://r", "content_type": "robotics"},
                {"title": "Unknown", "url": "https://u", "content_type": "ai"},
            ]
        }
        grouped = writer._group_by_clusters(results, None)
        # Known categories grouped; the unconfigured "ai" article is dropped.
        assert set(grouped) == {"quantum", "robotics"}
        assert "ai" not in grouped
