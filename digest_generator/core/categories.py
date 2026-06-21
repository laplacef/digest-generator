"""Runtime category set: the digest sections, defined in feeds.yaml.

A category is a user-defined section id paired with a display title. The
ordered set defines the digest's section order. Categories load from the
``categories:`` block of feeds.yaml via the RSS config loader.

``category_registry`` is the lazily-loaded singleton. Stages take a
``CategorySet`` by constructor injection and default to the registry, so a
caller (or a test) can supply an explicit set without touching global state.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Category:
    """One digest section: a stable id and a human-readable title."""

    id: str
    title: str


class CategorySet:
    """An ordered, immutable collection of categories.

    Iteration order is digest section order. Lookups by id are O(1).
    """

    def __init__(self, categories: list[Category]) -> None:
        self._ordered: tuple[Category, ...] = tuple(categories)
        self._by_id: dict[str, Category] = {c.id: c for c in self._ordered}

    @property
    def ids(self) -> tuple[str, ...]:
        """Category ids in section order."""
        return tuple(c.id for c in self._ordered)

    def id_set(self) -> frozenset[str]:
        """The category ids as a membership set."""
        return frozenset(self._by_id)

    def title(self, category_id: str) -> str:
        """Return the title for ``category_id``, or the id itself if unknown."""
        category = self._by_id.get(category_id)
        return category.title if category is not None else category_id

    def __iter__(self):
        return iter(self._ordered)

    def __contains__(self, category_id: object) -> bool:
        return category_id in self._by_id

    def __len__(self) -> int:
        return len(self._ordered)


class CategoryRegistry:
    """Lazily provides the active ``CategorySet``, loaded from feeds.yaml.

    The set is cached after first load. ``set`` injects a set directly (for
    a caller that already loaded one, or for tests); ``reset`` clears the
    cache so the next access reloads.
    """

    def __init__(self) -> None:
        self._cached: CategorySet | None = None

    @property
    def active(self) -> CategorySet:
        if self._cached is None:
            # Lazy import: config.py imports this module, so a top-level import
            # here would be circular.
            from digest_generator.sources.rss.config import (  # noqa: PLC0415
                load_configured_categories,
            )

            self._cached = load_configured_categories()
        return self._cached

    def set(self, categories: CategorySet | None) -> None:
        """Inject a category set (or ``None`` to fall back to lazy load)."""
        self._cached = categories

    def reset(self) -> None:
        """Drop the cached set so the next access reloads from feeds.yaml."""
        self._cached = None


category_registry = CategoryRegistry()
