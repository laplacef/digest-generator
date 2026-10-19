"""RSS feed configuration: resolving and loading ``feeds.yaml``.

Feeds and categories are not hardcoded in the package. The user supplies a
``feeds.yaml`` (see ``feeds.example.yaml`` for the schema) with a
``categories:`` block (the digest sections, in order) and a ``feeds:`` block.
This module resolves that one file and validates it into a ``CategorySet``
plus a ``list[Feed]``.

The file is the explicit feeds file (``--feeds`` / ``FEEDS_FILE``) when one is
given, else ``<config dir>/feeds.yaml``, where the config dir is ``--config`` /
``DIGEST_CONFIG`` or ``~/.config/digest-generator``. There is no search: the
working directory is never consulted, and a selected file that does not exist
is an error rather than a cue to look elsewhere.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from digest_generator.core.categories import Category, CategorySet
from digest_generator.shared.settings import settings
from digest_generator.sources.rss.types import Feed

_CONFIG_FILENAME = "feeds.yaml"
_EXAMPLE_FILENAME = "feeds.example.yaml"


class FeedsConfigError(ValueError):
    """Raised when the feeds configuration is missing or invalid."""


class _CategoryEntry(BaseModel):
    """One category row in the ``categories:`` block of ``feeds.yaml``."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)


class _FeedEntry(BaseModel):
    """One feed row in ``feeds.yaml``."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    url: str = Field(min_length=1)
    category: str = Field(min_length=1)


class _FeedsDoc(BaseModel):
    """Top-level ``feeds.yaml`` document: categories plus feeds.

    Unknown top-level keys are ignored so a config carrying forward-looking
    sections still loads.
    """

    model_config = ConfigDict(extra="ignore")

    categories: list[_CategoryEntry] = Field(min_length=1)
    feeds: list[_FeedEntry] = Field(min_length=1)


def resolve_feeds_path(
    feeds_file: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> Path:
    """Return the ``feeds.yaml`` path the overrides select, whether or not it exists.

    ``feeds_file`` wins, then ``<config_dir>/feeds.yaml``. Unset arguments fall
    back to ``settings.feeds_file`` and ``settings.config_dir``.
    """
    feeds_file = feeds_file or settings.feeds_file
    if feeds_file:
        return Path(feeds_file).expanduser()
    if config_dir:
        return Path(config_dir).expanduser() / _CONFIG_FILENAME
    return settings.config_dir / _CONFIG_FILENAME


def load_config(path: Path) -> tuple[CategorySet, list[Feed]]:
    """Parse and validate a ``feeds.yaml`` file into categories and feeds.

    Categories and feeds are interdependent (every feed's ``category`` must
    name a defined category), so they load together.

    Args:
        path: Path to a ``feeds.yaml`` file.

    Returns:
        The ``CategorySet`` (in section order) and the feeds (in document
        order).

    Raises:
        FeedsConfigError: If the file is unreadable, empty, malformed,
            repeats a category id or feed name, or a feed names an unknown
            category.
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        msg = f"Cannot read feeds config at {path}: {exc}"
        raise FeedsConfigError(msg) from exc
    except yaml.YAMLError as exc:
        msg = f"Invalid YAML in feeds config at {path}: {exc}"
        raise FeedsConfigError(msg) from exc

    if raw is None:
        msg = f"Feeds config at {path} is empty. See feeds.example.yaml."
        raise FeedsConfigError(msg)

    try:
        doc = _FeedsDoc.model_validate(raw)
    except ValidationError as exc:
        msg = (
            f"Invalid feeds config at {path}:\n{exc}\n"
            "Needs a categories list (id + title) and a feeds list "
            "(name, url, category)."
        )
        raise FeedsConfigError(msg) from exc

    category_ids = [entry.id for entry in doc.categories]
    duplicate_categories = sorted({cid for cid in category_ids if category_ids.count(cid) > 1})
    if duplicate_categories:
        msg = f"Duplicate category id(s) in {path}: {', '.join(duplicate_categories)}"
        raise FeedsConfigError(msg)

    names = [entry.name for entry in doc.feeds]
    duplicate_names = sorted({name for name in names if names.count(name) > 1})
    if duplicate_names:
        msg = f"Duplicate feed name(s) in {path}: {', '.join(duplicate_names)}"
        raise FeedsConfigError(msg)

    known = set(category_ids)
    unknown = sorted({entry.category for entry in doc.feeds if entry.category not in known})
    if unknown:
        valid = ", ".join(category_ids)
        msg = (
            f"Feed(s) in {path} name unknown categor(ies): {', '.join(unknown)}. "
            f"Defined categories: {valid}."
        )
        raise FeedsConfigError(msg)

    categories = CategorySet([Category(id=entry.id, title=entry.title) for entry in doc.categories])
    feeds = [
        Feed(name=entry.name, url=entry.url, content_type=entry.category) for entry in doc.feeds
    ]
    return categories, feeds


def load_feeds(path: Path) -> list[Feed]:
    """Load just the feeds from a ``feeds.yaml`` file. See ``load_config``."""
    return load_config(path)[1]


def load_categories(path: Path) -> CategorySet:
    """Load just the categories from a ``feeds.yaml`` file. See ``load_config``."""
    return load_config(path)[0]


def load_configured(
    feeds_file: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> tuple[CategorySet, list[Feed]]:
    """Resolve and load the active ``feeds.yaml``. See ``resolve_feeds_path``.

    Raises:
        FeedsConfigError: If the resolved file does not exist or is invalid.
    """
    path = resolve_feeds_path(feeds_file, config_dir)
    if not path.is_file():
        msg = (
            f"No feeds.yaml found at {path}.\n"
            "Run 'digest-generator init' to create a starter feeds.yaml, "
            "or pass --feeds / --config to point at an existing one."
        )
        raise FeedsConfigError(msg)
    return load_config(path)


def load_configured_feeds(
    feeds_file: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> list[Feed]:
    """Resolve and load the active feeds. See ``load_configured``."""
    return load_configured(feeds_file, config_dir)[1]


def load_configured_categories(
    feeds_file: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> CategorySet:
    """Resolve and load the active category set. See ``load_configured``."""
    return load_configured(feeds_file, config_dir)[0]


def example_feeds_text() -> str:
    """Return the bundled starter ``feeds.yaml`` content.

    Reads the copy packaged into the wheel, or, in a source checkout where it
    is not packaged, the repo-root ``feeds.example.yaml``.
    """
    packaged = files("digest_generator") / _EXAMPLE_FILENAME
    if packaged.is_file():
        return packaged.read_text(encoding="utf-8")
    root = Path(__file__).resolve().parents[3] / _EXAMPLE_FILENAME
    return root.read_text(encoding="utf-8")


def write_starter_feeds(
    *,
    feeds_file: str | Path | None = None,
    config_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    """Write a starter ``feeds.yaml`` and return the path written.

    The target is the file later runs will read: see ``resolve_feeds_path``.

    Raises:
        FeedsConfigError: If the target already exists and ``force`` is False.
    """
    target = resolve_feeds_path(feeds_file, config_dir)
    if target.exists() and not force:
        msg = f"{target} already exists. Pass --force to overwrite it."
        raise FeedsConfigError(msg)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(example_feeds_text(), encoding="utf-8")
    return target
