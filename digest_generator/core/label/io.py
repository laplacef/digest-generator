"""Per-feed persistence for the label stages.

Writes ``<run_dir>/source-labeled/<feed>.json``. The sidecar
shape (``{url, labels[]}``) is minimal: ``source_type`` isn't
carried per record, so ``iter_labeled`` yields ``"rss"`` as a
placeholder until a multi-source manifest pattern emerges.

One artifact per stage: list of ``{"url": str, "labels": [{"value",
"confidence"}, ...]}``. URL-keyed (not index-keyed) so consumers are
robust to reordering. The topic stage writes here; future
sentiment / entailment stages use the same minimal shape.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from digest_generator.core.types import Label
from digest_generator.shared.logging import logger

_LABELED_DIR = "source-labeled"


def labeled_dir(run_dir: Path) -> Path:
    """Return the flat labeled directory under ``run_dir``."""
    return run_dir / _LABELED_DIR


def labeled_path(run_dir: Path, feed_name: str) -> Path:
    """Return the canonical path for one feed's labeled batch."""
    return labeled_dir(run_dir) / f"{feed_name}.json"


def save_labeled(
    run_dir: Path,
    feed_name: str,
    *,
    urls: list[str],
    labels_per_entry: list[list[Label]],
) -> Path:
    """Serialize per-entry label lists keyed by URL; return the written path.

    Schema: list of ``{"url": str, "labels": [{"value": str, "confidence":
    float}, ...]}``. URL-keyed (not index-keyed) so downstream consumers
    are robust to reordering between fetched and labeled batches.
    Atomic via tempfile + rename.

    Args:
        run_dir: Run root.
        feed_name: Filename stem.
        urls: URLs aligned with ``labels_per_entry`` by index.
        labels_per_entry: Stage output, one list of Labels per entry.

    Raises:
        ValueError: If ``len(urls) != len(labels_per_entry)``.
    """
    if len(urls) != len(labels_per_entry):
        msg = f"len(urls)={len(urls)} != len(labels_per_entry)={len(labels_per_entry)}"
        raise ValueError(msg)

    target = labeled_path(run_dir, feed_name)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = [
        {
            "url": url,
            "labels": [{"value": label.value, "confidence": label.confidence} for label in labels],
        }
        for url, labels in zip(urls, labels_per_entry, strict=True)
    ]

    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    tmp.replace(target)
    return target


def load_labeled(run_dir: Path, feed_name: str) -> dict[str, list[Label]] | None:
    """Read one feed's labeled JSON. Returns ``None`` when absent.

    Returns ``{url: list[Label]}`` for direct lookup by the digest
    consumer.
    """
    target = labeled_path(run_dir, feed_name)
    if not target.exists():
        return None
    with target.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        msg = f"Labeled file {target} is not a JSON list"
        raise ValueError(msg)
    return {item["url"]: _deserialize_labels(item["labels"]) for item in data}


def iter_labeled(
    run_dir: Path,
) -> Iterator[tuple[str, str, dict[str, list[Label]]]]:
    """Yield ``(source_type, feed_name, {url: labels})`` for every labeled batch.

    Walks ``<run_dir>/source-labeled/*.json``. ``source_type`` yields
    ``"rss"`` as a placeholder since the sidecar shape doesn't carry
    it per record; a multi-source-type mechanism lands alongside HN
    integration. Skips files that fail to parse, emitting a WARNING.
    """
    root = labeled_dir(run_dir)
    if not root.exists():
        return
    for json_path in sorted(root.glob("*.json")):
        try:
            with json_path.open(encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "iter_labeled: skipping {} ({}: {})",
                json_path,
                type(exc).__name__,
                exc,
            )
            continue
        if not isinstance(data, list):
            logger.warning("iter_labeled: {} is not a JSON list, skipping", json_path)
            continue
        try:
            labels_by_url = {item["url"]: _deserialize_labels(item["labels"]) for item in data}
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "iter_labeled: malformed entry in {} ({}: {})",
                json_path,
                type(exc).__name__,
                exc,
            )
            continue
        yield "rss", json_path.stem, labels_by_url


def _deserialize_labels(items: list[dict[str, Any]]) -> list[Label]:
    """Convert a JSON list of ``{value, confidence}`` dicts to ``list[Label]``."""
    return [Label(value=str(item["value"]), confidence=float(item["confidence"])) for item in items]
