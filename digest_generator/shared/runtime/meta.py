"""Pipeline run metadata: ``RunMeta`` dataclass + ``meta.json`` lifecycle.

Schema version 2 captures, in one file, every metric needed to evaluate or
reproduce a run: per-stage telemetry rolled up from each ``stage.done`` line,
the three-layer sampling state (user, then model_defaults, then effective)
per LLM stage, per-section editor outcomes, run-level token totals, and the
digest's terminal output metrics.

There is no v1 read path; the schema break is intentional. The eval harness
assumes v2.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from digest_generator.shared.logging import logger

if TYPE_CHECKING:
    from pathlib import Path


SCHEMA_VERSION = 2


@dataclass
class StageMeta:
    """Terminal telemetry harvested from a stage's ``stage.done`` line.

    The six common fields are typed; stage-specific fields (counts,
    booleans, lists like ``title_retry_reasons``) live in ``extras`` so
    the dataclass doesn't have to track every span field a stage adds.
    ``to_dict`` flattens ``extras`` into the stage object on JSON write,
    so the on-disk shape stays clean and consumers don't need to know
    the dataclass layout.
    """

    duration_ms: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_duration_ms: int = 0
    model: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flatten ``extras`` into the stage object for JSON output."""
        out: dict[str, Any] = {
            "duration_ms": self.duration_ms,
            "llm_calls": self.llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "llm_duration_ms": self.llm_duration_ms,
        }
        if self.model is not None:
            out["model"] = self.model
        # ``extras`` keys never collide with the typed fields above by convention;
        # if they do, ``extras`` wins (the harvest path is the authoritative source).
        out.update(self.extras)
        return out


@dataclass
class SamplingLayer:
    """One layer of sampling resolution. ``None`` = unset at this layer."""

    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None


SeedSource = Literal["user", "settings", "cli", "random"]


@dataclass
class SamplingMeta:
    """Three-layer sampling state for one stage.

    - ``user``: what the caller passed (CLI flag, API arg). Fields can be ``None``.
    - ``model_defaults``: what the model's Modelfile pins, falling back per-key
      to ``OLLAMA_DEFAULTS`` when the Modelfile doesn't carry the key (cloud
      models often don't expose every parameter).
    - ``effective``: the values actually used in the Ollama call. No ``None``s
      in any field; the seed in particular is materialized to a concrete int
      even when no layer supplied one.

    ``seed_source`` records which layer supplied the effective seed so an eval
    script can tell at a glance whether a run is reproducible from the meta
    alone (``"user"``, ``"settings"``, or ``"cli"``) or only via the materialized
    random value (``"random"``).
    """

    user: SamplingLayer = field(default_factory=SamplingLayer)
    model_defaults: SamplingLayer = field(default_factory=SamplingLayer)
    effective: SamplingLayer = field(default_factory=SamplingLayer)
    seed_source: SeedSource = "random"


EditOutcome = Literal["rewritten", "fell_back"]
RejectedReason = Literal["link_set", "h2_heading", "length_delta", "empty_response"]


@dataclass
class SectionMeta:
    """Per-section editor outcome: which sections kept the editor's rewrite."""

    name: str
    articles: int = 0
    edit_outcome: EditOutcome = "rewritten"
    rejected_reason: RejectedReason | None = None


@dataclass
class DigestMeta:
    """Composer-level output metrics, rolled up from ``DigestResult``."""

    title: str | None = None
    filename: str | None = None
    word_count: int = 0
    reading_time_minutes: int = 0
    article_count: int = 0
    section_counts: dict[str, int] = field(default_factory=dict)
    watch_item_count: int = 0


@dataclass
class RunMeta:
    """Operational metadata for a pipeline run, serialized to ``meta.json``.

    Schema v2: see module docstring for rationale and the full schema layout.
    """

    schema_version: int = SCHEMA_VERSION
    timestamp: str = ""
    duration_seconds: float = 0.0
    since: str | None = None
    until: str | None = None
    feed_count: int = 0
    article_count: int = 0
    content_types: list[str] = field(default_factory=list)
    models: dict[str, str | None] = field(default_factory=dict)
    revisions: dict[str, str] = field(default_factory=dict)
    sampling: dict[str, SamplingMeta] = field(default_factory=dict)
    stages: dict[str, StageMeta] = field(default_factory=dict)
    totals: dict[str, int] = field(default_factory=dict)
    sections: list[SectionMeta] = field(default_factory=list)
    digest: DigestMeta | None = None


_V2_KEY_ORDER: tuple[str, ...] = (
    "schema_version",
    "timestamp",
    "duration_seconds",
    "since",
    "until",
    "feed_count",
    "article_count",
    "content_types",
    "models",
    "revisions",
    "sampling",
    "stages",
    "totals",
    "sections",
    "digest",
)

# Flat v1 keys that duplicate their nested v2 counterpart and should be
# stripped on every write. ``<stage>_revision`` keys promote into the
# nested ``revisions`` map instead of being dropped outright.
_LEGACY_FLAT_REVISION_KEYS: dict[str, str] = {
    "summarizer_revision": "summarizer",
    "topic_revision": "topic",
}


def _to_dict(meta: RunMeta) -> dict[str, Any]:
    """Serialize ``RunMeta`` to a JSON-friendly dict in canonical v2 order.

    Stages flatten their ``extras`` map into the stage object via
    ``StageMeta.to_dict``. Other nested dataclasses use ``asdict`` since
    they have no extras layer.
    """
    return {
        "schema_version": meta.schema_version,
        "timestamp": meta.timestamp,
        "duration_seconds": meta.duration_seconds,
        "since": meta.since,
        "until": meta.until,
        "feed_count": meta.feed_count,
        "article_count": meta.article_count,
        "content_types": list(meta.content_types),
        "models": dict(meta.models),
        "revisions": dict(meta.revisions),
        "sampling": {k: asdict(v) for k, v in meta.sampling.items()},
        "stages": {k: v.to_dict() for k, v in meta.stages.items()},
        "totals": dict(meta.totals),
        "sections": [asdict(s) for s in meta.sections],
        "digest": asdict(meta.digest) if meta.digest is not None else None,
    }


def _normalize_v2(data: dict[str, Any]) -> dict[str, Any]:
    """Return a fresh dict containing only v2 keys, in canonical order.

    Promotes legacy flat ``<stage>_revision`` values into the nested
    ``revisions`` map (no overwrite: the nested value wins on conflict),
    drops every other legacy v1 key (``<stage>_model``, ``digest_title``,
    ``digest_filename``, etc.), and stamps ``schema_version``. Patch
    callers run this at the end of their read/modify/write cycle so the
    on-disk file converges on the v2 shape regardless of the input
    state.
    """
    revisions = dict(data.get("revisions") or {})
    for legacy_key, target in _LEGACY_FLAT_REVISION_KEYS.items():
        legacy_value = data.get(legacy_key)
        if legacy_value and target not in revisions:
            revisions[target] = legacy_value

    normalized: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    for key in _V2_KEY_ORDER[1:]:
        if key == "revisions":
            normalized[key] = revisions
        elif key in data:
            normalized[key] = data[key]
    return normalized


# Module-level lock for all ``meta.json`` mutation paths. ``api.summarize``
# and ``api.label`` run concurrently inside ``api.run`` and both patch
# ``meta.json``; serializing the read/modify/write cycle prevents one
# from clobbering the other's update. Atomic file replacement (``os.replace``)
# prevents partial-write artifacts under any reader's nose.
_meta_lock = threading.Lock()


def _atomic_write_json(meta_path: Path, data: dict[str, Any]) -> None:
    """Write JSON via tempfile + atomic rename so readers never see partials."""
    tmp_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    tmp_path.replace(meta_path)


def write_run_meta(meta: RunMeta, run_dir: Path) -> None:
    """Write the full v2 ``meta.json`` to the run directory (atomic, locked)."""
    meta_path = run_dir / "meta.json"
    with _meta_lock:
        _atomic_write_json(meta_path, _to_dict(meta))
    logger.debug("Run metadata written to {}", meta_path)


def _load_meta_or_empty(meta_path: Path) -> dict[str, Any]:
    """Return parsed ``meta.json`` contents, or an empty dict if absent.

    Tolerates a missing file so per-stage CLI workflows
    (``digest_generator summarize`` / ``label`` / ``digest`` run against a run_dir
    whose ``digest_generator run`` was never invoked or crashed before
    ``_write_initial_meta``) can still incrementally patch the file. The
    patch path is the load-bearing creator in that workflow; downstream
    consumers see a normalized v2 file with whatever blocks the patches
    contributed, just without the run-level fields (``run_id``, ``feeds``,
    ``since`` and ``until``) that only ``_write_initial_meta`` populates.
    """
    if not meta_path.exists():
        return {}
    with meta_path.open(encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
    return data


def update_run_meta_digest(run_dir: Path, title: str, filename: str) -> None:
    """Patch ``digest.title`` and ``digest.filename`` into ``meta.json``.

    Initializes ``digest`` to an empty ``DigestMeta`` if missing. Other digest
    fields (word_count, section_counts, etc.) are populated by
    ``update_run_meta_telemetry`` from the orchestrator's harvest path; this
    helper is the legacy CLI hook that only knew about the title and filename.

    Tolerates a missing ``meta.json`` and creates a v2 stub on first patch.
    """
    meta_path = run_dir / "meta.json"
    with _meta_lock:
        data = _load_meta_or_empty(meta_path)
        digest = data.get("digest") or {}
        digest["title"] = title
        digest["filename"] = filename
        data["digest"] = digest
        _atomic_write_json(meta_path, _normalize_v2(data))
    logger.debug("Updated meta.json with digest title + filename")


def update_run_meta_telemetry(
    run_dir: Path,
    *,
    models: dict[str, str | None] | None = None,
    stages: dict[str, StageMeta] | None = None,
    sampling: dict[str, SamplingMeta] | None = None,
    sections: list[SectionMeta] | None = None,
    totals: dict[str, int] | None = None,
    digest: DigestMeta | None = None,
) -> None:
    """Patch telemetry blocks into an existing ``meta.json``.

    Behavior per arg when supplied:

    - ``models``, ``stages``, ``sampling``: **merge per stage key** into the
      existing block. Lets ``api.summarize``, ``api.label``, and the digest
      orchestrator each contribute their own stage entries without erasing
      siblings written by the others.
    - ``sections``, ``totals``, ``digest``: **replace** the entire block.
      Each of these has a single canonical writer (the digest orchestrator),
      so a full replace is correct.

    The full read/modify/write cycle is locked so concurrent callers don't
    race; the file write itself is atomic via tempfile + rename.

    Tolerates a missing ``meta.json`` and creates a v2 stub on first patch
    so the per-stage CLI workflow (where ``cli.summarize`` and ``cli.label``
    run independently of ``cli.run``'s ``_write_initial_meta`` call) doesn't
    blow up at the orchestrator's terminal harvest step.
    """
    meta_path = run_dir / "meta.json"
    with _meta_lock:
        data = _load_meta_or_empty(meta_path)
        if models is not None:
            existing = data.get("models") or {}
            existing.update(models)
            data["models"] = existing
        if stages is not None:
            existing = data.get("stages") or {}
            existing.update({k: v.to_dict() for k, v in stages.items()})
            data["stages"] = existing
        if sampling is not None:
            existing = data.get("sampling") or {}
            existing.update({k: asdict(v) for k, v in sampling.items()})
            data["sampling"] = existing
        if sections is not None:
            data["sections"] = [asdict(s) for s in sections]
        if totals is not None:
            data["totals"] = dict(totals)
        if digest is not None:
            data["digest"] = asdict(digest)
        _atomic_write_json(meta_path, _normalize_v2(data))
    logger.debug("Updated meta.json with telemetry blocks")
