"""Cross-stage domain vocabulary used by every pipeline stage.

These types describe the data flowing through the pipeline regardless of
its source: an ``Entry`` is whatever a fetcher produced, a ``Summary``
is what the summarizer made of it, ``Label`` is a generic
``(value, confidence)`` shape produced by every label stage (currently
``TopicType`` via the topic classifier; ``SentimentType`` and
``EntailmentType`` slot in as siblings when their stages exist),
``Filter`` parameterizes the fetch contract, and ``ContentType`` is
the section taxonomy the digest writer groups by.

Source-specific configuration types (``Feed``, RSS selectors, etc.) live
with their source in ``digest_generator.sources.rss.types``.
Stage-internal types (``DigestResult``, ``SectionDraft``, ...) live with
their stage in ``digest_generator.core.digest.types``. Infrastructure
types (``DeviceType``, ``ModelConfig``) live in ``digest_generator.shared.transformers.types``.
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum


class TopicType(StrEnum):
    """Zero-shot topic-classification labels assigned by the topic stage."""

    # AI / ML
    AGENTS = "agents"
    LLM = "large-language-models"
    MULTIMODAL = "multimodal"
    TRAINING = "training"
    MODEL_RELEASE = "model-release"
    EVALUATION = "evaluation-benchmarks"
    SAFETY = "safety"
    ENTERPRISE_AI = "enterprise-ai"
    ROBOTICS = "robotics"

    # Engineering
    API = "api"
    DATA_ENGINEERING = "data-engineering"
    DISTRIBUTED_SYSTEMS = "distributed-systems"
    SYSTEM_DESIGN = "system-design"
    OBSERVABILITY = "observability"
    CLOUD = "cloud"
    INFRASTRUCTURE = "infrastructure"
    DEVEX = "devex"
    TOOLING = "tooling"

    # Security
    CYBERSECURITY = "cybersecurity"
    THREAT_INTELLIGENCE = "threat-intelligence"
    PRIVACY = "privacy"
    VULNERABILITY = "vulnerability"

    # Event Type
    LAUNCH = "launch"
    RESEARCH = "research"
    CASE_STUDY = "case-study"
    OPEN_SOURCE = "open-source"
    POLICY = "policy"

    # Cryptocurrency
    CRYPTOCURRENCY = "cryptocurrency"
    BLOCKCHAIN = "blockchain"


class ContentType(StrEnum):
    """Section taxonomy for grouping articles in the weekly digest.

    Enum order defines the section order in the digest output.
    """

    AI = "ai"
    ENGINEERING = "engineering"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    BUSINESS = "business"

    @property
    def display_name(self) -> str:
        """Human-readable section heading for the weekly digest."""
        if self is ContentType.AI:
            return "AI & Machine Learning"
        return self.name.capitalize()


@dataclass
class Label:
    """A single classification result pairing a label value with its confidence.

    Generic across every label stage: ``value`` carries a string from one
    of the stage's enums (``TopicType``, and later ``SentimentType`` /
    ``EntailmentType`` once those stages exist). The discriminator is
    which field on ``Summary`` you read it from (``topics`` versus the
    eventual ``sentiment`` / ``entailment``), not the dataclass type.
    ``value`` is declared as ``str`` because every stage enum is a
    ``StrEnum``, so members coerce to strings transparently and
    persistence is symmetric across stages.
    """

    value: str
    confidence: float


@dataclass
class Filter:
    """Query filter for controlling which entries the pipeline processes.

    Use ``Filter.resolve()`` to create a filter with absolute timestamps
    from relative parameters. The API layer calls this before passing
    the filter to the pipeline, so the fetcher receives resolved dates.
    """

    days_back: int = 7
    since: datetime | None = None
    until: datetime | None = None
    limit: int | None = None

    @classmethod
    def resolve(
        cls,
        *,
        days_back: int = 7,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int | None = None,
    ) -> "Filter":
        """Create a Filter with absolute timestamps resolved from inputs.

        If ``since``/``until`` are provided, they take precedence over
        ``days_back``. If only ``since`` is given, ``until`` defaults to now.
        If neither is given, both are computed from ``days_back``.

        Args:
            days_back: Number of days back (used when since/until not set).
            since: Absolute start date (exclusive lower bound).
            until: Absolute end date (exclusive upper bound).
            limit: Maximum entries per feed.

        Returns:
            A fully resolved Filter with ``since`` and ``until`` populated.

        Raises:
            ValueError: If since >= until, or limit <= 0.
        """
        now = datetime.now(UTC)

        if since is not None or until is not None:
            resolved_since = since or (now - timedelta(days=days_back))
            resolved_until = until or now
        else:
            resolved_since = now - timedelta(days=days_back)
            resolved_until = now

        if resolved_since >= resolved_until:
            msg = (
                f"'since' ({resolved_since.isoformat()}) must be before "
                f"'until' ({resolved_until.isoformat()})"
            )
            raise ValueError(msg)

        if limit is not None and limit <= 0:
            msg = f"'limit' must be positive, got {limit}"
            raise ValueError(msg)

        return cls(
            days_back=days_back,
            since=resolved_since,
            until=resolved_until,
            limit=limit,
        )


@dataclass
class Entry:
    """A single article fetched by any source, with full extracted content.

    Identity:
        ``origin`` is the per-source-type identifier of where this entry
        came from (feed slug for RSS, subreddit for Reddit, board for
        HN, channel name for YouTube). Polymorphic across content
        providers. ``source_type`` is the content-source provider
        family (such as ``"rss"``, ``"hn"``, ``"reddit"``, or
        ``"youtube"``). Together ``(source_type, origin)`` uniquely
        identifies an entry's origin. ``content_type`` is the editorial
        category from the source's config; it lives in the entry rather
        than the directory path so downstream stages have a single
        source of truth. ``fetched_at`` is the timestamp at fetch time,
        useful for staleness checks and debugging.

    Payload:
        ``content`` carries the full extracted article text for
        summarization. ``content_head`` is a truncated copy
        (word-boundary-safe) fed to the digest writer so it can cite
        publisher prose without going through the LLM summarizer's
        compression. Empty string when the source provided neither full
        content nor an excerpt long enough to trim.
    """

    title: str
    url: str
    origin: str
    published: datetime
    description: str
    content: str
    content_head: str = ""
    source_type: str = "rss"
    content_type: ContentType | None = None
    fetched_at: datetime | None = None


@dataclass
class Summary:
    """An article summary with its source entry, generated text, and topic labels.

    ``topics`` carries the topic-stage output. Future stages add
    sibling fields (``sentiment: Label | None`` for a sentiment stage;
    ``entailment: list[Label]`` for a per-claim entailment stage).
    """

    entry: Entry
    summary: str
    length: int
    topics: list[Label]
