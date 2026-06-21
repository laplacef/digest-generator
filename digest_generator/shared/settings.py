"""Centralized application settings via pydantic-settings.

All hardcoded configuration values are consolidated here. Each field maps to an
uppercase environment variable (e.g., ``device`` -> ``DEVICE``). Values can be
overridden via ``.env`` file or environment variables.

Field ordering follows pipeline structure:

1. Foundational / shared infrastructure: device, HuggingFace, Ollama
2. Pipeline stages in execution order: fetcher, summarizer, topic, writer,
   editorial, framer, watcher
3. Pipeline-level: days_back, output_dir
4. Logging

All modules should import the pre-configured singleton::

    from digest_generator.shared.settings import settings
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # =========================================================================
    # Foundational / shared infrastructure
    # =========================================================================

    # -- Device --
    device: str = Field(default="cpu", description="Compute device: cpu, cuda, or mps")

    # -- HuggingFace --
    hf_token: str = Field(default="", description="HuggingFace API token for model downloads")

    # -- Ollama (shared by summarizer + every digest stage) --
    ollama_host: str = "http://localhost:11434"
    ollama_api_key: str = ""
    # Global cap on in-flight Ollama calls across every stage. Acquired by
    # `chat_with_logging` so it covers the summarizer fan-out *and* any
    # future digest stage that parallelizes calls. The summarizer's own
    # `summarizer_concurrency` is the inner per-instance cap that bounds
    # async-task fan-out (and thus thread-pool consumption); this one
    # bounds actual cloud Ollama load.
    #
    # Default 3 is conservative for cloud Ollama: the per-account ceiling is
    # well under 8, and a default of 8 can trip 429 storms. Local Ollama
    # operators can raise this via OLLAMA_CONCURRENCY=8 (or whatever the GPU's
    # parallel-decode capacity supports).
    ollama_concurrency: int = 3
    # httpx read timeout (seconds) for Ollama calls. The Ollama SDK
    # defaults to no timeout, which means a hung socket blocks a worker
    # thread forever. 300s is generous for legitimate slow cloud
    # generations; lower for fast local models if desired.
    ollama_read_timeout_s: int = 300

    # =========================================================================
    # Pipeline stages (in execution order)
    # =========================================================================

    # -- Feeds source (RSS feed discovery) --
    # Explicit path to a feeds.yaml file. When unset, the loader walks the
    # config-dir / project-local / user-level search path (see
    # digest_generator.sources.rss.config).
    feeds_file: str | None = None
    # Config directory holding feeds.yaml (and optional prompts/ overrides).
    # Maps to DIGEST_CONFIG.
    digest_config: str | None = None
    # Directory of prompt-template overrides (<name>.md). When unset, the
    # loader checks <DIGEST_CONFIG>/prompts/, then ./digest-generator/prompts/,
    # then ~/.config/digest-generator/prompts/, else the bundled baselines.
    prompts_dir: str | None = None

    # -- Fetcher --
    fetch_timeout: int = 10
    fetch_rate_limit: float = 0.4
    fetch_concurrency: int = 10
    min_content_length: int = 200
    max_boilerplate_ratio: float = 0.05

    # -- Summarizer (LLM-driven per-article fact extraction) --
    summarizer_model: str = "gemma4:31b-cloud"
    summarizer_temperature: float = 0.2
    summarizer_top_p: float | None = None
    summarizer_repetition_penalty: float | None = None
    summarizer_seed: int | None = None
    # Per-article LLM calls fan out via asyncio.gather; this caps in-flight
    # calls. Tune separately from fetch_concurrency since cloud LLM rate
    # limits and feed-host rate limits are different ceilings.
    summarizer_concurrency: int = 8
    # Truncated raw-content snippet fed to the digest writer alongside the
    # LLM-generated summary, for per-article framing that isn't lossy.
    # Trimmed at a word boundary when the extracted content exceeds this
    # many characters.
    content_head_max_chars: int = 2000

    # -- Topic classifier (BART-MNLI zero-shot NLI) --
    topic_model: str = "facebook/bart-large-mnli"
    topic_revision: str = (
        "d7645e127eaf1aefc7862fd59a17a5aa8558b8ce"  # pragma: allowlist secret  # gitleaks:allow
    )
    topic_threshold: float = 0.5
    topic_max_length: int = 512

    # -- Writer (digest map-phase section drafts) --
    writer_model: str = "gemma4:31b-cloud"
    writer_temperature: float = 0.4
    writer_top_p: float | None = None
    writer_repetition_penalty: float | None = None
    writer_seed: int | None = None
    writer_section_batch_size: int = 30

    # -- Editorial (per-section prose cleanup pass) --
    editorial_model: str = "gpt-oss:120b-cloud"
    editorial_temperature: float = 0.4
    editorial_top_p: float | None = None
    editorial_repetition_penalty: float | None = None
    editorial_seed: int | None = None

    # -- Framer (digest title + intro lede): set to None to fall back to writer_model --
    framer_model: str | None = "gpt-oss:120b-cloud"
    framer_temperature: float = 0.4
    framer_top_p: float | None = None
    framer_repetition_penalty: float | None = None
    framer_seed: int | None = None

    # -- Watcher (cross-section what-to-watch): set to None to fall back to writer_model --
    watcher_model: str | None = "gpt-oss:120b-cloud"
    watcher_temperature: float = 0.4
    watcher_top_p: float | None = None
    watcher_repetition_penalty: float | None = None
    watcher_seed: int | None = None

    # -- Clusterer (story-cluster pre-stage, cross-section dedup): set to None to fall back to writer_model --
    clusterer_model: str | None = "gpt-oss:120b-cloud"
    clusterer_temperature: float = 0.4
    clusterer_top_p: float | None = None
    clusterer_repetition_penalty: float | None = None
    clusterer_seed: int | None = None

    # -- Audio (TTS narration: opt-in terminal stage after the digest composer) --
    # Gates the audio renderer. Default off so `digest-generator run` stays cheap for
    # dev iteration; pass --audio explicitly to enable it.
    audio_enabled: bool = False
    # Piper voice id (e.g. ``en_US-amy-medium``). Resolved to the matching
    # path under the official ``rhasspy/piper-voices`` HuggingFace repo on
    # first use.
    audio_voice_model: str = "en_US-amy-medium"
    # Pinned revision of the ``rhasspy/piper-voices`` repo. Pinning to a
    # specific commit (or tag) protects against upstream model swaps
    # silently changing narration output, the same reasoning as
    # ``topic_revision`` for the BART-MNLI classifier.
    audio_voice_revision: str = (
        "375a0fe641dea077c2a47b4e9a056d6da521eed3"  # pragma: allowlist secret  # gitleaks:allow
    )
    # Opus encode bitrate. 24 kbps is enough for spoken-word mono.
    audio_bitrate_kbps: int = 24
    # Native Piper sample rate. Voice-model dependent ("medium" voices
    # ship at 22050 Hz). Used for sanity-checking the loaded voice config.
    audio_sample_rate: int = 22050
    # On-disk cache for downloaded Piper voice files. Separate root from
    # the HuggingFace transformer cache so voice ONNX files don't coexist
    # with model weights.
    audio_voice_cache: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "digest_generator" / "piper-voices",
    )
    # Override for the ffmpeg binary. Resolved via shutil.which when left
    # as the bare name; useful for pinning a specific build.
    audio_ffmpeg_path: str = "ffmpeg"
    # Piper's --sentence-silence flag: pause duration (seconds) inserted
    # at every sentence terminator (.!?). Headings and paragraphs end
    # with a period, so this is the main pacing knob. Raise for slower /
    # more deliberate narration; lower for tighter pacing.
    audio_sentence_silence_s: float = 0.4

    # =========================================================================
    # Pipeline-level
    # =========================================================================

    days_back: int = 7
    output_dir: str = "output"

    # =========================================================================
    # Logging
    # =========================================================================

    log_level_console: str = "INFO"
    log_level_file: str = "DEBUG"
    log_dir: str = "logs"
    # Global fallback sink (catches pre-run/ambient lines). Size-based rotation
    # because the primary log artifact is per-run ({run_dir}/run.log).
    log_rotation: str = "20 MB"
    log_retention: int = 5


settings = Settings()
