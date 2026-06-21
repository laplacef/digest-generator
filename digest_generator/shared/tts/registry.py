"""Lazy-loaded voice registry for Piper TTS.

Provides a singleton ``voice_registry`` whose ``@cached_property`` access
downloads and hydrates a ``VoiceConfig`` the first time it's referenced.
Mirror of ``digest_generator.shared.transformers.registry.ModelRegistry`` (HuggingFace)
and ``digest_generator.shared.llm.clients.ClientRegistry`` (Ollama).

Voice files live in the official ``rhasspy/piper-voices`` HuggingFace
repo; ``huggingface_hub.hf_hub_download`` handles the download,
ETag-based integrity verification, and local caching. The file
location is pinned to ``settings.audio_voice_cache`` (default
``~/.cache/digest_generator/piper-voices/``) via ``local_dir=`` so voice ONNX
files stay separate from the transformer-model cache.
"""

from functools import cached_property
from pathlib import Path

from huggingface_hub import hf_hub_download

from digest_generator.shared.logging import logger
from digest_generator.shared.settings import settings
from digest_generator.shared.tts.types import VoiceConfig

__all__ = [
    "VoiceRegistry",
    "voice_registry",
]

# Official Piper voices repo. Each voice ships as an ONNX model plus a
# sidecar JSON config, organized by ``<lang>/<lang_LOCALE>/<name>/<quality>/``.
_PIPER_VOICES_REPO = "rhasspy/piper-voices"


def _piper_hf_paths(voice_id: str) -> tuple[str, str]:
    """Map a Piper voice id to its ``(onnx_path, config_path)`` in the HF repo.

    Voice ids follow the convention ``<lang>_<COUNTRY>-<name>-<quality>``
    (e.g. ``en_US-amy-medium``). The repo layout mirrors that structure:
    ``en/en_US/amy/medium/en_US-amy-medium.onnx``.

    Raises:
        ValueError: ``voice_id`` doesn't match the Piper naming convention.
    """
    parts = voice_id.split("-")
    if len(parts) != 3:
        msg = (
            f"voice_id {voice_id!r} doesn't match Piper convention "
            "<lang>_<COUNTRY>-<name>-<quality>"
        )
        raise ValueError(msg)
    locale, name, quality = parts
    lang = locale.split("_")[0]
    base = f"{lang}/{locale}/{name}/{quality}"
    return (f"{base}/{voice_id}.onnx", f"{base}/{voice_id}.onnx.json")


def _download_voice(voice_id: str, cache_dir: Path, revision: str) -> Path:
    """Download a Piper voice's ONNX + sidecar JSON. Returns the ONNX path.

    Idempotent: ``hf_hub_download`` uses ETags to skip already-cached files,
    so re-running on the same cache directory is a no-op after first use.
    ``revision`` pins the repo commit so upstream model swaps can't silently
    change narration output.
    """
    onnx_path, config_path = _piper_hf_paths(voice_id)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Piper voice: {} -> {}", voice_id, cache_dir)
    # Bandit B615 only inspects string-literal `revision=` arguments and can't
    # see through the settings-backed variable. The revision is pinned at the
    # call site (see `audio_voice_revision` in shared/settings.py).
    onnx_local = hf_hub_download(  # nosec B615
        repo_id=_PIPER_VOICES_REPO,
        filename=onnx_path,
        revision=revision,
        local_dir=str(cache_dir),
    )
    hf_hub_download(  # nosec B615
        repo_id=_PIPER_VOICES_REPO,
        filename=config_path,
        revision=revision,
        local_dir=str(cache_dir),
    )
    return Path(onnx_local)


class VoiceRegistry:
    """Registry of pre-configured Piper voices, loaded lazily on first access."""

    @cached_property
    def default(self) -> VoiceConfig:
        """Load and cache the default narration voice from ``settings``."""
        voice_id = settings.audio_voice_model
        model_path = _download_voice(
            voice_id,
            settings.audio_voice_cache,
            settings.audio_voice_revision,
        )
        return VoiceConfig(
            voice_id=voice_id,
            model_path=model_path,
            sample_rate=settings.audio_sample_rate,
        )


voice_registry = VoiceRegistry()
