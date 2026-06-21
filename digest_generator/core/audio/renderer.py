"""Audio renderer: orchestrates narration, synthesis, encode, and cache.

Sits between ``digest_generator.core.audio.narration`` (markdown to
speech-friendly script) and ``digest_generator.shared.tts.engine`` (Piper
subprocess plus ffmpeg encode). Cache-aware via
``digest_generator.core.audio.io.compute_cache_key``: if the digest markdown,
voice id, and bitrate all match the previous render, the existing ``.opus``
is left untouched.

The renderer is a class so the caller (``api.render_audio``) can hold one
configured instance across multiple renders. Voice and engine path overrides
come in at construction time, which keeps it dependency-injection friendly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from digest_generator.core.audio.io import (
    audio_dir,
    compute_cache_key,
    opus_path_for_digest,
    read_cache_key,
    write_cache_key,
)
from digest_generator.core.audio.narration import load_overrides, markdown_to_narration
from digest_generator.core.audio.types import AudioArtifact, NarrationScript
from digest_generator.shared.logging import logger
from digest_generator.shared.tts.engine import estimate_audio_duration_s, synthesize
from digest_generator.shared.tts.types import VoiceConfig

__all__ = [
    "AudioRenderer",
]


@dataclass
class AudioRenderer:
    """Render a digest markdown file to a cached Opus artifact.

    Attributes:
        voice: Loaded ``VoiceConfig`` for the synthesizer.
        bitrate_kbps: Opus target bitrate; participates in the cache key.
        sentence_silence_s: Piper ``--sentence-silence`` value (seconds);
            ``None`` keeps Piper's compiled default. Participates in the
            cache key so changes auto-invalidate prior renders.
        piper_path: Override for the Piper binary path.
        ffmpeg_path: Override for the ffmpeg binary path.
        overrides: Pronunciation overrides applied during narration.
            ``None`` falls back to the bundled YAML.
    """

    voice: VoiceConfig
    bitrate_kbps: int
    sentence_silence_s: float | None = None
    piper_path: str = "piper"
    ffmpeg_path: str = "ffmpeg"
    overrides: dict[str, str] | None = None
    _resolved_overrides: dict[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._resolved_overrides = (
            self.overrides if self.overrides is not None else load_overrides()
        )

    def render(self, run_dir: Path, digest_md_path: Path) -> AudioArtifact:
        """Render the digest to ``run_dir/audio/`` and return an artifact.

        Cache hit (matching ``cache_key.txt`` + existing ``.opus``) returns
        immediately with ``cached=True``. Cache miss re-narrates, re-synthesizes,
        re-encodes, and writes the new cache key.
        """
        md_bytes = digest_md_path.read_bytes()
        key = compute_cache_key(
            md_bytes,
            self.voice.voice_id,
            self.bitrate_kbps,
            sentence_silence_s=self.sentence_silence_s,
        )
        out_path = opus_path_for_digest(run_dir, digest_md_path)

        cached = self._try_cache(run_dir, key, out_path)
        if cached is not None:
            return cached

        return self._render_fresh(run_dir, key, md_bytes, out_path)

    def _try_cache(
        self,
        run_dir: Path,
        key: str,
        out_path: Path,
    ) -> AudioArtifact | None:
        """Return an artifact on cache hit; ``None`` if a fresh render is needed."""
        existing = read_cache_key(run_dir)
        if existing != key or not out_path.exists():
            return None
        byte_size = out_path.stat().st_size
        logger.info(
            "audio cache hit: {} ({} bytes)",
            out_path.name,
            byte_size,
        )
        return AudioArtifact(
            opus_path=out_path,
            voice_id=self.voice.voice_id,
            bitrate_kbps=self.bitrate_kbps,
            narration_chars=0,
            audio_bytes=byte_size,
            audio_duration_s=estimate_audio_duration_s(byte_size, self.bitrate_kbps),
            cached=True,
        )

    def _render_fresh(
        self,
        run_dir: Path,
        key: str,
        md_bytes: bytes,
        out_path: Path,
    ) -> AudioArtifact:
        """Run narration, then Piper, then ffmpeg, write the cache key, return artifact."""
        audio_dir(run_dir).mkdir(parents=True, exist_ok=True)
        narration = NarrationScript(
            text=markdown_to_narration(
                md_bytes.decode("utf-8"),
                overrides=self._resolved_overrides,
            )
        )
        logger.info(
            "audio render: {} ({} narration chars, voice={}, bitrate={}kbps)",
            out_path.name,
            narration.char_count,
            self.voice.voice_id,
            self.bitrate_kbps,
        )
        synthesize(
            narration.text,
            self.voice,
            out_path,
            bitrate_kbps=self.bitrate_kbps,
            piper_path=self.piper_path,
            ffmpeg_path=self.ffmpeg_path,
            sentence_silence_s=self.sentence_silence_s,
        )
        write_cache_key(run_dir, key)
        byte_size = out_path.stat().st_size
        return AudioArtifact(
            opus_path=out_path,
            voice_id=self.voice.voice_id,
            bitrate_kbps=self.bitrate_kbps,
            narration_chars=narration.char_count,
            audio_bytes=byte_size,
            audio_duration_s=estimate_audio_duration_s(byte_size, self.bitrate_kbps),
            cached=False,
        )
