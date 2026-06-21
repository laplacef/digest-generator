"""Tests for digest_generator/core/audio/renderer.py: AudioRenderer orchestration and cache."""

from pathlib import Path
from unittest.mock import patch

import pytest

from digest_generator.core.audio.io import cache_key_path, compute_cache_key
from digest_generator.core.audio.renderer import AudioRenderer
from digest_generator.shared.tts.types import VoiceConfig


@pytest.fixture
def voice():
    return VoiceConfig(
        voice_id="en_US-amy-medium",
        model_path=Path("/fake/amy.onnx"),
        sample_rate=22050,
    )


@pytest.fixture
def digest_md(tmp_path):
    path = tmp_path / "2026-05-11-weekly-ai-digest.md"
    path.write_text("# Weekly AI Digest\n\nBody paragraph.\n")
    return path


def _fake_synthesize_factory(payload: bytes):
    """Return a fake synthesize that writes ``payload`` to the requested path."""

    def fake(_text, _voice, out_path, **_kw):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(payload)
        return out_path

    return fake


class TestFreshRender:
    def test_writes_opus_at_expected_path(self, tmp_path, voice, digest_md):
        renderer = AudioRenderer(voice=voice, bitrate_kbps=24, overrides={})
        with patch(
            "digest_generator.core.audio.renderer.synthesize",
            side_effect=_fake_synthesize_factory(b"opus" * 100),
        ):
            artifact = renderer.render(tmp_path, digest_md)
        expected = tmp_path / "audio" / "2026-05-11-weekly-ai-digest.opus"
        assert artifact.opus_path == expected
        assert expected.exists()
        assert artifact.cached is False
        assert artifact.audio_bytes == 400

    def test_writes_cache_key(self, tmp_path, voice, digest_md):
        renderer = AudioRenderer(voice=voice, bitrate_kbps=24, overrides={})
        with patch(
            "digest_generator.core.audio.renderer.synthesize",
            side_effect=_fake_synthesize_factory(b"x"),
        ):
            renderer.render(tmp_path, digest_md)
        key_path = cache_key_path(tmp_path)
        assert key_path.exists()
        expected_key = compute_cache_key(digest_md.read_bytes(), "en_US-amy-medium", 24)
        assert key_path.read_text().strip() == expected_key

    def test_narration_chars_populated(self, tmp_path, voice, digest_md):
        renderer = AudioRenderer(voice=voice, bitrate_kbps=24, overrides={})
        with patch(
            "digest_generator.core.audio.renderer.synthesize",
            side_effect=_fake_synthesize_factory(b"x"),
        ):
            artifact = renderer.render(tmp_path, digest_md)
        assert artifact.narration_chars > 0

    def test_voice_and_bitrate_reported(self, tmp_path, voice, digest_md):
        renderer = AudioRenderer(voice=voice, bitrate_kbps=32, overrides={})
        with patch(
            "digest_generator.core.audio.renderer.synthesize",
            side_effect=_fake_synthesize_factory(b"x"),
        ):
            artifact = renderer.render(tmp_path, digest_md)
        assert artifact.voice_id == "en_US-amy-medium"
        assert artifact.bitrate_kbps == 32

    def test_passes_engine_overrides(self, tmp_path, voice, digest_md):
        renderer = AudioRenderer(
            voice=voice,
            bitrate_kbps=24,
            sentence_silence_s=0.6,
            piper_path="/bin/piper-custom",
            ffmpeg_path="/bin/ffmpeg-custom",
            overrides={},
        )
        captured = {}

        def fake(_text, _voice, out_path, **kwargs):
            captured.update(kwargs)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"x")
            return out_path

        with patch("digest_generator.core.audio.renderer.synthesize", side_effect=fake):
            renderer.render(tmp_path, digest_md)

        assert captured["piper_path"] == "/bin/piper-custom"
        assert captured["ffmpeg_path"] == "/bin/ffmpeg-custom"
        assert captured["bitrate_kbps"] == 24
        assert captured["sentence_silence_s"] == 0.6

    def test_default_sentence_silence_is_none(self, tmp_path, voice, digest_md):
        """``sentence_silence_s=None`` lets Piper use its compiled default."""
        renderer = AudioRenderer(voice=voice, bitrate_kbps=24, overrides={})
        captured = {}

        def fake(_text, _voice, out_path, **kwargs):
            captured.update(kwargs)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"x")
            return out_path

        with patch("digest_generator.core.audio.renderer.synthesize", side_effect=fake):
            renderer.render(tmp_path, digest_md)

        assert captured["sentence_silence_s"] is None


class TestCacheHit:
    def test_second_render_skips_synthesis(self, tmp_path, voice, digest_md):
        renderer = AudioRenderer(voice=voice, bitrate_kbps=24, overrides={})
        with patch(
            "digest_generator.core.audio.renderer.synthesize",
            side_effect=_fake_synthesize_factory(b"opus" * 100),
        ) as mock_synth:
            renderer.render(tmp_path, digest_md)
            renderer.render(tmp_path, digest_md)
        assert mock_synth.call_count == 1

    def test_cache_hit_reports_cached_true(self, tmp_path, voice, digest_md):
        renderer = AudioRenderer(voice=voice, bitrate_kbps=24, overrides={})
        with patch(
            "digest_generator.core.audio.renderer.synthesize",
            side_effect=_fake_synthesize_factory(b"x"),
        ):
            renderer.render(tmp_path, digest_md)
            artifact = renderer.render(tmp_path, digest_md)
        assert artifact.cached is True
        assert artifact.narration_chars == 0  # not computed on cache hit
        assert artifact.audio_bytes > 0  # read from disk

    def test_cache_miss_on_md_change(self, tmp_path, voice, digest_md):
        renderer = AudioRenderer(voice=voice, bitrate_kbps=24, overrides={})
        with patch(
            "digest_generator.core.audio.renderer.synthesize",
            side_effect=_fake_synthesize_factory(b"x"),
        ) as mock_synth:
            renderer.render(tmp_path, digest_md)
            digest_md.write_text("# Different content\n\nNew body.\n")
            renderer.render(tmp_path, digest_md)
        assert mock_synth.call_count == 2

    def test_cache_miss_on_voice_change(self, tmp_path, voice, digest_md):
        renderer_a = AudioRenderer(voice=voice, bitrate_kbps=24, overrides={})
        voice_b = VoiceConfig(
            voice_id="en_GB-alan-medium",
            model_path=Path("/fake/alan.onnx"),
            sample_rate=22050,
        )
        renderer_b = AudioRenderer(voice=voice_b, bitrate_kbps=24, overrides={})
        with patch(
            "digest_generator.core.audio.renderer.synthesize",
            side_effect=_fake_synthesize_factory(b"x"),
        ) as mock_synth:
            renderer_a.render(tmp_path, digest_md)
            renderer_b.render(tmp_path, digest_md)
        assert mock_synth.call_count == 2

    def test_cache_miss_on_bitrate_change(self, tmp_path, voice, digest_md):
        renderer_a = AudioRenderer(voice=voice, bitrate_kbps=24, overrides={})
        renderer_b = AudioRenderer(voice=voice, bitrate_kbps=48, overrides={})
        with patch(
            "digest_generator.core.audio.renderer.synthesize",
            side_effect=_fake_synthesize_factory(b"x"),
        ) as mock_synth:
            renderer_a.render(tmp_path, digest_md)
            renderer_b.render(tmp_path, digest_md)
        assert mock_synth.call_count == 2

    def test_cache_miss_on_sentence_silence_change(self, tmp_path, voice, digest_md):
        renderer_a = AudioRenderer(
            voice=voice, bitrate_kbps=24, sentence_silence_s=0.4, overrides={}
        )
        renderer_b = AudioRenderer(
            voice=voice, bitrate_kbps=24, sentence_silence_s=0.6, overrides={}
        )
        with patch(
            "digest_generator.core.audio.renderer.synthesize",
            side_effect=_fake_synthesize_factory(b"x"),
        ) as mock_synth:
            renderer_a.render(tmp_path, digest_md)
            renderer_b.render(tmp_path, digest_md)
        assert mock_synth.call_count == 2

    def test_missing_opus_invalidates_cache(self, tmp_path, voice, digest_md):
        """Cache key present but opus deleted triggers re-render."""
        renderer = AudioRenderer(voice=voice, bitrate_kbps=24, overrides={})
        with patch(
            "digest_generator.core.audio.renderer.synthesize",
            side_effect=_fake_synthesize_factory(b"x"),
        ) as mock_synth:
            artifact = renderer.render(tmp_path, digest_md)
            artifact.opus_path.unlink()
            renderer.render(tmp_path, digest_md)
        assert mock_synth.call_count == 2


class TestOverridesResolution:
    def test_explicit_empty_dict_disables_overrides(self, tmp_path, voice, digest_md):
        """Passing ``overrides={}`` must not load the bundled YAML defaults."""
        renderer = AudioRenderer(voice=voice, bitrate_kbps=24, overrides={})
        digest_md.write_text("# Discuss AGI here.\n")

        captured_text = {}

        def fake(text, _voice, out_path, **_kw):
            captured_text["text"] = text
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"x")
            return out_path

        with patch("digest_generator.core.audio.renderer.synthesize", side_effect=fake):
            renderer.render(tmp_path, digest_md)

        assert "AGI" in captured_text["text"]
        assert "A G I" not in captured_text["text"]

    def test_none_uses_bundled_overrides(self, tmp_path, voice, digest_md):
        """``overrides=None`` (the default) falls back to the bundled YAML."""
        renderer = AudioRenderer(voice=voice, bitrate_kbps=24)
        digest_md.write_text("# Discuss AGI here.\n")

        captured_text = {}

        def fake(text, _voice, out_path, **_kw):
            captured_text["text"] = text
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(b"x")
            return out_path

        with patch("digest_generator.core.audio.renderer.synthesize", side_effect=fake):
            renderer.render(tmp_path, digest_md)

        assert "A G I" in captured_text["text"]
