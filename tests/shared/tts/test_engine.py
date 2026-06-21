"""Tests for digest_generator/shared/tts/engine.py: subprocess wrappers + duration helper.

Subprocess paths (``synthesize_to_wav`` / ``encode_opus``) require real
piper / ffmpeg binaries to be meaningful; these tests cover the smaller, pure
helpers here and rely on the renderer-level tests to exercise the
subprocess plumbing via mocks.
"""

import pytest

from digest_generator.shared.tts.engine import (
    MissingBinaryError,
    encode_opus,
    estimate_audio_duration_s,
    synthesize_to_wav,
)
from digest_generator.shared.tts.types import VoiceConfig


class TestEstimateAudioDuration:
    def test_24kbps_one_second(self):
        # 24 kbps = 24_000 bits per second = 3_000 bytes per second.
        assert estimate_audio_duration_s(3_000, 24) == pytest.approx(1.0)

    def test_zero_bytes(self):
        assert estimate_audio_duration_s(0, 24) == 0.0

    def test_negative_bytes(self):
        assert estimate_audio_duration_s(-5, 24) == 0.0

    def test_zero_bitrate(self):
        assert estimate_audio_duration_s(1000, 0) == 0.0

    def test_realistic_digest(self):
        # ~1.2 MB at 24 kbps ≈ ~6:40 of audio.
        seconds = estimate_audio_duration_s(1_200_000, 24)
        assert 390 < seconds < 410


class TestMissingBinary:
    def test_synthesize_raises_when_piper_absent(self, tmp_path):
        voice = VoiceConfig(voice_id="v", model_path=tmp_path / "voice.onnx", sample_rate=22050)
        with pytest.raises(MissingBinaryError, match="piper"):
            synthesize_to_wav("hi", voice, piper_path="/no/such/binary")

    def test_encode_opus_raises_when_ffmpeg_absent(self, tmp_path):
        with pytest.raises(MissingBinaryError, match="ffmpeg"):
            encode_opus(
                b"wav",
                tmp_path / "out.opus",
                bitrate_kbps=24,
                ffmpeg_path="/no/such/binary",
            )
