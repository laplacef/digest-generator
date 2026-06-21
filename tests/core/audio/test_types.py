"""Tests for digest_generator/core/audio/types.py: NarrationScript, AudioArtifact."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from digest_generator.core.audio.types import AudioArtifact, NarrationScript


class TestNarrationScript:
    def test_char_count(self):
        s = NarrationScript(text="hello world")
        assert s.char_count == 11

    def test_char_count_with_ssml(self):
        s = NarrationScript(text='Title. <break time="700ms"/> Body.')
        # SSML tags are characters too, so caller sees the raw script length.
        assert s.char_count == len(s.text)

    def test_frozen(self):
        s = NarrationScript(text="hi")
        with pytest.raises(FrozenInstanceError):
            s.text = "other"  # type: ignore[misc]


class TestAudioArtifact:
    def test_construction(self):
        a = AudioArtifact(
            opus_path=Path("/tmp/x.opus"),
            voice_id="en_US-amy-medium",
            bitrate_kbps=24,
            narration_chars=1000,
            audio_bytes=50000,
            audio_duration_s=16.7,
            cached=False,
        )
        assert a.opus_path == Path("/tmp/x.opus")
        assert a.cached is False

    def test_frozen(self):
        a = AudioArtifact(
            opus_path=Path("/tmp/x.opus"),
            voice_id="v",
            bitrate_kbps=24,
            narration_chars=0,
            audio_bytes=0,
            audio_duration_s=0.0,
            cached=True,
        )
        with pytest.raises(FrozenInstanceError):
            a.cached = False  # type: ignore[misc]
