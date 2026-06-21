"""Audio renderer: markdown digest to Piper TTS to Opus artifact.

Peer to ``digest_generator.core.digest``. Domain pipeline that converts the
composed digest markdown into a narrated audio file. Engine concerns
(Piper subprocess, voice download, ffmpeg encode) live in
``digest_generator.shared.tts``; this package owns the markdown-to-speech
narration pre-pass and the cache-key and output-path logic.

The narration pre-pass (``narration.markdown_to_narration``) turns digest
markdown into a speech-friendly script. The renderer, io, and typed
artifacts build on top of it.
"""
