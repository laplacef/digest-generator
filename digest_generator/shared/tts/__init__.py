"""TTS infrastructure: voice configuration, lazy registry, and engine wrapper.

Mirror of ``digest_generator.shared.transformers`` (in-process HF / PyTorch) and
``digest_generator.shared.llm`` (Ollama HTTP). All three cluster engine-specific
model-serving infrastructure that's shared across ``core/`` consumers:
``llm/`` for Ollama clients used by the summarizer and digest stages,
``transformers/`` for HuggingFace transformer models used by the topic
classifier, and ``tts/`` for Piper-style text-to-speech used by the
audio renderer.

- ``types``: ``VoiceConfig``, a dataclass holding voice id, on-disk model
  path, and native sample rate.
- ``registry``: ``VoiceRegistry`` + ``voice_registry`` singleton, with lazy
  ``@cached_property`` access to pinned voices wired against
  ``settings.audio_voice_model`` (download and caching handled on first use).
- ``engine``: subprocess pipeline wrapping ``piper`` and ``ffmpeg``. The
  process boundary is the swap point: replacing Piper with another engine
  changes ``engine.py`` only, not ``core/audio/``.
"""
