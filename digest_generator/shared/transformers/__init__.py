"""HuggingFace transformer serving: in-process PyTorch via the transformers library.

Mirror of ``digest_generator.shared.llm`` (Ollama HTTP) and ``digest_generator.shared.tts``
(Piper subprocess). All three cluster engine-specific model-serving
infrastructure that's shared across ``core/`` consumers: ``llm/`` for
Ollama clients used by the summarizer and digest stages,
``transformers/`` for HuggingFace transformer models loaded in-process
(the topic classifier, with additional label stages such as sentiment
and entailment slotting in as siblings), and ``tts/`` for Piper
text-to-speech used by the audio renderer.

HuggingFace Hub authentication is shared across both `transformers/`
and `tts/` and lives at ``digest_generator.shared.hf_hub`` (a sibling module,
not inside this package), because Hub auth is orthogonal to the serving
engine consuming the downloaded artifact.

- ``types``: ``DeviceType``, ``ModelConfig``, the device enum and the
  HF model-config dataclass with chainable ``load()``.
- ``registry``: ``ModelRegistry`` + ``model_registry`` singleton, with lazy
  ``@cached_property`` access to pinned HF models (``topic`` for
  BART-MNLI, with ``sentiment`` and ``entailment`` slotting in as siblings).
"""
