"""HuggingFace models run in-process with PyTorch.

One of three serving-engine packages, beside ``shared.llm`` (Ollama over
HTTP) and ``shared.tts`` (Piper subprocess). Hub authentication lives in
``shared.hf_hub`` because ``tts`` downloads from the Hub too.

- ``types``: ``DeviceType`` and ``ModelConfig``, the HF model config with a
  chainable ``load()``.
- ``registry``: the ``model_registry`` singleton, whose ``topic`` property
  loads the pinned GLiClass topic model on first access.
"""
