"""Lazy-loaded model registry for HuggingFace transformer models.

Provides a singleton ``model_registry`` whose properties load models on first
access via ``@cached_property``. If custom models are injected via DI in the
pipeline, the defaults here are never loaded.

Model names, revisions, and device are configured via ``Settings``.

Scope: only the topic stage's ``TopicClassifier`` still uses a HuggingFace
model today. Future label stages (sentiment, entailment) will add their own
properties here when they land. The article summarizer was migrated to an
LLM call (Ollama) and no longer appears here; see
``digest_generator.shared.llm.clients.ClientRegistry`` for the Ollama client
singleton it shares with the digest stages.
"""

from functools import cached_property

from transformers import BartForSequenceClassification

from digest_generator.shared.logging import logger
from digest_generator.shared.settings import settings
from digest_generator.shared.transformers.types import DeviceType, ModelConfig


class ModelRegistry:
    """Registry of pre-configured HuggingFace models, loaded lazily on first access."""

    @cached_property
    def topic(self) -> ModelConfig:
        """Load and cache the zero-shot topic classification model."""
        logger.info("Loading topic model: {}", settings.topic_model)
        return ModelConfig(
            model_name=settings.topic_model,
            revision=settings.topic_revision,
            device=DeviceType(settings.device),
        ).load(BartForSequenceClassification)


model_registry = ModelRegistry()
