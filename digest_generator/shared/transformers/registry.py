"""Lazy-loaded registry of pinned HuggingFace models.

``model_registry`` loads each model on first access through
``@cached_property``, taking names, revisions, and device from ``Settings``,
so a caller that injects its own ``ModelConfig`` never loads the default.
The topic classifier's GLiClass model is the only one.
"""

from functools import cached_property

from gliclass import GLiClassModel

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
        ).load(GLiClassModel)


model_registry = ModelRegistry()
