"""Infrastructure types shared across the application.

Holds device configuration and HuggingFace model configuration. The
domain vocabulary (``Label``, ``TopicType``, ``Filter``, ``Entry``,
``Summary``) lives in ``digest_generator.core.types``.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import torch
from transformers import AutoTokenizer

__all__ = [
    "DeviceType",
    "ModelConfig",
]


class DeviceType(StrEnum):
    """PyTorch compute device targets."""

    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


@dataclass
class ModelConfig:
    """HuggingFace transformer model configuration with chainable loading.

    Holds model identity (name, revision, device) and, after ``load()``,
    the hydrated tokenizer and model objects. Only for HuggingFace models;
    API-based models (e.g., Ollama) manage their own configuration.
    """

    model_name: str
    revision: str
    device: DeviceType
    tokenizer: Any = field(default=None, repr=False)
    model: Any = field(default=None, repr=False)
    torch_dtype: torch.dtype | None = None

    def load(
        self,
        model_cls: type[Any],
        *,
        tokenizer_cls: type[Any] = AutoTokenizer,
        eval_mode: bool = True,
    ) -> "ModelConfig":
        """Load tokenizer and model from HuggingFace, moving to the configured device.

        Args:
            model_cls: HuggingFace model class (e.g., ``AutoModelForSeq2SeqLM``).
            tokenizer_cls: Tokenizer class to use. Defaults to ``AutoTokenizer``.
            eval_mode: Whether to set the model to evaluation mode.

        Returns:
            Self, for chaining: ``ModelConfig(...).load(cls)``.
        """
        self.tokenizer = tokenizer_cls.from_pretrained(
            self.model_name,
            revision=self.revision,
            use_fast=True,
        )

        # Optional dtype (e.g., torch.float16 on GPU later)
        model_kwargs: dict[str, object] = {"revision": self.revision}
        if self.torch_dtype is not None:
            model_kwargs["torch_dtype"] = self.torch_dtype

        self.model = model_cls.from_pretrained(self.model_name, **model_kwargs)
        self.model.to(self.device.value)

        if eval_mode:
            self.model.eval()

        return self

    @property
    def device_str(self) -> str:
        """Return the device name as a plain string (e.g., ``'cpu'``, ``'cuda'``)."""
        return self.device.value
