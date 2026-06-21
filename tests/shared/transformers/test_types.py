"""Tests for digest_generator/shared/transformers/types.py: DeviceType enum and ModelConfig."""

import torch

from digest_generator.shared.transformers.types import DeviceType, ModelConfig


class TestDeviceType:
    """CPU / CUDA / MPS: the compute targets the topic labeler can run on."""

    def test_values(self):
        assert DeviceType.CPU == "cpu"
        assert DeviceType.CUDA == "cuda"
        assert DeviceType.MPS == "mps"

    def test_count(self):
        assert len(DeviceType) == 3


class TestModelConfig:
    """Construction and properties (no real model loading; load() tested via mocks elsewhere)."""

    def test_construction(self):
        config = ModelConfig(
            model_name="facebook/bart-large-cnn",
            revision="abc123",
            device=DeviceType.CPU,
        )
        assert config.model_name == "facebook/bart-large-cnn"
        assert config.revision == "abc123"
        assert config.device == DeviceType.CPU

    def test_defaults_are_none(self):
        """Tokenizer and model should be None before load() is called."""
        config = ModelConfig(
            model_name="test-model",
            revision="abc",
            device=DeviceType.CPU,
        )
        assert config.tokenizer is None
        assert config.model is None
        assert config.torch_dtype is None

    def test_device_str_property(self):
        """device_str should return the string value of the device enum."""
        config = ModelConfig(
            model_name="test-model",
            revision="abc",
            device=DeviceType.CUDA,
        )
        assert config.device_str == "cuda"

    def test_torch_dtype_optional(self):
        """torch_dtype can be set for GPU optimization."""
        config = ModelConfig(
            model_name="test-model",
            revision="abc",
            device=DeviceType.CUDA,
            torch_dtype=torch.float16,
        )
        assert config.torch_dtype == torch.float16
