"""Tests for digest_generator/core/label/stages/topic.py: TopicClassifier with mocked model.

Mocks the tokenizer and model to test NLI-based classification flow:
hypothesis generation, then model inference, then softmax, then tag assignment.
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from digest_generator.core.label import TopicClassifier
from digest_generator.core.types import Entry, Label, Summary, TopicType
from digest_generator.shared.transformers.types import DeviceType, ModelConfig

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_model_config():
    config = ModelConfig(
        model_name="test/model",
        revision="abc",
        device=DeviceType.CPU,
    )
    config.tokenizer = MagicMock()
    config.model = MagicMock()
    return config


@pytest.fixture
def classifier(mock_model_config):
    return TopicClassifier(model_config=mock_model_config)


@pytest.fixture
def sample_summary():
    now = datetime.now(tz=UTC)
    entry = Entry(
        title="GPT-5 Released",
        url="https://example.com/gpt5",
        origin="openai",
        published=now,
        description="OpenAI releases GPT-5",
        content="Full content about GPT-5 release.",
    )
    return Summary(entry=entry, summary="OpenAI released GPT-5.", length=22, topics=[])


def _make_logits(num_labels, high_indices, high_value=5.0, low_value=-5.0):
    """Create fake logits tensor with high entailment scores at given indices.

    Shape: (num_labels, 3) where columns are [contradiction, neutral, entailment].
    """
    logits = torch.full((num_labels, 3), low_value)
    for idx in high_indices:
        logits[idx, 2] = high_value  # entailment column
    return logits


# =============================================================================
# Tests
# =============================================================================


class TestTopicClassifier:
    def test_sets_eval_mode(self, mock_model_config):
        TopicClassifier(model_config=mock_model_config)
        mock_model_config.model.eval.assert_called_once()

    def test_builds_label_list_from_tag_type(self, classifier):
        assert classifier.labels == [t.value for t in TopicType]
        assert len(classifier.labels) == len(TopicType)

    def test_assigns_tags_above_threshold(self, classifier, sample_summary):
        """Labels with entailment probability above threshold should be assigned."""
        num_labels = len(TopicType)
        # Make LLM and MODEL_RELEASE score high
        llm_idx = list(TopicType).index(TopicType.LLM)
        release_idx = list(TopicType).index(TopicType.MODEL_RELEASE)
        logits = _make_logits(num_labels, [llm_idx, release_idx])

        classifier.model.return_value = SimpleNamespace(logits=logits)
        classifier.tokenizer.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }

        results = classifier.classify_summaries([sample_summary], threshold=0.5)

        tag_types = {t.value for t in results[0].topics}
        assert TopicType.LLM in tag_types
        assert TopicType.MODEL_RELEASE in tag_types

    def test_assigns_top_one_if_none_above_threshold(self, classifier, sample_summary):
        """If no label exceeds threshold, the highest-scoring one is still assigned."""
        num_labels = len(TopicType)
        # All scores low, but LLM slightly less low
        logits = torch.full((num_labels, 3), -5.0)
        llm_idx = list(TopicType).index(TopicType.LLM)
        logits[llm_idx, 2] = -1.0  # Still below threshold after softmax, but highest

        classifier.model.return_value = SimpleNamespace(logits=logits)
        classifier.tokenizer.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }

        results = classifier.classify_summaries([sample_summary], threshold=0.99)

        assert len(results[0].topics) >= 1
        # The top-1 tag should be assigned regardless of threshold

    def test_tags_sorted_by_confidence(self, classifier, sample_summary):
        """Tags should be sorted by confidence descending."""
        num_labels = len(TopicType)
        llm_idx = list(TopicType).index(TopicType.LLM)
        safety_idx = list(TopicType).index(TopicType.SAFETY)
        # LLM higher than SAFETY
        logits = torch.full((num_labels, 3), -5.0)
        logits[llm_idx, 2] = 6.0
        logits[safety_idx, 2] = 4.0

        classifier.model.return_value = SimpleNamespace(logits=logits)
        classifier.tokenizer.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }

        results = classifier.classify_summaries([sample_summary], threshold=0.5)

        confidences = [t.confidence for t in results[0].topics]
        assert confidences == sorted(confidences, reverse=True)

    def test_tags_are_tag_objects(self, classifier, sample_summary):
        """Each tag should be a Label dataclass with TopicType and float confidence."""
        num_labels = len(TopicType)
        llm_idx = list(TopicType).index(TopicType.LLM)
        logits = _make_logits(num_labels, [llm_idx])

        classifier.model.return_value = SimpleNamespace(logits=logits)
        classifier.tokenizer.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }

        results = classifier.classify_summaries([sample_summary])

        for tag in results[0].topics:
            assert isinstance(tag, Label)
            assert isinstance(tag.value, TopicType)
            assert isinstance(tag.confidence, float)

    def test_processes_multiple_summaries(self, classifier, sample_summary):
        """Should process each summary independently."""
        num_labels = len(TopicType)
        llm_idx = list(TopicType).index(TopicType.LLM)
        logits = _make_logits(num_labels, [llm_idx])

        classifier.model.return_value = SimpleNamespace(logits=logits)
        classifier.tokenizer.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }

        # Create a second summary
        entry2 = Entry(
            title="Test",
            url="https://example.com/2",
            origin="github",
            published=datetime.now(tz=UTC),
            description="desc",
            content="content",
        )
        summary2 = Summary(entry=entry2, summary="text", length=4, topics=[])

        results = classifier.classify_summaries([sample_summary, summary2])
        assert len(results) == 2
        assert len(results[0].topics) >= 1
        assert len(results[1].topics) >= 1

    def test_empty_summaries(self, classifier):
        assert classifier.classify_summaries([]) == []

    def test_hypothesis_format(self, classifier, sample_summary):
        """Hypotheses should follow 'This article is about {label}.' format."""
        num_labels = len(TopicType)
        logits = _make_logits(num_labels, [0])

        classifier.model.return_value = SimpleNamespace(logits=logits)
        classifier.tokenizer.return_value = {
            "input_ids": torch.tensor([[1]]),
            "attention_mask": torch.tensor([[1]]),
        }

        classifier.classify_summaries([sample_summary])

        # Check the tokenizer was called with text-hypothesis pairs
        call_args = classifier.tokenizer.call_args
        hypotheses = call_args[0][1]
        assert all(h.startswith("This article is about ") for h in hypotheses)
        assert all(h.endswith(".") for h in hypotheses)
