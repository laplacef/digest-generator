"""Tests for digest_generator/core/label/stages/topic.py: TopicClassifier with a fake pipeline.

The GLiClass pipeline is injected as a fake that returns one score per label,
so the tests cover label rendering, the score cut, the top-1 fallback,
ordering, and alignment without loading a model. Fakes and patches are
autospecced from ``gliclass.ZeroShotClassificationPipeline``, so a keyword
the installed gliclass doesn't accept fails here.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, create_autospec, patch

import pytest
import torch
from gliclass import ZeroShotClassificationPipeline

from digest_generator.core.label.stages import topic as topic_module
from digest_generator.core.label.stages.topic import TopicClassifier
from digest_generator.core.types import Entry, Label, TopicType
from digest_generator.shared.logging import collect_stage_telemetry
from digest_generator.shared.transformers.types import DeviceType, ModelConfig

# =============================================================================
# Fixtures
# =============================================================================


def _label_text(topic: TopicType) -> str:
    return topic.value.replace("-", " ")


def _fake_pipeline(*per_text: dict[TopicType, float], default: float = 0.01) -> MagicMock:
    """Pipeline stand-in returning every label's score for each text.

    ``per_text[i]`` sets text i's scores (the last mapping repeats for any
    further texts); labels not named score ``default``.
    """

    def run(texts, labels, threshold, batch_size):
        rows = []
        for i, _ in enumerate(texts):
            scores = per_text[min(i, len(per_text) - 1)] if per_text else {}
            by_text = {_label_text(t): s for t, s in scores.items()}
            rows.append([{"label": lab, "score": by_text.get(lab, default)} for lab in labels])
        return rows

    return create_autospec(ZeroShotClassificationPipeline, instance=True, side_effect=run)


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


def _classifier(model_config, pipeline=None):
    if pipeline is None:
        pipeline = _fake_pipeline()
    return TopicClassifier(model_config=model_config, pipeline=pipeline)


def _entry(n: int = 1, **fields) -> Entry:
    defaults = {
        "title": f"T{n}",
        "url": f"https://e.com/{n}",
        "origin": "f",
        "published": datetime.now(tz=UTC),
        "description": f"d{n}",
        "content": "body",
    }
    return Entry(**(defaults | fields))


# =============================================================================
# Tests
# =============================================================================


class TestTopicClassifier:
    def test_builds_label_list_from_topic_type(self, mock_model_config):
        classifier = _classifier(mock_model_config)
        assert classifier.labels == [t.value for t in TopicType]
        assert len(classifier.labels) == len(TopicType)

    def test_default_pipeline_wraps_model_config(self, mock_model_config):
        """Without an injected pipeline, one is built over the hydrated model."""
        with patch.object(topic_module, "ZeroShotClassificationPipeline", autospec=True) as cls:
            classifier = TopicClassifier(model_config=mock_model_config)
        cls.assert_called_once_with(
            mock_model_config.model,
            mock_model_config.tokenizer,
            max_length=topic_module.settings.topic_max_length,
            classification_type="multi-label",
            device=torch.device("cpu"),
            progress_bar=False,
        )
        assert classifier.pipeline is cls.return_value

    def test_default_pipeline_keeps_mps_device(self, mock_model_config):
        """The pipeline gets a torch.device, since it maps a non-cuda string to CPU."""
        mock_model_config.device = DeviceType.MPS
        with patch.object(topic_module, "ZeroShotClassificationPipeline", autospec=True) as cls:
            TopicClassifier(model_config=mock_model_config)
        assert cls.call_args.kwargs["device"] == torch.device("mps")

    def test_keeps_a_falsy_injected_pipeline(self, mock_model_config):
        """An injected pipeline is used even when it is falsy."""
        pipeline = MagicMock()
        pipeline.__bool__.return_value = False
        with patch.object(topic_module, "ZeroShotClassificationPipeline", autospec=True) as cls:
            classifier = TopicClassifier(model_config=mock_model_config, pipeline=pipeline)
        cls.assert_not_called()
        assert classifier.pipeline is pipeline

    def test_raises_when_pipeline_drops_rows(self, mock_model_config):
        """One result row per input text, or the labels would misalign with entries."""
        row = [{"label": _label_text(TopicType.LLM), "score": 0.999}]
        pipeline = create_autospec(
            ZeroShotClassificationPipeline, instance=True, return_value=[row]
        )
        classifier = _classifier(mock_model_config, pipeline)

        with pytest.raises(RuntimeError, match="1 result rows for 2 texts"):
            classifier._infer_batch(["a", "b"], min_score=0.9)

    def test_requests_every_label_score_in_batches(self, mock_model_config, monkeypatch):
        """Labels reach the model as plain text, unfiltered, in configured batches."""
        monkeypatch.setattr(topic_module.settings, "topic_articles_per_batch", 4)
        pipeline = _fake_pipeline()
        classifier = _classifier(mock_model_config, pipeline)

        classifier._infer_batch(["a", "b"], min_score=0.9)

        texts, labels = pipeline.call_args.args
        assert texts == ["a", "b"]
        assert labels == [_label_text(t) for t in TopicType]
        assert "large language models" in labels
        assert pipeline.call_args.kwargs == {"threshold": 0.0, "batch_size": 4}

    def test_assigns_labels_at_or_above_min_score(self, mock_model_config):
        pipeline = _fake_pipeline({TopicType.LLM: 0.999, TopicType.MODEL_RELEASE: 0.998})
        classifier = _classifier(mock_model_config, pipeline)

        [labels] = classifier._infer_batch(["text"], min_score=0.998)

        assert {t.value for t in labels} == {TopicType.LLM, TopicType.MODEL_RELEASE}

    def test_assigns_top_one_if_none_reach_min_score(self, mock_model_config):
        pipeline = _fake_pipeline({TopicType.LLM: 0.4, TopicType.SAFETY: 0.3})
        classifier = _classifier(mock_model_config, pipeline)

        [labels] = classifier._infer_batch(["text"], min_score=0.99)

        assert [t.value for t in labels] == [TopicType.LLM]
        assert labels[0].confidence == pytest.approx(0.4)

    def test_tags_sorted_by_confidence(self, mock_model_config):
        pipeline = _fake_pipeline(
            {TopicType.SAFETY: 0.9991, TopicType.LLM: 0.9999, TopicType.AGENTS: 0.9995}
        )
        classifier = _classifier(mock_model_config, pipeline)

        [labels] = classifier._infer_batch(["text"], min_score=0.999)

        assert [t.value for t in labels] == [TopicType.LLM, TopicType.AGENTS, TopicType.SAFETY]

    def test_tags_are_label_objects(self, mock_model_config):
        classifier = _classifier(mock_model_config, _fake_pipeline({TopicType.LLM: 0.999}))

        [labels] = classifier._infer_batch(["text"], min_score=0.998)

        for tag in labels:
            assert isinstance(tag, Label)
            assert isinstance(tag.value, TopicType)
            assert isinstance(tag.confidence, float)

    def test_results_align_with_input_order(self, mock_model_config):
        pipeline = _fake_pipeline({TopicType.ROBOTICS: 0.999}, {TopicType.PRIVACY: 0.999})
        classifier = _classifier(mock_model_config, pipeline)

        results = classifier._infer_batch(["robots", "privacy"], min_score=0.998)

        assert [r[0].value for r in results] == [TopicType.ROBOTICS, TopicType.PRIVACY]

    def test_classify_entries_reads_content_head_then_title(self, mock_model_config):
        """content_head is scored when present; title and description otherwise."""
        pipeline = _fake_pipeline({TopicType.CLOUD: 0.999})
        classifier = _classifier(mock_model_config, pipeline)

        results = classifier.classify_entries(
            [_entry(1, content_head="head text"), _entry(2)], feed="f"
        )

        assert pipeline.call_args.args[0] == ["head text", "T2\nd2"]
        assert all(r[0].value == TopicType.CLOUD for r in results)

    def test_classify_entries_cuts_at_the_settings_min_score(self, mock_model_config, monkeypatch):
        monkeypatch.setattr(topic_module.settings, "topic_min_score", 0.95)
        pipeline = _fake_pipeline({TopicType.API: 0.97, TopicType.DEVEX: 0.9})
        classifier = _classifier(mock_model_config, pipeline)

        [labels] = classifier.classify_entries([_entry()])

        assert [t.value for t in labels] == [TopicType.API]

    def test_span_records_the_cut_and_batch_settings(self, mock_model_config, monkeypatch):
        """These span fields land in meta.json under stages.topic."""
        monkeypatch.setattr(topic_module.settings, "topic_min_score", 0.95)
        monkeypatch.setattr(topic_module.settings, "topic_articles_per_batch", 4)
        classifier = _classifier(mock_model_config, _fake_pipeline({TopicType.API: 0.97}))

        with collect_stage_telemetry() as sink:
            classifier.classify_entries([_entry(1), _entry(2)], feed="f")

        span = sink["topic"]
        assert span["min_score"] == 0.95
        assert span["articles_per_batch"] == 4
        assert span["entries"] == 2
        assert span["labels"] == 2
        assert span["vocabulary"] == len(TopicType)

    def test_empty_entries(self, mock_model_config):
        pipeline = _fake_pipeline()
        classifier = _classifier(mock_model_config, pipeline)
        assert classifier.classify_entries([]) == []
        pipeline.assert_not_called()
