"""Zero-shot topic classification using BART-large-MNLI.

Three public entry points:

- ``classify_text(text)`` takes a single text and returns ``list[Label]``.
  This is the canonical shape; ``api.label`` feeds raw entry text rather
  than the post-summarizer string.
- ``classify_entries(entries)`` takes a batch ``list[Entry]`` and returns
  ``list[list[Label]]``, using ``entry.content_head`` (with
  ``title + description`` fallback when the head is empty). Wrapped in
  ``log_stage("topic")``.
- ``classify_summaries(summaries)`` is a backward-compatible wrapper for
  direct programmatic callers. It builds the input string from
  ``title + description + summary`` and attaches topics to copies of the
  input summaries. The production path does not use it.
"""

from dataclasses import replace
from time import perf_counter

import torch

from digest_generator.core.types import Entry, Label, Summary, TopicType
from digest_generator.shared.logging import log_stage
from digest_generator.shared.settings import settings
from digest_generator.shared.transformers.types import ModelConfig


class TopicClassifier:
    """Classifies text into ``TopicType`` categories using zero-shot NLI.

    Builds hypotheses of the form "This article is about {label}" for each
    ``TopicType`` value and scores them via entailment probability.

    Args:
        model_config: A pre-loaded ``ModelConfig`` with tokenizer and model hydrated.
    """

    def __init__(self, model_config: ModelConfig):
        self.tokenizer = model_config.tokenizer
        self.model = model_config.model
        self.device = model_config.device
        self.model.eval()
        self.labels = [t.value for t in TopicType]

    def classify_text(self, text: str, threshold: float | None = None) -> list[Label]:
        """Run NLI over ``text``, return labels above threshold (top-1 fallback).

        No log_stage wrapper; meant to be called from a batch entry point or
        from the api layer's ``label`` orchestrator that wraps its own span.
        """
        threshold = threshold if threshold is not None else settings.topic_threshold
        return self._infer(text, threshold)

    def classify_entries(
        self,
        entries: list[Entry],
        threshold: float | None = None,
        *,
        feed: str | None = None,
    ) -> list[list[Label]]:
        """Classify a batch of entries via raw text.

        Uses ``entry.content_head`` when present; falls back to
        ``title + '\\n' + description`` when the head is empty (fetcher-side
        fallback case for feeds without extractable prose). Returns
        ``list[list[Label]]`` aligned with the input order.

        ``feed`` (optional) tags the ``stage.start`` / ``stage.done`` lines
        so concurrent per-feed invocations are distinguishable in run.log.
        """
        threshold = threshold if threshold is not None else settings.topic_threshold

        start_fields = {"feed": feed} if feed else {}
        with log_stage("topic", **start_fields) as span:
            labels_per_entry: list[list[Label]] = []
            total_labels = 0
            start = perf_counter()
            for entry in entries:
                text = entry.content_head or f"{entry.title}\n{entry.description}"
                labels = self._infer(text, threshold)
                labels_per_entry.append(labels)
                total_labels += len(labels)

            elapsed = perf_counter() - start
            n = len(entries)
            articles_per_sec = round(n / elapsed, 3) if elapsed > 0 else 0.0
            avg_labels = round(total_labels / n, 2) if n > 0 else 0.0
            span.set(
                entries=n,
                labels=total_labels,
                avg_labels_per_entry=avg_labels,
                threshold=threshold,
                vocabulary=len(self.labels),
                articles_per_sec=articles_per_sec,
            )
            return labels_per_entry

    def classify_summaries(
        self,
        summaries: list[Summary],
        threshold: float | None = None,
        *,
        feed: str | None = None,
    ) -> list[Summary]:
        """Classify based on title + description + summary (backward-compatible).

        Returns new ``Summary`` objects with ``topics`` populated; the
        originals are not modified. Provided for direct programmatic callers;
        the production path uses ``classify_entries`` via ``api.label``.

        Args:
            summaries: Summaries to classify.
            threshold: Minimum entailment probability to accept a label.
            feed: Optional tag rendered on ``stage.start`` / ``stage.done``
                so concurrent per-feed invocations are distinguishable.

        Returns:
            New list of summaries with ``topics`` populated.
        """
        threshold = threshold if threshold is not None else settings.topic_threshold

        start_fields = {"feed": feed} if feed else {}
        with log_stage("topic", **start_fields) as span:
            labeled: list[Summary] = []
            total_labels = 0
            start = perf_counter()
            for s in summaries:
                text = f"{s.entry.title}\n{s.entry.description}\n{s.summary}"
                labels = self._infer(text, threshold)
                total_labels += len(labels)
                labeled.append(replace(s, topics=labels))

            elapsed = perf_counter() - start
            n = len(labeled)
            articles_per_sec = round(n / elapsed, 3) if elapsed > 0 else 0.0
            avg_labels = round(total_labels / n, 2) if n > 0 else 0.0
            span.set(
                summaries=n,
                labels=total_labels,
                avg_labels_per_summary=avg_labels,
                threshold=threshold,
                vocabulary=len(self.labels),
                articles_per_sec=articles_per_sec,
            )
            return labeled

    def _infer(self, text: str, threshold: float) -> list[Label]:
        """Run NLI inference on a single text against every ``TopicType`` hypothesis."""
        hypotheses = [f"This article is about {lab.replace('-', ' ')}." for lab in self.labels]

        inputs = self.tokenizer(
            [text] * len(hypotheses),
            hypotheses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.topic_max_length,
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=1)
        entail = probs[:, 2]

        scored = sorted(
            ((lab, float(score)) for lab, score in zip(self.labels, entail, strict=True)),
            key=lambda x: x[1],
            reverse=True,
        )

        chosen = [(lab, sc) for lab, sc in scored if sc >= threshold] or [scored[0]]
        return [Label(value=TopicType(lab), confidence=float(sc)) for lab, sc in chosen]
