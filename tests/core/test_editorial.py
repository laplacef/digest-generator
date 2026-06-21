"""Tests for digest_generator/core/digest/stages/editorial.py: SectionEditor cleanup pass."""

from unittest.mock import MagicMock

import pytest

from digest_generator.core.digest.stages.editorial import (
    _EDITORIAL_SYSTEM_PROMPT,
    _LENGTH_DELTA_MAX,
    SectionEditor,
)
from digest_generator.core.digest.types import SectionDraft
from digest_generator.shared.logging import log_stage, logger
from digest_generator.shared.runtime.meta import SectionMeta


@pytest.fixture
def captured_logs():
    """Capture every log record at DEBUG+ for the duration of one test.

    Loguru sinks receive a ``Message`` whose ``.record`` is the structured
    record dict; we collect those dicts so tests can assert on
    ``record["message"]``, ``record["level"].name``, etc.
    """
    records: list[dict] = []
    sink_id = logger.add(lambda msg: records.append(dict(msg.record)), level="DEBUG")
    try:
        yield records
    finally:
        logger.remove(sink_id)


def _mock_response(content: str) -> MagicMock:
    """Build a mock response matching the ollama client shape."""
    resp = MagicMock()
    resp.message.content = content
    return resp


@pytest.fixture
def sample_draft():
    return SectionDraft(
        name="AI & Machine Learning",
        content=(
            "## AI & Machine Learning\n\n"
            "OpenAI shipped GPT-5 this week, tightening the gap with open-weight "
            "models. See [the release notes](https://openai.com/gpt5) for details. "
            "Anthropic followed with a Claude 5 preview; see "
            "[the benchmark results](https://anthropic.com/claude5) for numbers.\n\n"
            "What to watch: whether Mistral ships a competing 200B open model next month."
        ),
        article_count=12,
    )


@pytest.fixture
def editor():
    mock_client = MagicMock()
    return SectionEditor(client=mock_client, model="test-editorial-model")


# =============================================================================
# clean: happy path
# =============================================================================


class TestCleanHappyPath:
    def test_returns_edited_draft(self, editor, sample_draft):
        edited_content = sample_draft.content.replace("this week", "over the past week")
        editor._client.chat.return_value = _mock_response(edited_content)
        result = editor.clean(sample_draft)
        assert isinstance(result, SectionDraft)
        assert result.name == sample_draft.name
        assert result.article_count == sample_draft.article_count
        assert result.content == edited_content

    def test_preserves_article_count(self, editor, sample_draft):
        editor._client.chat.return_value = _mock_response(sample_draft.content)
        result = editor.clean(sample_draft)
        assert result.article_count == 12

    def test_invokes_llm_with_editorial_prompt(self, editor, sample_draft):
        editor._client.chat.return_value = _mock_response(sample_draft.content)
        editor.clean(sample_draft)
        call_args = editor._client.chat.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == _EDITORIAL_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert sample_draft.content in messages[1]["content"]
        assert sample_draft.name in messages[1]["content"]

    def test_uses_configured_model(self, editor, sample_draft):
        editor._client.chat.return_value = _mock_response(sample_draft.content)
        editor.clean(sample_draft)
        assert editor._client.chat.call_args.kwargs["model"] == "test-editorial-model"

    def test_passes_editorial_temperature(self, editor, sample_draft):
        editor._client.chat.return_value = _mock_response(sample_draft.content)
        editor.clean(sample_draft)
        options = editor._client.chat.call_args.kwargs["options"]
        assert "temperature" in options


# =============================================================================
# clean: fallback behavior
# =============================================================================


class TestCleanFallback:
    def test_empty_response_returns_original(self, editor, sample_draft):
        editor._client.chat.return_value = _mock_response("")
        result = editor.clean(sample_draft)
        assert result is sample_draft

    def test_dropped_link_returns_original(self, editor, sample_draft):
        edited = sample_draft.content.replace(
            "[the release notes](https://openai.com/gpt5)", "the release notes"
        )
        editor._client.chat.return_value = _mock_response(edited)
        result = editor.clean(sample_draft)
        assert result is sample_draft

    def test_added_link_returns_original(self, editor, sample_draft):
        edited = sample_draft.content + "\n\nExtra [link](https://example.com/extra)."
        # Pad content so length delta alone doesn't trigger the reject.
        editor._client.chat.return_value = _mock_response(edited)
        result = editor.clean(sample_draft)
        assert result is sample_draft

    def test_rewritten_link_anchor_returns_original(self, editor, sample_draft):
        edited = sample_draft.content.replace(
            "[the release notes](https://openai.com/gpt5)",
            "[release notes](https://openai.com/gpt5)",
        )
        editor._client.chat.return_value = _mock_response(edited)
        result = editor.clean(sample_draft)
        assert result is sample_draft

    def test_changed_h2_returns_original(self, editor, sample_draft):
        edited = sample_draft.content.replace("## AI & Machine Learning", "## AI and ML")
        editor._client.chat.return_value = _mock_response(edited)
        result = editor.clean(sample_draft)
        assert result is sample_draft

    def test_length_delta_exceeds_threshold_returns_original(self, editor, sample_draft):
        # Generate output that preserves links and H2 but is dramatically shorter.
        edited = (
            "## AI & Machine Learning\n\n"
            "Releases: [the release notes](https://openai.com/gpt5), "
            "[the benchmark results](https://anthropic.com/claude5)."
        )
        original_words = len(sample_draft.content.split())
        edited_words = len(edited.split())
        delta = abs(edited_words - original_words) / original_words
        assert delta > _LENGTH_DELTA_MAX, "fixture needs to exceed threshold"
        editor._client.chat.return_value = _mock_response(edited)
        result = editor.clean(sample_draft)
        assert result is sample_draft


class TestTypographyNormalization:
    def test_strips_non_breaking_hyphen_from_edited_output(self, editor, sample_draft):
        # gpt-oss:120b substitutes U+2011 inside compound terms; the post-process
        # should restore plain ASCII hyphens before validation/return.
        edited = sample_draft.content.replace(
            "OpenAI shipped GPT-5 this week",
            "OpenAI shipped GPT‑5 this week, an AI‑ready release",  # noqa: RUF001
        )
        editor._client.chat.return_value = _mock_response(edited)
        result = editor.clean(sample_draft)
        assert "‑" not in result.content  # noqa: RUF001  # verifying U+2011 stripped
        assert "GPT-5" in result.content
        assert "AI-ready" in result.content


# =============================================================================
# validator unit tests
# =============================================================================


class TestValidate:
    def test_identical_content_passes(self, sample_draft):
        assert SectionEditor._validate(sample_draft, sample_draft.content) is None

    def test_minor_rewrite_passes(self, sample_draft):
        edited = sample_draft.content.replace("this week", "over the past seven days")
        assert SectionEditor._validate(sample_draft, edited) is None

    def test_empty_original_allows_any_edit(self):
        draft = SectionDraft(name="X", content="", article_count=0)
        assert SectionEditor._validate(draft, "") is None

    def test_link_drift_returns_link_set_reason(self, sample_draft):
        edited = sample_draft.content.replace(
            "[the release notes](https://openai.com/gpt5)", "the release notes"
        )
        assert SectionEditor._validate(sample_draft, edited) == "link_set"

    def test_h2_drift_returns_h2_heading_reason(self, sample_draft):
        edited = sample_draft.content.replace("## AI & Machine Learning", "## AI and ML")
        assert SectionEditor._validate(sample_draft, edited) == "h2_heading"

    def test_length_delta_returns_length_delta_reason(self, sample_draft):
        edited = (
            "## AI & Machine Learning\n\n"
            "Releases: [the release notes](https://openai.com/gpt5), "
            "[the benchmark results](https://anthropic.com/claude5)."
        )
        assert SectionEditor._validate(sample_draft, edited) == "length_delta"


# =============================================================================
# validator telemetry: structured DEBUG lines plus per-reason span counters
# =============================================================================


class TestValidatorTelemetry:
    """Each validator outcome emits one structured DEBUG line and (inside a
    log_stage block) bumps the per-reason counter on the active span."""

    def test_passing_validation_emits_passed_line(self, sample_draft, captured_logs):
        SectionEditor._validate(sample_draft, sample_draft.content)
        passed = [r for r in captured_logs if "editor.validator.passed" in r["message"]]
        assert len(passed) == 1
        assert "AI & Machine Learning" in passed[0]["message"]
        assert "delta_words=0" in passed[0]["message"]

    def test_link_drift_emits_structured_rejection(self, sample_draft, captured_logs):
        edited = sample_draft.content.replace(
            "[the release notes](https://openai.com/gpt5)", "the release notes"
        )
        SectionEditor._validate(sample_draft, edited)
        rejected = [r for r in captured_logs if "editor.validator.rejected" in r["message"]]
        assert len(rejected) == 1
        assert "check=link_set" in rejected[0]["message"]
        assert "dropped=1" in rejected[0]["message"]
        assert "added=0" in rejected[0]["message"]

    def test_h2_drift_emits_structured_rejection(self, sample_draft, captured_logs):
        edited = sample_draft.content.replace("## AI & Machine Learning", "## AI and ML")
        SectionEditor._validate(sample_draft, edited)
        rejected = [r for r in captured_logs if "editor.validator.rejected" in r["message"]]
        assert len(rejected) == 1
        assert "check=h2_heading" in rejected[0]["message"]

    def test_length_delta_emits_structured_rejection(self, sample_draft, captured_logs):
        edited = (
            "## AI & Machine Learning\n\n"
            "Releases: [the release notes](https://openai.com/gpt5), "
            "[the benchmark results](https://anthropic.com/claude5)."
        )
        SectionEditor._validate(sample_draft, edited)
        rejected = [r for r in captured_logs if "editor.validator.rejected" in r["message"]]
        assert len(rejected) == 1
        assert "check=length_delta" in rejected[0]["message"]
        assert "max_pct=30" in rejected[0]["message"]

    def test_rejection_lines_emit_at_debug_level(self, sample_draft, captured_logs):
        edited = sample_draft.content.replace(
            "[the release notes](https://openai.com/gpt5)", "the release notes"
        )
        SectionEditor._validate(sample_draft, edited)
        rejected = [r for r in captured_logs if "editor.validator.rejected" in r["message"]]
        assert all(r["level"].name == "DEBUG" for r in rejected)

    def test_rejection_increments_span_counter(self, sample_draft):
        edited = sample_draft.content.replace(
            "[the release notes](https://openai.com/gpt5)", "the release notes"
        )
        with log_stage("editor") as span:
            SectionEditor._validate(sample_draft, edited)
            SectionEditor._validate(sample_draft, edited)
        assert span.fields.get("rejected_link_set") == 2

    def test_distinct_rejection_reasons_increment_separate_counters(self, sample_draft):
        link_drifted = sample_draft.content.replace(
            "[the release notes](https://openai.com/gpt5)", "the release notes"
        )
        h2_drifted = sample_draft.content.replace("## AI & Machine Learning", "## AI and ML")
        with log_stage("editor") as span:
            SectionEditor._validate(sample_draft, link_drifted)
            SectionEditor._validate(sample_draft, h2_drifted)
        assert span.fields.get("rejected_link_set") == 1
        assert span.fields.get("rejected_h2_heading") == 1
        assert "rejected_length_delta" not in span.fields

    def test_passing_validation_does_not_increment_counters(self, sample_draft):
        with log_stage("editor") as span:
            SectionEditor._validate(sample_draft, sample_draft.content)
        # No rejected_* counters should appear
        assert not any(k.startswith("rejected_") for k in span.fields)

    def test_validator_outside_stage_silently_skips_counter(self, sample_draft):
        """Static-method invocation outside log_stage must not raise."""
        edited = sample_draft.content.replace(
            "[the release notes](https://openai.com/gpt5)", "the release notes"
        )
        # Just don't raise; current_span() returns None outside log_stage
        result = SectionEditor._validate(sample_draft, edited)
        assert result == "link_set"


# =============================================================================
# clean_all
# =============================================================================


class TestCleanAll:
    def test_preserves_order(self, editor):
        body = " ".join(["word"] * 50)
        drafts = [
            SectionDraft(name="A", content=f"## A\n\n{body}", article_count=1),
            SectionDraft(name="B", content=f"## B\n\n{body}", article_count=2),
            SectionDraft(name="C", content=f"## C\n\n{body}", article_count=3),
        ]
        editor._client.chat.side_effect = [
            _mock_response(f"## A\n\n{body} edited"),
            _mock_response(f"## B\n\n{body} edited"),
            _mock_response(f"## C\n\n{body} edited"),
        ]
        result = editor.clean_all(drafts)
        assert [d.name for d in result] == ["A", "B", "C"]
        assert all("edited" in d.content for d in result)
        assert [d.article_count for d in result] == [1, 2, 3]

    def test_empty_input_returns_empty(self, editor):
        assert editor.clean_all([]) == []


# =============================================================================
# section_outcomes: per-section telemetry harvested by the orchestrator
# =============================================================================


class TestSectionOutcomes:
    """Editor exposes per-section outcomes for the orchestrator's meta.json harvest.

    Each entry records (name, articles, edit_outcome, rejected_reason).
    Rewritten sections have ``rejected_reason=None``; fall-back sections carry
    a concrete reason: link_set, h2_heading, length_delta, or empty_response.
    """

    def test_rewrites_record_no_rejection_reason(self, editor, sample_draft):
        # A clean rewrite preserving links/headings/length.
        editor._client.chat.side_effect = [_mock_response(sample_draft.content + " edited")]
        editor.clean_all([sample_draft])
        outcomes = editor.section_outcomes
        assert len(outcomes) == 1
        assert outcomes[0].name == sample_draft.name
        assert outcomes[0].articles == sample_draft.article_count
        assert outcomes[0].edit_outcome == "rewritten"
        assert outcomes[0].rejected_reason is None

    def test_fall_back_records_link_set_reason(self, editor, sample_draft):
        # Edited output drops a link, triggering link_set rejection, then falls back to original.
        edited = sample_draft.content.replace(
            "[the release notes](https://openai.com/gpt5)", "the release notes"
        )
        editor._client.chat.side_effect = [_mock_response(edited)]
        editor.clean_all([sample_draft])
        outcomes = editor.section_outcomes
        assert len(outcomes) == 1
        assert outcomes[0].edit_outcome == "fell_back"
        assert outcomes[0].rejected_reason == "link_set"

    def test_empty_llm_response_records_empty_response_reason(self, editor, sample_draft):
        editor._client.chat.side_effect = [_mock_response("")]
        editor.clean_all([sample_draft])
        outcomes = editor.section_outcomes
        assert len(outcomes) == 1
        assert outcomes[0].edit_outcome == "fell_back"
        assert outcomes[0].rejected_reason == "empty_response"

    def test_outcomes_reset_between_clean_all_calls(self, editor, sample_draft):
        editor._client.chat.side_effect = [
            _mock_response(sample_draft.content + " edited"),
            _mock_response(sample_draft.content + " edited again"),
        ]
        editor.clean_all([sample_draft])
        editor.clean_all([sample_draft])
        # Second call resets the list, so only one entry visible.
        assert len(editor.section_outcomes) == 1

    def test_section_outcomes_returns_a_copy(self, editor, sample_draft):
        editor._client.chat.side_effect = [_mock_response(sample_draft.content + " edited")]
        editor.clean_all([sample_draft])
        outcomes = editor.section_outcomes
        outcomes.append(SectionMeta(name="tampered"))
        # Internal state should not reflect the mutation.
        assert len(editor.section_outcomes) == 1
