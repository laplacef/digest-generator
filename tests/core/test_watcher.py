"""Tests for digest_generator/core/digest/stages/watcher.py: WhatToWatch cross-section stage."""

import json
from unittest.mock import MagicMock

import pytest

from digest_generator.core.digest.stages.watcher import (
    _WATCH_SYSTEM_PROMPT,
    WhatToWatch,
    _parse_watch_items,
)
from digest_generator.core.digest.types import Cluster, SectionDraft, WatchItem
from digest_generator.shared import settings as settings_module
from digest_generator.shared.logging import collect_stage_telemetry


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.message.content = content
    return resp


@pytest.fixture
def sections():
    return [
        SectionDraft(
            name="AI & Machine Learning",
            content="## AI & Machine Learning\n\nOpenAI shipped GPT-5.",
            article_count=20,
        ),
        SectionDraft(
            name="Security",
            content="## Security\n\nCloudflare disclosed CVE-2026-0142.",
            article_count=7,
        ),
    ]


@pytest.fixture
def watcher():
    mock_client = MagicMock()
    return WhatToWatch(client=mock_client, model="test-watcher-model")


# =============================================================================
# generate: end-to-end
# =============================================================================


class TestGenerate:
    def test_returns_watch_items(self, watcher, sections):
        payload = json.dumps(
            [
                {
                    "heading": "Open-weight models close the benchmark gap",
                    "body": "Mistral's 200B release scored within 3 points of GPT-5.",
                },
                {
                    "heading": "Supply-chain attacks intensify",
                    "body": "Cloudflare's CVE-2026-0142 disclosure followed two similar reports.",
                },
            ]
        )
        watcher._client.chat.return_value = _mock_response(payload)
        result = watcher.generate(sections)
        assert len(result) == 2
        assert all(isinstance(item, WatchItem) for item in result)
        assert result[0].heading == "Open-weight models close the benchmark gap"
        assert result[1].body.startswith("Cloudflare's CVE-2026-0142")

    def test_uses_watch_prompt(self, watcher, sections):
        watcher._client.chat.return_value = _mock_response("[]")
        watcher.generate(sections)
        messages = watcher._client.chat.call_args.kwargs["messages"]
        assert messages[0]["content"] == _WATCH_SYSTEM_PROMPT

    def test_empty_sections_skips_llm(self, watcher):
        result = watcher.generate([])
        assert result == []
        watcher._client.chat.assert_not_called()

    def test_empty_response_returns_empty_list(self, watcher, sections):
        watcher._client.chat.return_value = _mock_response("")
        assert watcher.generate(sections) == []

    def test_malformed_json_returns_empty_list(self, watcher, sections):
        watcher._client.chat.return_value = _mock_response("not json at all")
        assert watcher.generate(sections) == []

    def test_user_prompt_includes_all_sections(self, watcher, sections):
        watcher._client.chat.return_value = _mock_response("[]")
        watcher.generate(sections)
        user_prompt = watcher._client.chat.call_args.kwargs["messages"][1]["content"]
        assert "AI & Machine Learning" in user_prompt
        assert "Security" in user_prompt

    def test_user_prompt_includes_lede_when_provided(self, watcher, sections):
        watcher._client.chat.return_value = _mock_response("[]")
        watcher.generate(sections, lede_intro="Anthropic's Claude Mythos weaponized zero-days.")
        user_prompt = watcher._client.chat.call_args.kwargs["messages"][1]["content"]
        assert "<lede-already-framed>" in user_prompt
        assert "Anthropic's Claude Mythos weaponized zero-days." in user_prompt

    def test_user_prompt_omits_lede_block_when_none(self, watcher, sections):
        watcher._client.chat.return_value = _mock_response("[]")
        watcher.generate(sections)
        user_prompt = watcher._client.chat.call_args.kwargs["messages"][1]["content"]
        assert "<lede-already-framed>" not in user_prompt

    def test_user_prompt_omits_lede_block_when_empty_string(self, watcher, sections):
        watcher._client.chat.return_value = _mock_response("[]")
        watcher.generate(sections, lede_intro="   ")
        user_prompt = watcher._client.chat.call_args.kwargs["messages"][1]["content"]
        assert "<lede-already-framed>" not in user_prompt


# =============================================================================
# model fallback
# =============================================================================


class TestModelFallback:
    def test_explicit_model_wins(self):
        w = WhatToWatch(client=MagicMock(), model="explicit-model")
        assert w.model == "explicit-model"

    def test_watcher_model_setting_used_when_set(self, monkeypatch):
        monkeypatch.setattr(settings_module.settings, "watcher_model", "watcher-only")
        monkeypatch.setattr(settings_module.settings, "writer_model", "writer-only")
        w = WhatToWatch(client=MagicMock())
        assert w.model == "watcher-only"

    def test_falls_back_to_writer_model(self, monkeypatch):
        monkeypatch.setattr(settings_module.settings, "watcher_model", None)
        monkeypatch.setattr(settings_module.settings, "writer_model", "writer-only")
        w = WhatToWatch(client=MagicMock())
        assert w.model == "writer-only"


# =============================================================================
# _parse_watch_items
# =============================================================================


class TestParseWatchItems:
    def test_parses_raw_json_array(self):
        payload = '[{"heading": "A", "body": "B"}]'
        items = _parse_watch_items(payload)
        assert items == [WatchItem(heading="A", body="B")]

    def test_strips_code_fence(self):
        payload = '```json\n[{"heading": "A", "body": "B"}]\n```'
        items = _parse_watch_items(payload)
        assert items == [WatchItem(heading="A", body="B")]

    def test_strips_plain_fence(self):
        payload = '```\n[{"heading": "A", "body": "B"}]\n```'
        items = _parse_watch_items(payload)
        assert items == [WatchItem(heading="A", body="B")]

    def test_skips_entries_missing_heading(self):
        payload = '[{"heading": "", "body": "B"}, {"heading": "A", "body": "B"}]'
        items = _parse_watch_items(payload)
        assert len(items) == 1
        assert items[0].heading == "A"

    def test_skips_entries_missing_body(self):
        payload = '[{"heading": "A", "body": ""}, {"heading": "A", "body": "B"}]'
        items = _parse_watch_items(payload)
        assert len(items) == 1

    def test_non_array_returns_empty(self):
        payload = '{"heading": "A", "body": "B"}'
        assert _parse_watch_items(payload) == []

    def test_strips_whitespace_around_fields(self):
        payload = '[{"heading": "  A  ", "body": "  B  "}]'
        items = _parse_watch_items(payload)
        assert items == [WatchItem(heading="A", body="B")]

    def test_strips_non_breaking_hyphens_from_heading_and_body(self):

        payload = (
            '[{"heading": "Cloudflare‑Stripe agent protocol outpaces controls", '  # noqa: RUF001
            '"body": "Auth0 pushed policy‑as‑code as the AI‑ready answer."}]'  # noqa: RUF001
        )
        items = _parse_watch_items(payload)
        assert len(items) == 1
        assert items[0].heading == "Cloudflare-Stripe agent protocol outpaces controls"
        assert "policy-as-code" in items[0].body
        assert "AI-ready" in items[0].body


# =============================================================================
# Cluster index: <clusters> block in user prompt
# =============================================================================


class TestClusterIndex:
    def test_clusters_seeded_when_supplied(self, watcher, sections):

        watcher._client.chat.return_value = _mock_response("[]")
        clusters = [
            Cluster(
                id="c0001",
                lede="Anthropic raises $50B at $900B valuation",
                article_urls=["https://a", "https://b"],
                primary_section="business",
                secondary_sections=["ai"],
            ),
            Cluster(
                id="c0002",
                lede="OpenAI adds 3 GW of compute in 90 days",
                article_urls=["https://c"],
                primary_section="infrastructure",
                secondary_sections=[],
            ),
        ]
        watcher.generate(sections, clusters=clusters)
        user_prompt = watcher._client.chat.call_args.kwargs["messages"][1]["content"]
        assert "<clusters>" in user_prompt
        assert "Anthropic raises $50B at $900B valuation" in user_prompt
        assert 'primary="Business"' in user_prompt
        assert 'secondaries="AI & Machine Learning"' in user_prompt
        assert "OpenAI adds 3 GW of compute in 90 days" in user_prompt
        assert 'primary="Infrastructure"' in user_prompt

    def test_clusters_omitted_when_not_supplied(self, watcher, sections):
        watcher._client.chat.return_value = _mock_response("[]")
        watcher.generate(sections)
        user_prompt = watcher._client.chat.call_args.kwargs["messages"][1]["content"]
        # "<clusters>" appears in the task-description prose; the block itself would
        # appear as a standalone line. Check for the block, not the prose mention.
        assert "\n<clusters>\n" not in user_prompt
        assert "</clusters>" not in user_prompt

    def test_clusters_ordered_multi_article_first(self, watcher, sections):

        watcher._client.chat.return_value = _mock_response("[]")
        clusters = [
            Cluster(
                id="c0001",
                lede="size-1 cluster",
                article_urls=["https://a"],
                primary_section="ai",
            ),
            Cluster(
                id="c0002",
                lede="multi-article cluster",
                article_urls=["https://b", "https://c"],
                primary_section="business",
            ),
        ]
        watcher.generate(sections, clusters=clusters)
        user_prompt = watcher._client.chat.call_args.kwargs["messages"][1]["content"]
        # Multi-article cluster should appear before size-1 in the index.
        assert user_prompt.index("multi-article cluster") < user_prompt.index("size-1 cluster")

    def test_clusters_with_invalid_primary_skipped(self, watcher, sections):
        """Defensive: malformed clusters (no primary or no lede) drop out of the index."""

        watcher._client.chat.return_value = _mock_response("[]")
        clusters = [
            Cluster(id="c1", lede="", article_urls=["u"], primary_section="ai"),  # no lede
            Cluster(id="c2", lede="real one", article_urls=["v"], primary_section=""),  # no primary
            Cluster(id="c3", lede="kept", article_urls=["w"], primary_section="ai"),
        ]
        watcher.generate(sections, clusters=clusters)
        user_prompt = watcher._client.chat.call_args.kwargs["messages"][1]["content"]
        assert "<clusters>" in user_prompt
        assert "kept" in user_prompt
        assert "real one" not in user_prompt

    def test_span_records_cluster_telemetry(self, watcher, sections):

        watcher._client.chat.return_value = _mock_response('[{"heading": "H", "body": "B body"}]')
        clusters = [
            Cluster(
                id="c1",
                lede="x",
                article_urls=["u"],
                primary_section="ai",
            )
        ]
        with collect_stage_telemetry() as sink:
            watcher.generate(sections, clusters=clusters)
        assert sink["watcher"]["clusters_seeded"] is True
        assert sink["watcher"]["clusters_supplied"] == 1
