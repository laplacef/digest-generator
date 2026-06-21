"""Tests for digest_generator/core/digest/stages/writer.py: SectionWriter logic.

SectionWriter covers only the map phase of digest generation: articles to
SectionDraft. Synthesis, title, and framing live in separate modules.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest

from digest_generator.core.digest.stages.writer import (
    _SECTION_MERGE_SYSTEM_PROMPT,
    _SECTION_SYSTEM_PROMPT,
    SectionWriter,
)
from digest_generator.core.digest.types import Cluster, SectionDraft


@pytest.fixture
def sample_articles():
    """A small list of article dicts mimicking pipeline output."""
    return [
        {
            "title": "GPT-5 Released",
            "url": "https://openai.com/gpt5",
            "origin": "openai-blog",
            "published": "2026-03-10T12:00:00+00:00",
            "summary": "OpenAI released GPT-5 with improved reasoning.",
            "content_type": "ai",
            "topics": {"large-language-models": 0.95, "model-release": 0.88},
        },
        {
            "title": "Zero Trust Architecture",
            "url": "https://blog.cloudflare.com/zero-trust",
            "origin": "cloudflare-blog",
            "published": "2026-03-09T10:00:00+00:00",
            "summary": "Cloudflare introduces new zero trust features.",
            "content_type": "security",
            "topics": {"cybersecurity": 0.82},
        },
        {
            "title": "Scaling Microservices",
            "url": "https://netflixtechblog.com/scaling",
            "origin": "netflix-tech",
            "published": "2026-03-08T08:00:00+00:00",
            "summary": "Netflix shares lessons from scaling microservices.",
            "content_type": "engineering",
            "topics": {"distributed-systems": 0.91, "system-design": 0.75},
        },
    ]


@pytest.fixture
def writer():
    """Create a SectionWriter with a mocked Ollama client."""
    mock_client = MagicMock()
    return SectionWriter(client=mock_client, model="test-model")


# =============================================================================
# _rank_articles
# =============================================================================


class TestRankArticles:
    def test_sorts_by_highest_confidence(self, sample_articles):
        ranked = SectionWriter._rank_articles(sample_articles)
        assert ranked[0]["title"] == "GPT-5 Released"
        assert ranked[1]["title"] == "Scaling Microservices"
        assert ranked[2]["title"] == "Zero Trust Architecture"

    def test_handles_empty_topics(self):
        articles = [
            {"title": "No Topics", "topics": {}},
            {"title": "Has Topics", "topics": {"llm": 0.5}},
        ]
        ranked = SectionWriter._rank_articles(articles)
        assert ranked[0]["title"] == "Has Topics"
        assert ranked[1]["title"] == "No Topics"

    def test_handles_missing_topics_key(self):
        articles: list[dict[str, Any]] = [
            {"title": "Missing Topics"},
            {"title": "Has Topics", "topics": {"llm": 0.5}},
        ]
        ranked = SectionWriter._rank_articles(articles)
        assert ranked[0]["title"] == "Has Topics"

    def test_empty_list(self):
        assert SectionWriter._rank_articles([]) == []


# =============================================================================
# _group_by_clusters: dict input, with optional cluster-routing
# =============================================================================


class TestGroupResults:
    def test_groups_by_content_type(self, writer, sample_articles):
        results = {"feed1": sample_articles}
        grouped = writer._group_by_clusters(results, None)

        assert len(grouped) == 3
        assert len(grouped["ai"]) == 1
        assert len(grouped["security"]) == 1
        assert len(grouped["engineering"]) == 1

    def test_skips_invalid_content_type(self, writer):
        results = {
            "feed1": [
                {"title": "Valid", "content_type": "ai"},
                {"title": "Invalid", "content_type": "nonexistent"},
            ]
        }
        grouped = writer._group_by_clusters(results, None)
        assert len(grouped) == 1
        assert "ai" in grouped

    def test_skips_missing_content_type(self, writer):
        results = {"feed1": [{"title": "No CT"}]}
        grouped = writer._group_by_clusters(results, None)
        assert len(grouped) == 0

    def test_merges_across_feeds(self, writer):
        results = {
            "feed1": [{"title": "A", "content_type": "ai"}],
            "feed2": [{"title": "B", "content_type": "ai"}],
        }
        grouped = writer._group_by_clusters(results, None)
        assert len(grouped["ai"]) == 2


# =============================================================================
# _format_articles
# =============================================================================


class TestFormatArticles:
    def test_wraps_in_articles_tag(self, sample_articles):
        lines = SectionWriter._format_articles(sample_articles)
        assert lines[0] == "<articles>"
        assert lines[-1] == "</articles>"

    def test_includes_article_fields(self, sample_articles):
        lines = SectionWriter._format_articles(sample_articles[:1])
        text = "\n".join(lines)
        assert "<title>GPT-5 Released</title>" in text
        assert "<url>https://openai.com/gpt5</url>" in text
        assert "<origin>openai-blog</origin>" in text
        assert "<summary>" in text

    def test_includes_top_3_topics(self):
        articles = [
            {
                "title": "Test",
                "url": "http://test.com",
                "summary": "Test summary",
                "topics": {"tag1": 0.9, "tag2": 0.8, "tag3": 0.7, "tag4": 0.6},
            }
        ]
        lines = SectionWriter._format_articles(articles)
        text = "\n".join(lines)
        assert "tag1" in text
        assert "tag2" in text
        assert "tag3" in text
        assert "tag4" not in text

    def test_handles_no_topics(self):
        articles = [{"title": "Test", "url": "http://test.com", "summary": "Test", "topics": {}}]
        lines = SectionWriter._format_articles(articles)
        text = "\n".join(lines)
        assert "<topics>" not in text

    def test_includes_description_when_present(self):
        articles = [
            {
                "title": "T",
                "url": "http://x",
                "summary": "S",
                "description": "Publisher-authored blurb",
                "topics": {},
            }
        ]
        text = "\n".join(SectionWriter._format_articles(articles))
        assert "<description>Publisher-authored blurb</description>" in text

    def test_omits_description_when_missing(self):
        articles = [{"title": "T", "url": "http://x", "summary": "S", "topics": {}}]
        text = "\n".join(SectionWriter._format_articles(articles))
        assert "<description>" not in text

    def test_omits_description_when_empty_string(self):
        articles = [
            {"title": "T", "url": "http://x", "summary": "S", "description": "   ", "topics": {}}
        ]
        text = "\n".join(SectionWriter._format_articles(articles))
        assert "<description>" not in text

    def test_includes_content_head_when_present(self):
        articles = [
            {
                "title": "T",
                "url": "http://x",
                "summary": "S",
                "content_head": "Raw article prose here.",
                "topics": {},
            }
        ]
        text = "\n".join(SectionWriter._format_articles(articles))
        assert "<content_head>Raw article prose here.</content_head>" in text

    def test_omits_content_head_when_missing(self):
        articles = [{"title": "T", "url": "http://x", "summary": "S", "topics": {}}]
        text = "\n".join(SectionWriter._format_articles(articles))
        assert "<content_head>" not in text


# =============================================================================
# _call_llm
# =============================================================================


class TestCallLlm:
    def test_returns_content(self, writer):
        mock_response = MagicMock()
        mock_response.message.content = "Generated text here"
        writer._client.chat.return_value = mock_response

        result = writer._call_llm("system prompt", "user prompt")
        assert result == "Generated text here"

    def test_returns_empty_on_none_content(self, writer):
        mock_response = MagicMock()
        mock_response.message.content = None
        writer._client.chat.return_value = mock_response
        assert writer._call_llm("sys", "user") == ""

    def test_returns_empty_on_empty_content(self, writer):
        mock_response = MagicMock()
        mock_response.message.content = ""
        writer._client.chat.return_value = mock_response
        assert writer._call_llm("sys", "user") == ""

    def test_passes_think_false(self, writer):
        mock_response = MagicMock()
        mock_response.message.content = "text"
        writer._client.chat.return_value = mock_response

        writer._call_llm("sys", "user")
        assert writer._client.chat.call_args.kwargs["think"] is False

    def test_passes_temperature(self, writer):
        mock_response = MagicMock()
        mock_response.message.content = "text"
        writer._client.chat.return_value = mock_response

        writer._call_llm("sys", "user", temperature=0.7)
        assert writer._client.chat.call_args.kwargs["options"] == {"temperature": 0.7}

    def test_passes_model_name(self, writer):
        mock_response = MagicMock()
        mock_response.message.content = "text"
        writer._client.chat.return_value = mock_response

        writer._call_llm("sys", "user")
        assert writer._client.chat.call_args.kwargs["model"] == "test-model"


# =============================================================================
# _group_feed_results: typed FeedResult input
# =============================================================================


# =============================================================================
# _write_section_draft
# =============================================================================


class TestWriteSectionDraft:
    def test_includes_date_range_in_prompt(self, writer):
        mock_response = MagicMock()
        mock_response.message.content = "Section draft"
        writer._client.chat.return_value = mock_response

        articles = [{"title": "T", "url": "u", "summary": "s", "topics": {}}]
        writer._write_section_draft(
            "AI & Machine Learning", articles, [], date_range=("2026-03-01", "2026-03-15")
        )

        user_msg = writer._client.chat.call_args.kwargs["messages"][1]["content"]
        assert "covering 2026-03-01 to 2026-03-15" in user_msg

    def test_uses_section_system_prompt(self, writer):
        mock_response = MagicMock()
        mock_response.message.content = "Section draft"
        writer._client.chat.return_value = mock_response

        articles = [{"title": "T", "url": "u", "summary": "s", "topics": {}}]
        writer._write_section_draft("Security", articles, [])

        system_msg = writer._client.chat.call_args.kwargs["messages"][0]["content"]
        assert system_msg == _SECTION_SYSTEM_PROMPT

    def test_includes_article_count_in_prompt(self, writer):
        mock_response = MagicMock()
        mock_response.message.content = "Section draft"
        writer._client.chat.return_value = mock_response

        articles = [
            {"title": "A", "url": "u", "summary": "s", "topics": {}},
            {"title": "B", "url": "u", "summary": "s", "topics": {}},
        ]
        writer._write_section_draft("AI & Machine Learning", articles, [])

        user_msg = writer._client.chat.call_args.kwargs["messages"][1]["content"]
        assert "2 articles" in user_msg


# =============================================================================
# _write_section: batching
# =============================================================================


class TestWriteSection:
    def test_small_section_calls_draft_directly(self, writer):
        mock_response = MagicMock()
        mock_response.message.content = "Section draft"
        writer._client.chat.return_value = mock_response

        articles = [{"title": f"A{i}", "url": "u", "summary": "s", "topics": {}} for i in range(5)]
        result = writer._write_section("AI & Machine Learning", articles, [])

        assert result == "Section draft"
        assert writer._client.chat.call_count == 1

    @pytest.fixture
    def _batch_size_3(self, monkeypatch):
        monkeypatch.setattr(
            "digest_generator.core.digest.stages.writer.settings.writer_section_batch_size", 3
        )

    @pytest.mark.usefixtures("_batch_size_3")
    def test_large_section_splits_into_batches(self, writer):
        mock_response = MagicMock()
        mock_response.message.content = "Draft content"
        writer._client.chat.return_value = mock_response

        articles = [{"title": f"A{i}", "url": "u", "summary": "s", "topics": {}} for i in range(7)]
        writer._write_section("AI & Machine Learning", articles, [])
        assert writer._client.chat.call_count == 4

    @pytest.mark.usefixtures("_batch_size_3")
    def test_merge_call_uses_merge_prompt(self, writer):
        mock_response = MagicMock()
        mock_response.message.content = "Draft content"
        writer._client.chat.return_value = mock_response

        articles = [{"title": f"A{i}", "url": "u", "summary": "s", "topics": {}} for i in range(7)]
        writer._write_section("AI & Machine Learning", articles, [])

        merge_call = writer._client.chat.call_args_list[-1]
        system_msg = merge_call.kwargs["messages"][0]["content"]
        assert system_msg == _SECTION_MERGE_SYSTEM_PROMPT

    @pytest.mark.usefixtures("_batch_size_3")
    def test_all_sub_drafts_empty_returns_empty(self, writer):
        mock_response = MagicMock()
        mock_response.message.content = ""
        writer._client.chat.return_value = mock_response

        articles = [{"title": f"A{i}", "url": "u", "summary": "s", "topics": {}} for i in range(7)]
        assert writer._write_section("AI & Machine Learning", articles, []) == ""

    @pytest.mark.usefixtures("_batch_size_3")
    def test_single_sub_draft_skips_merge(self, writer):
        call_count = 0

        def fake_chat(**_kwargs):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.message.content = "Only draft" if call_count == 1 else ""
            return resp

        writer._client.chat.side_effect = fake_chat

        articles = [{"title": f"A{i}", "url": "u", "summary": "s", "topics": {}} for i in range(7)]
        result = writer._write_section("AI & Machine Learning", articles, [])

        assert result == "Only draft"
        assert writer._client.chat.call_count == 3


# =============================================================================
# _merge_section_drafts
# =============================================================================


class TestMergeSectionDrafts:
    def test_uses_merge_system_prompt(self, writer):
        mock_response = MagicMock()
        mock_response.message.content = "Merged section"
        writer._client.chat.return_value = mock_response

        writer._merge_section_drafts("Security", ["draft 1", "draft 2"])

        system_msg = writer._client.chat.call_args.kwargs["messages"][0]["content"]
        assert system_msg == _SECTION_MERGE_SYSTEM_PROMPT

    def test_includes_draft_count_in_task(self, writer):
        mock_response = MagicMock()
        mock_response.message.content = "Merged section"
        writer._client.chat.return_value = mock_response

        writer._merge_section_drafts("Security", ["d1", "d2", "d3"])

        user_msg = writer._client.chat.call_args.kwargs["messages"][1]["content"]
        assert "3 partial drafts" in user_msg

    def test_wraps_each_draft_in_partial_draft_tags(self, writer):
        mock_response = MagicMock()
        mock_response.message.content = "Merged section"
        writer._client.chat.return_value = mock_response

        writer._merge_section_drafts("Security", ["Alpha draft", "Beta draft"])

        user_msg = writer._client.chat.call_args.kwargs["messages"][1]["content"]
        assert '<partial-draft number="1">' in user_msg
        assert "Alpha draft" in user_msg
        assert '<partial-draft number="2">' in user_msg
        assert "Beta draft" in user_msg


# =============================================================================
# write_all_from_json: public entry point
# =============================================================================


class TestWriteAllFromJson:
    def test_returns_section_drafts(self, writer, sample_articles):
        mock_response = MagicMock()
        mock_response.message.content = "## Body\n\nBody."
        writer._client.chat.return_value = mock_response

        drafts = writer.write_all_from_json({"feed1": sample_articles})

        assert all(isinstance(d, SectionDraft) for d in drafts)
        # Three different content types in sample_articles produce three drafts
        assert {d.name for d in drafts} == {
            "AI & Machine Learning",
            "Engineering",
            "Security",
        }


# =============================================================================
# Cluster routing + <covered-elsewhere> block (post-clusterer integration)
# =============================================================================


class TestClusterRouting:
    def test_primary_section_overrides_content_type(self, writer):
        """primary='business' wins over content_type='infrastructure'."""

        results = {
            "techcrunch": [
                {
                    "title": "Cloudflare/Stripe protocol",
                    "url": "https://example.com/cf-stripe",
                    "content_type": "infrastructure",
                }
            ]
        }
        clusters = [
            Cluster(
                id="c0001",
                lede="Cloudflare/Stripe agent payment protocol",
                article_urls=["https://example.com/cf-stripe"],
                primary_section="business",
                secondary_sections=["infrastructure"],
            )
        ]
        grouped = writer._group_by_clusters(results, clusters)
        assert "business" in grouped
        assert "infrastructure" not in grouped

    def test_articles_without_cluster_fall_back_to_content_type(self, writer):

        results = {
            "feed": [
                {"title": "A", "url": "https://a", "content_type": "ai"},
                {"title": "B", "url": "https://b", "content_type": "security"},
            ]
        }
        clusters = [
            Cluster(
                id="c1",
                lede="x",
                article_urls=["https://a"],
                primary_section="business",
            )
        ]
        grouped = writer._group_by_clusters(results, clusters)
        assert "business" in grouped
        assert "security" in grouped
        assert "ai" not in grouped

    def test_cluster_with_empty_primary_falls_back_to_content_type(self, writer):

        results = {"feed": [{"title": "T", "url": "https://u", "content_type": "ai"}]}
        clusters = [
            Cluster(
                id="c1",
                lede="x",
                article_urls=["https://u"],
                primary_section="",
            )
        ]
        grouped = writer._group_by_clusters(results, clusters)
        assert "ai" in grouped

    def test_build_cross_refs_groups_by_secondary_section(self, writer):

        clusters = [
            Cluster(
                id="c1",
                lede="primary=business",
                article_urls=["u1"],
                primary_section="business",
                secondary_sections=["infrastructure", "ai"],
            ),
            Cluster(
                id="c2",
                lede="primary=security",
                article_urls=["u2"],
                primary_section="security",
                secondary_sections=["ai"],
            ),
            Cluster(
                id="c3",
                lede="no secondaries",
                article_urls=["u3"],
                primary_section="engineering",
                secondary_sections=[],
            ),
        ]
        refs = writer._build_cross_refs(clusters)
        assert {c.id for c in refs["infrastructure"]} == {"c1"}
        assert {c.id for c in refs["ai"]} == {"c1", "c2"}
        assert "engineering" not in refs

    def test_format_cross_refs_renders_lede_entities_urls(self, writer):

        clusters = [
            Cluster(
                id="c1",
                lede="Cloudflare/Stripe agent protocol",
                article_urls=["https://blog.cloudflare.com/x", "https://infoworld.com/y"],
                primary_section="business",
                secondary_sections=["infrastructure"],
                entities=["Cloudflare", "Stripe", "$100/mo cap"],
            )
        ]
        block = "\n".join(writer._format_cross_refs(clusters))
        assert "<covered-elsewhere>" in block
        assert 'primary="Business"' in block
        assert "<lede>Cloudflare/Stripe agent protocol</lede>" in block
        assert "<entities>Cloudflare, Stripe, $100/mo cap</entities>" in block
        assert "<urls>https://blog.cloudflare.com/x, https://infoworld.com/y</urls>" in block

    def test_format_cross_refs_omits_optional_fields_when_empty(self, writer):

        clusters = [
            Cluster(
                id="c1",
                lede="some lede",
                article_urls=[],
                primary_section="business",
                secondary_sections=[],
                entities=[],
            )
        ]
        block = "\n".join(writer._format_cross_refs(clusters))
        assert "<entities>" not in block
        assert "<urls>" not in block
        assert "<lede>some lede</lede>" in block

    def test_write_all_passes_cross_refs_to_user_prompt(self, writer):

        mock_response = MagicMock()
        mock_response.message.content = "## Body\n\nBody."
        writer._client.chat.return_value = mock_response

        results = {
            "feed": [
                {
                    "title": "AI article",
                    "url": "https://ai",
                    "summary": "s",
                    "topics": {},
                    "content_type": "ai",
                }
            ]
        }
        clusters = [
            Cluster(
                id="c1",
                lede="Cloudflare/Stripe agent protocol",
                article_urls=["https://elsewhere"],
                primary_section="business",
                secondary_sections=["ai"],
                entities=["Cloudflare", "Stripe", "$100/mo"],
            )
        ]
        writer.write_all_from_json(results, clusters=clusters)
        user_prompt = writer._client.chat.call_args.kwargs["messages"][1]["content"]
        assert "<covered-elsewhere>" in user_prompt
        assert "Cloudflare/Stripe agent protocol" in user_prompt
        assert "<entities>Cloudflare, Stripe, $100/mo</entities>" in user_prompt

    def test_section_with_only_cross_refs_no_primary_is_skipped(self, writer):

        mock_response = MagicMock()
        mock_response.message.content = "## Body\n\nBody."
        writer._client.chat.return_value = mock_response

        results = {
            "feed": [
                {
                    "title": "AI article",
                    "url": "https://ai",
                    "summary": "s",
                    "topics": {},
                    "content_type": "ai",
                }
            ]
        }
        clusters = [
            Cluster(
                id="c1",
                lede="Some other story",
                article_urls=["https://elsewhere"],
                primary_section="business",
                secondary_sections=["security"],
            )
        ]
        drafts = writer.write_all_from_json(results, clusters=clusters)
        # AI section drafted; Security skipped (cross-ref only, no primary anchor).
        assert {d.name for d in drafts} == {"AI & Machine Learning"}
