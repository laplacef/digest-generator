"""Tests for digest_generator/core/digest/stages/clusterer.py: ArticleClusterer.

Covers both code paths through ``cluster()``:

- **LLM happy path** with a mocked Ollama client returning well-formed JSON.
- **Trivial fallback** via the module-level helper, plus each soft-failure
  mode that drops back to it (LLM exception, empty response, malformed JSON,
  zero-valid-clusters).
- Validator semantics (duplicate-article first-wins, backfill of dropped
  articles, secondary-section dedup + cap, invalid section coercion).
- Orchestrator cache helper round-trip.
- Cluster dataclass serialization shape (locked for clusters.json io).
"""

import json
from dataclasses import asdict
from unittest.mock import MagicMock

import pytest

from digest_generator.core.digest.orchestrator import _cached_or_fresh_clusters
from digest_generator.core.digest.stages.clusterer import (
    _CLUSTER_SYSTEM_PROMPT,
    ArticleClusterer,
    _coerce_secondary_sections,
    _majority_content_type,
    _parse_clusters_json,
    _trivial_fallback,
    _validate_and_normalize,
)
from digest_generator.core.digest.types import Cluster

# Category ids the validation helpers accept; mirrors the conftest test set.
_SECTIONS = frozenset({"ai", "engineering", "infrastructure", "security", "business"})


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.message.content = content
    return resp


@pytest.fixture
def results():
    """Two feeds, three articles spanning two content types."""
    return {
        "techcrunch": [
            {
                "url": "https://techcrunch.com/anthropic-50b",
                "title": "Anthropic seeks $50B at $900B valuation",
                "content_type": "business",
                "summary": "Anthropic courts investors for a $50B fundraise...",
                "origin": "techcrunch",
                "published": "2026-04-29T00:00:00Z",
                "topics": {"business": 0.9, "funding": 0.7},
            },
            {
                "url": "https://techcrunch.com/cloudflare-stripe",
                "title": "Cloudflare and Stripe ship agent payment protocol",
                "content_type": "infrastructure",
                "summary": "New primitives let AI agents create accounts...",
                "origin": "techcrunch",
                "published": "2026-04-30T00:00:00Z",
                "topics": {"infrastructure": 0.8, "cloud": 0.7},
            },
        ],
        "aws-blog": [
            {
                "url": "https://aws.amazon.com/strands-agents",
                "title": "AWS Strands Agents framework GA",
                "content_type": "infrastructure",
                "summary": "Strands lets agents diagnose RDS deadlocks...",
                "origin": "aws-blog",
                "published": "2026-05-01T00:00:00Z",
                "topics": {"infrastructure": 0.85, "ai": 0.75},
            },
        ],
    }


@pytest.fixture
def clusterer():
    return ArticleClusterer(client=MagicMock(), model="test-clusterer-model")


# =============================================================================
# Constructor
# =============================================================================


class TestArticleClustererInit:
    def test_defaults_resolve_to_settings(self):
        c = ArticleClusterer()
        assert c.model == "gpt-oss:120b-cloud"  # settings.clusterer_model default

    def test_explicit_model_wins_over_settings(self):
        c = ArticleClusterer(model="custom-model:latest")
        assert c.model == "custom-model:latest"

    def test_client_injection(self):
        mock_client = MagicMock()
        c = ArticleClusterer(client=mock_client)
        assert c._client is mock_client


# =============================================================================
# LLM happy path
# =============================================================================


class TestClusterLLMPath:
    def test_returns_llm_clusters_when_response_well_formed(self, clusterer, results):
        payload = json.dumps(
            [
                {
                    "id": "k1",
                    "lede": "Cloudflare/Stripe and AWS ship agent infra primitives",
                    "articles": ["a0002", "a0003"],
                    "primary_section": "infrastructure",
                    "secondary_sections": ["business"],
                },
                {
                    "id": "k2",
                    "lede": "Anthropic raises $50B at $900B valuation",
                    "articles": ["a0001"],
                    "primary_section": "business",
                    "secondary_sections": [],
                },
            ]
        )
        clusterer._client.chat.return_value = _mock_response(payload)
        clusters = clusterer.cluster(results)
        assert len(clusters) == 2
        # Canonical re-numbering: ids become c0001/c0002 regardless of LLM ids.
        assert [c.id for c in clusters] == ["c0001", "c0002"]
        assert clusters[0].article_urls == [
            "https://techcrunch.com/cloudflare-stripe",
            "https://aws.amazon.com/strands-agents",
        ]
        assert clusters[0].primary_section == "infrastructure"
        assert clusters[0].secondary_sections == ["business"]
        assert clusters[1].primary_section == "business"

    def test_uses_cluster_system_prompt(self, clusterer, results):
        clusterer._client.chat.return_value = _mock_response("[]")
        clusterer.cluster(results)
        messages = clusterer._client.chat.call_args.kwargs["messages"]
        assert messages[0]["content"] == _CLUSTER_SYSTEM_PROMPT

    def test_user_prompt_contains_every_article_id_and_url(self, clusterer, results):
        clusterer._client.chat.return_value = _mock_response("[]")
        clusterer.cluster(results)
        user_prompt = clusterer._client.chat.call_args.kwargs["messages"][1]["content"]
        # Three articles produce a0001, a0002, a0003 present
        for i in range(1, 4):
            assert f'<article id="a{i:04d}">' in user_prompt
        # All three URLs appear
        for article in (
            "https://techcrunch.com/anthropic-50b",
            "https://techcrunch.com/cloudflare-stripe",
            "https://aws.amazon.com/strands-agents",
        ):
            assert article in user_prompt

    def test_user_prompt_lists_configured_sections(self, clusterer, results):
        """The <sections> block names the configured category ids + titles."""
        clusterer._client.chat.return_value = _mock_response("[]")
        clusterer.cluster(results)
        user_prompt = clusterer._client.chat.call_args.kwargs["messages"][1]["content"]
        assert "<sections>" in user_prompt
        assert '<section id="ai">AI & Machine Learning</section>' in user_prompt
        assert '<section id="security">Security</section>' in user_prompt


# =============================================================================
# Soft failure modes: each drops back to the trivial fallback
# =============================================================================


class TestClusterLLMFallbacks:
    def test_empty_input_skips_llm(self, clusterer):
        assert clusterer.cluster({}) == []
        clusterer._client.chat.assert_not_called()

    def test_llm_exception_falls_back(self, clusterer, results):
        clusterer._client.chat.side_effect = RuntimeError("ollama down")
        clusters = clusterer.cluster(results)
        assert len(clusters) == 3  # one per article
        assert all(len(c.article_urls) == 1 for c in clusters)

    def test_empty_response_falls_back(self, clusterer, results):
        clusterer._client.chat.return_value = _mock_response("")
        clusters = clusterer.cluster(results)
        assert len(clusters) == 3
        assert all(len(c.article_urls) == 1 for c in clusters)

    def test_non_json_response_falls_back(self, clusterer, results):
        clusterer._client.chat.return_value = _mock_response("not json at all")
        clusters = clusterer.cluster(results)
        assert len(clusters) == 3

    def test_non_array_json_falls_back(self, clusterer, results):
        clusterer._client.chat.return_value = _mock_response('{"oops": "object"}')
        clusters = clusterer.cluster(results)
        assert len(clusters) == 3

    def test_zero_valid_clusters_falls_back(self, clusterer, results):
        # Every cluster references unknown article ids, so validator drops them all.
        payload = json.dumps([{"id": "k1", "articles": ["a9999"], "primary_section": "ai"}])
        clusterer._client.chat.return_value = _mock_response(payload)
        clusters = clusterer.cluster(results)
        assert len(clusters) == 3
        assert all(len(c.article_urls) == 1 for c in clusters)


# =============================================================================
# Validator semantics (module-level helper, decoupled from LLM call)
# =============================================================================


class TestValidator:
    @pytest.fixture
    def indexed(self):
        return [
            ("a0001", {"url": "u1", "content_type": "ai", "title": "t1"}),
            ("a0002", {"url": "u2", "content_type": "ai", "title": "t2"}),
            ("a0003", {"url": "u3", "content_type": "business", "title": "t3"}),
        ]

    def test_first_wins_on_duplicate_article_assignment(self, indexed):
        raw = [
            {"id": "k1", "articles": ["a0001", "a0002"], "primary_section": "ai"},
            {"id": "k2", "articles": ["a0002", "a0003"], "primary_section": "business"},
        ]
        clusters = _validate_and_normalize(raw, indexed, _SECTIONS)
        assert clusters[0].article_urls == ["u1", "u2"]
        # a0002 already taken, so cluster 2 only gets a0003
        assert clusters[1].article_urls == ["u3"]

    def test_backfills_missing_articles_as_size_one_clusters(self, indexed):
        raw = [{"id": "k1", "articles": ["a0001"], "primary_section": "ai"}]
        clusters = _validate_and_normalize(raw, indexed, _SECTIONS)
        # 1 LLM cluster + 2 backfilled size-1 clusters
        assert len(clusters) == 3
        backfilled = [c for c in clusters if len(c.article_urls) == 1]
        assert {c.article_urls[0] for c in backfilled} == {"u1", "u2", "u3"}

    def test_drops_cluster_when_primary_invalid_and_no_majority(self, indexed):
        # Articles without content_type cause the cluster to get dropped, then backfilled.
        no_ct = [("a0001", {"url": "u1", "title": "t1"})]
        raw = [{"id": "k1", "articles": ["a0001"], "primary_section": "not-real"}]
        clusters = _validate_and_normalize(raw, no_ct, _SECTIONS)
        # Cluster dropped; backfill creates size-1 with primary=""
        assert len(clusters) == 1
        assert clusters[0].primary_section == ""
        assert clusters[0].article_urls == ["u1"]

    def test_invalid_primary_section_falls_back_to_majority(self, indexed):
        raw = [
            {
                "id": "k1",
                "articles": ["a0001", "a0002", "a0003"],
                "primary_section": "garbage",
            }
        ]
        clusters = _validate_and_normalize(raw, indexed, _SECTIONS)
        # Two `ai` articles vs one `business`, so majority picks ai
        assert clusters[0].primary_section == "ai"

    def test_secondary_sections_validated_deduped_capped(self):
        primary = "ai"
        out = _coerce_secondary_sections(
            ["security", "garbage", "security", "business", "engineering", "ai"],
            primary,
            _SECTIONS,
        )
        # garbage dropped, duplicate dropped, primary dropped, cap at 2
        assert out == ["security", "business"]

    def test_non_list_secondary_sections_returns_empty(self):
        assert _coerce_secondary_sections("not-a-list", "ai", _SECTIONS) == []
        assert _coerce_secondary_sections(None, "ai", _SECTIONS) == []

    def test_canonical_re_id_independent_of_llm_handles(self, indexed):
        raw = [
            {"id": "zzz", "articles": ["a0003"], "primary_section": "business"},
            {"id": "kaboom", "articles": ["a0001"], "primary_section": "ai"},
        ]
        clusters = _validate_and_normalize(raw, indexed, _SECTIONS)
        ids = [c.id for c in clusters]
        # First two are LLM clusters re-numbered; the third is backfilled (a0002).
        assert ids[0] == "c0001"
        assert ids[1] == "c0002"


class TestMajorityContentType:
    def test_majority_wins(self):
        articles = [
            {"content_type": "ai"},
            {"content_type": "ai"},
            {"content_type": "business"},
        ]
        assert _majority_content_type(articles, _SECTIONS) == "ai"

    def test_first_seen_wins_on_tie(self):
        articles = [{"content_type": "ai"}, {"content_type": "business"}]
        # Both count once, so max() is stable, first-seen wins
        assert _majority_content_type(articles, _SECTIONS) == "ai"

    def test_invalid_content_types_ignored(self):
        articles = [{"content_type": "garbage"}, {"content_type": "ai"}]
        assert _majority_content_type(articles, _SECTIONS) == "ai"

    def test_no_valid_content_type_returns_empty(self):
        assert _majority_content_type([{"content_type": "garbage"}], _SECTIONS) == ""
        assert _majority_content_type([{}], _SECTIONS) == ""


# =============================================================================
# Parser (module-level)
# =============================================================================


class TestParseClustersJson:
    def test_bare_array_parses(self):
        out = _parse_clusters_json('[{"id":"k1","articles":["a0001"]}]')
        assert out == [{"id": "k1", "articles": ["a0001"]}]

    def test_json_fence_stripped(self):
        wrapped = '```json\n[{"id":"k1","articles":["a0001"]}]\n```'
        out = _parse_clusters_json(wrapped)
        assert out == [{"id": "k1", "articles": ["a0001"]}]

    def test_non_dict_elements_dropped(self):
        out = _parse_clusters_json('[{"id":"k1"}, "bogus", 42]')
        assert out == [{"id": "k1"}]

    def test_malformed_returns_none(self):
        assert _parse_clusters_json("not json") is None

    def test_non_array_returns_none(self):
        assert _parse_clusters_json('{"oops": "obj"}') is None


# =============================================================================
# Trivial fallback (module-level helper)
# =============================================================================


class TestTrivialFallback:
    def test_one_cluster_per_article(self):
        indexed = [
            ("a0001", {"url": "u1", "title": "t1", "content_type": "ai"}),
            ("a0002", {"url": "u2", "title": "t2", "content_type": "business"}),
        ]
        clusters = _trivial_fallback(indexed, _SECTIONS)
        assert len(clusters) == 2
        assert clusters[0].id == "c0001"
        assert clusters[1].id == "c0002"
        assert clusters[0].primary_section == "ai"
        assert clusters[1].primary_section == "business"
        assert all(c.secondary_sections == [] for c in clusters)

    def test_invalid_content_type_yields_empty_primary(self):
        indexed = [("a0001", {"url": "u", "title": "t", "content_type": "not-real"})]
        clusters = _trivial_fallback(indexed, _SECTIONS)
        assert clusters[0].primary_section == ""

    def test_missing_url_yields_empty_url_list(self):
        indexed = [("a0001", {"title": "t", "content_type": "ai"})]
        clusters = _trivial_fallback(indexed, _SECTIONS)
        assert clusters[0].article_urls == []


# =============================================================================
# Cluster dataclass serialization (locked for clusters.json io)
# =============================================================================


class TestClusterSerialization:
    def test_round_trip_via_asdict(self):
        c = Cluster(
            id="c0042",
            lede="Anthropic funding round",
            article_urls=["https://a", "https://b"],
            primary_section="business",
            secondary_sections=["ai"],
            entities=["Anthropic", "$50B"],
        )
        d = asdict(c)
        assert d == {
            "id": "c0042",
            "lede": "Anthropic funding round",
            "article_urls": ["https://a", "https://b"],
            "primary_section": "business",
            "secondary_sections": ["ai"],
            "entities": ["Anthropic", "$50B"],
        }
        assert Cluster(**d) == c

    def test_secondary_sections_default_is_empty_list(self):
        c = Cluster(id="c1", lede="x", article_urls=[], primary_section="ai")
        assert c.secondary_sections == []


# =============================================================================
# Orchestrator cache helper
# =============================================================================


class TestOrchestratorCacheHelper:
    def test_first_call_builds_and_persists(self, tmp_path):
        calls = {"n": 0}

        def builder() -> list[Cluster]:
            calls["n"] += 1
            return [Cluster(id="c0001", lede="x", article_urls=["u"], primary_section="ai")]

        out = _cached_or_fresh_clusters(tmp_path, builder)
        assert calls["n"] == 1
        assert (tmp_path / "assembly" / "clusters.json").exists()
        assert out[0].id == "c0001"

    def test_second_call_hits_cache(self, tmp_path):
        calls = {"n": 0}

        def builder() -> list[Cluster]:
            calls["n"] += 1
            return [Cluster(id="c0001", lede="x", article_urls=["u"], primary_section="ai")]

        _cached_or_fresh_clusters(tmp_path, builder)
        out = _cached_or_fresh_clusters(tmp_path, builder)
        assert calls["n"] == 1
        assert out[0].lede == "x"

    def test_no_run_dir_still_builds_no_persistence(self, tmp_path):
        def builder() -> list[Cluster]:
            return [Cluster(id="c0001", lede="x", article_urls=["u"], primary_section="ai")]

        out = _cached_or_fresh_clusters(None, builder)
        assert len(out) == 1
        assert not (tmp_path / "assembly" / "clusters.json").exists()

    def test_empty_cluster_list_persists_and_round_trips(self, tmp_path):
        out = _cached_or_fresh_clusters(tmp_path, list)
        assert out == []
        assert (tmp_path / "assembly" / "clusters.json").exists()
        out2 = _cached_or_fresh_clusters(
            tmp_path,
            lambda: [Cluster(id="c1", lede="x", article_urls=[], primary_section="ai")],
        )
        assert out2 == []  # cache hit, builder skipped
