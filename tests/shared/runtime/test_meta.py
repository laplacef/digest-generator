"""Tests for digest_generator/shared/runtime/meta.py: schema v2 RunMeta + lifecycle helpers."""

import json

from digest_generator.shared.runtime.meta import (
    SCHEMA_VERSION,
    DigestMeta,
    RunMeta,
    SamplingLayer,
    SamplingMeta,
    SectionMeta,
    StageMeta,
    update_run_meta_digest,
    update_run_meta_telemetry,
    write_run_meta,
)


class TestStageMeta:
    """StageMeta: typed common fields + freeform extras."""

    def test_default_construction(self):
        s = StageMeta()
        assert s.duration_ms == 0
        assert s.llm_calls == 0
        assert s.model is None
        assert s.extras == {}

    def test_to_dict_flattens_extras(self):
        s = StageMeta(
            duration_ms=500,
            llm_calls=2,
            prompt_tokens=1000,
            completion_tokens=200,
            llm_duration_ms=499,
            model="writer-model",
            extras={"sections": 5, "articles": 169},
        )
        d = s.to_dict()
        assert d["duration_ms"] == 500
        assert d["sections"] == 5
        assert d["articles"] == 169
        assert "extras" not in d

    def test_to_dict_omits_model_when_none(self):
        d = StageMeta(duration_ms=10).to_dict()
        assert "model" not in d

    def test_extras_can_carry_lists(self):
        s = StageMeta(extras={"title_retry_reasons": ["ai_prefix", "colon_split"]})
        assert s.to_dict()["title_retry_reasons"] == ["ai_prefix", "colon_split"]


class TestSamplingMeta:
    """SamplingMeta: three-layer sampling state with seed_source."""

    def test_default_construction(self):
        s = SamplingMeta()
        assert s.user.temperature is None
        assert s.model_defaults.temperature is None
        assert s.effective.temperature is None
        assert s.seed_source == "random"

    def test_seed_source_values(self):
        # Just exercises that all four labels are accepted by the type checker.
        for src in ("user", "settings", "cli", "random"):
            s = SamplingMeta(seed_source=src)
            assert s.seed_source == src


class TestRunMeta:
    """RunMeta: root v2 dataclass."""

    def test_default_carries_schema_version(self):
        meta = RunMeta()
        assert meta.schema_version == SCHEMA_VERSION
        assert meta.schema_version == 2

    def test_construction_with_minimal_fields(self):
        meta = RunMeta(
            timestamp="2026-05-09T12:00:00+00:00",
            feed_count=5,
            article_count=42,
            content_types=["ai", "security"],
            duration_seconds=123.45,
        )
        assert meta.feed_count == 5
        assert meta.models == {}
        assert meta.sampling == {}
        assert meta.stages == {}
        assert meta.digest is None

    def test_full_construction(self):
        meta = RunMeta(
            timestamp="2026-05-09T12:00:00+00:00",
            feed_count=75,
            article_count=169,
            content_types=["ai", "engineering", "security"],
            duration_seconds=540.0,
            models={"writer": "gemma4:31b-cloud", "editor": "gpt-oss:120b-cloud"},
            sampling={
                "writer": SamplingMeta(
                    user=SamplingLayer(seed=None),
                    model_defaults=SamplingLayer(
                        temperature=0.7, top_p=0.9, repetition_penalty=1.1, seed=0
                    ),
                    effective=SamplingLayer(
                        temperature=0.7, top_p=0.9, repetition_penalty=1.1, seed=42
                    ),
                    seed_source="cli",
                ),
            },
            stages={
                "writer": StageMeta(
                    duration_ms=180000,
                    llm_calls=9,
                    prompt_tokens=147489,
                    completion_tokens=6695,
                    model="gemma4:31b-cloud",
                    extras={"sections": 5, "articles": 169},
                ),
            },
            totals={"llm_calls": 17, "prompt_tokens": 200000, "completion_tokens": 25000},
            sections=[
                SectionMeta(
                    name="security",
                    articles=33,
                    edit_outcome="fell_back",
                    rejected_reason="link_set",
                ),
            ],
            digest=DigestMeta(
                title="Funding Frenzy Meets Agentic Security Threats",
                filename="2026-05-03-funding-frenzy.md",
                word_count=2469,
                reading_time_minutes=12,
                article_count=169,
                section_counts={"security": 33, "engineering": 25},
                watch_item_count=3,
            ),
        )
        assert meta.sampling["writer"].seed_source == "cli"
        assert meta.sections[0].rejected_reason == "link_set"
        assert meta.digest is not None
        assert meta.digest.word_count == 2469


class TestWriteRunMeta:
    """write_run_meta: root JSON write + extras flattening + nested serialization."""

    def test_writes_schema_version(self, tmp_path):
        write_run_meta(RunMeta(timestamp="t", duration_seconds=1.0), tmp_path)
        data = json.loads((tmp_path / "meta.json").read_text())
        assert data["schema_version"] == 2

    def test_stage_extras_flatten_into_stage_object(self, tmp_path):
        meta = RunMeta(
            stages={
                "framer": StageMeta(
                    duration_ms=16000,
                    llm_calls=2,
                    extras={"title_retried": False, "title_chars": 45, "intro_words": 87},
                ),
            }
        )
        write_run_meta(meta, tmp_path)
        data = json.loads((tmp_path / "meta.json").read_text())
        framer = data["stages"]["framer"]
        # Common fields present.
        assert framer["duration_ms"] == 16000
        assert framer["llm_calls"] == 2
        # Extras flattened in, not nested.
        assert framer["title_retried"] is False
        assert framer["title_chars"] == 45
        assert framer["intro_words"] == 87
        assert "extras" not in framer

    def test_sampling_block_serializes_three_layers(self, tmp_path):
        meta = RunMeta(
            sampling={
                "writer": SamplingMeta(
                    user=SamplingLayer(seed=42),
                    model_defaults=SamplingLayer(temperature=0.7),
                    effective=SamplingLayer(temperature=0.7, seed=42),
                    seed_source="user",
                ),
            }
        )
        write_run_meta(meta, tmp_path)
        data = json.loads((tmp_path / "meta.json").read_text())
        w = data["sampling"]["writer"]
        assert w["user"]["seed"] == 42
        assert w["user"]["temperature"] is None  # explicit null preserved
        assert w["model_defaults"]["temperature"] == 0.7
        assert w["effective"]["seed"] == 42
        assert w["seed_source"] == "user"

    def test_digest_serializes_when_present(self, tmp_path):
        meta = RunMeta(
            digest=DigestMeta(
                title="Test",
                filename="test.md",
                word_count=100,
                section_counts={"ai": 5},
            )
        )
        write_run_meta(meta, tmp_path)
        data = json.loads((tmp_path / "meta.json").read_text())
        assert data["digest"]["title"] == "Test"
        assert data["digest"]["section_counts"] == {"ai": 5}

    def test_digest_null_when_absent(self, tmp_path):
        write_run_meta(RunMeta(), tmp_path)
        data = json.loads((tmp_path / "meta.json").read_text())
        assert data["digest"] is None

    def test_sections_serialize_as_list(self, tmp_path):
        meta = RunMeta(
            sections=[
                SectionMeta(name="ai", articles=47, edit_outcome="rewritten"),
                SectionMeta(
                    name="security",
                    articles=33,
                    edit_outcome="fell_back",
                    rejected_reason="link_set",
                ),
            ]
        )
        write_run_meta(meta, tmp_path)
        data = json.loads((tmp_path / "meta.json").read_text())
        assert len(data["sections"]) == 2
        assert data["sections"][1]["rejected_reason"] == "link_set"


class TestNormalizeV2OnPatch:
    """Patch helpers re-canonicalize on every write: strip legacy v1 keys, stamp schema_version."""

    def test_digest_patch_drops_legacy_flat_model_keys(self, tmp_path):
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "summarizer_model": "gemma4:31b-cloud",
                    "topic_model": "facebook/bart-large-mnli",
                    "writer_model": None,
                    "digest_title": "Old Title",
                    "digest_filename": "old.md",
                    "feed_count": 5,
                }
            )
        )
        update_run_meta_digest(tmp_path, "New Title", "new.md")
        data = json.loads(meta_path.read_text())
        for legacy in (
            "summarizer_model",
            "topic_model",
            "writer_model",
            "digest_title",
            "digest_filename",
        ):
            assert legacy not in data
        assert data["schema_version"] == 2
        assert data["digest"]["title"] == "New Title"
        assert data["feed_count"] == 5

    def test_digest_patch_promotes_topic_revision_into_revisions(self, tmp_path):
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps({"topic_revision": "abc123", "summarizer_revision": ""}))
        update_run_meta_digest(tmp_path, "T", "f.md")
        data = json.loads(meta_path.read_text())
        assert data["revisions"] == {"topic": "abc123"}
        assert "topic_revision" not in data
        assert "summarizer_revision" not in data

    def test_telemetry_patch_drops_legacy_keys_and_stamps_schema(self, tmp_path):
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(
            json.dumps({"editorial_model": "gpt-oss:120b-cloud", "framer_model": None})
        )
        update_run_meta_telemetry(tmp_path, totals={"llm_calls": 17})
        data = json.loads(meta_path.read_text())
        assert "editorial_model" not in data
        assert "framer_model" not in data
        assert data["schema_version"] == 2
        assert data["totals"] == {"llm_calls": 17}

    def test_nested_revisions_wins_over_legacy_flat(self, tmp_path):
        # If both forms are present (mid-migration corner case), the nested
        # value is authoritative; legacy flat is dropped silently.
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps({"revisions": {"topic": "new"}, "topic_revision": "old"}))
        update_run_meta_digest(tmp_path, "T", "f.md")
        data = json.loads(meta_path.read_text())
        assert data["revisions"] == {"topic": "new"}
        assert "topic_revision" not in data


class TestUpdateRunMetaDigest:
    """update_run_meta_digest: backward-compatible CLI hook for title + filename only."""

    def test_initializes_digest_block_when_missing(self, tmp_path):
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps({"schema_version": 2, "feed_count": 5}))
        update_run_meta_digest(tmp_path, "AI Agents", "2026-03-17-ai-agents.md")
        data = json.loads(meta_path.read_text())
        assert data["digest"]["title"] == "AI Agents"
        assert data["digest"]["filename"] == "2026-03-17-ai-agents.md"

    def test_preserves_other_top_level_fields(self, tmp_path):
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(
            json.dumps(
                {"schema_version": 2, "feed_count": 5, "stages": {"writer": {"duration_ms": 100}}}
            )
        )
        update_run_meta_digest(tmp_path, "Title", "fname.md")
        data = json.loads(meta_path.read_text())
        assert data["feed_count"] == 5
        assert data["stages"]["writer"]["duration_ms"] == 100

    def test_preserves_existing_digest_fields(self, tmp_path):
        # If the orchestrator's telemetry harvest ran first, the digest block
        # already carries word_count etc.; the CLI's title patch must not erase them.
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(
            json.dumps({"schema_version": 2, "digest": {"word_count": 2400, "watch_item_count": 3}})
        )
        update_run_meta_digest(tmp_path, "Title", "fname.md")
        data = json.loads(meta_path.read_text())
        assert data["digest"]["title"] == "Title"
        assert data["digest"]["word_count"] == 2400
        assert data["digest"]["watch_item_count"] == 3

    def test_creates_stub_when_meta_json_absent(self, tmp_path):
        # The per-stage CLI workflow can leave meta.json absent (cli.run
        # crashed before _write_initial_meta, or the user ran summarize/label/
        # digest as separate commands). The patch helper must create a v2 stub
        # rather than crash with FileNotFoundError.
        meta_path = tmp_path / "meta.json"
        assert not meta_path.exists()
        update_run_meta_digest(tmp_path, "Stub Title", "stub.md")
        data = json.loads(meta_path.read_text())
        assert data["schema_version"] == 2
        assert data["digest"]["title"] == "Stub Title"
        assert data["digest"]["filename"] == "stub.md"


class TestUpdateRunMetaTelemetry:
    """update_run_meta_telemetry: orchestrator's bulk patch for stage/sampling/section/totals/digest blocks."""

    def test_patches_stages_block(self, tmp_path):
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps({"schema_version": 2, "feed_count": 5}))
        update_run_meta_telemetry(
            tmp_path,
            stages={
                "writer": StageMeta(duration_ms=100, llm_calls=3, extras={"sections": 5}),
            },
        )
        data = json.loads(meta_path.read_text())
        assert data["stages"]["writer"]["duration_ms"] == 100
        assert data["stages"]["writer"]["sections"] == 5
        assert data["feed_count"] == 5  # preserved

    def test_patches_sampling_block(self, tmp_path):
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps({"schema_version": 2}))
        update_run_meta_telemetry(
            tmp_path,
            sampling={
                "framer": SamplingMeta(
                    effective=SamplingLayer(temperature=0.7, seed=42),
                    seed_source="cli",
                ),
            },
        )
        data = json.loads(meta_path.read_text())
        assert data["sampling"]["framer"]["effective"]["seed"] == 42
        assert data["sampling"]["framer"]["seed_source"] == "cli"

    def test_patches_sections_block(self, tmp_path):
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps({"schema_version": 2}))
        update_run_meta_telemetry(
            tmp_path,
            sections=[
                SectionMeta(
                    name="security",
                    articles=33,
                    edit_outcome="fell_back",
                    rejected_reason="link_set",
                )
            ],
        )
        data = json.loads(meta_path.read_text())
        assert data["sections"][0]["rejected_reason"] == "link_set"

    def test_patches_totals_block(self, tmp_path):
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps({"schema_version": 2}))
        update_run_meta_telemetry(
            tmp_path,
            totals={"llm_calls": 17, "prompt_tokens": 200000, "completion_tokens": 25000},
        )
        data = json.loads(meta_path.read_text())
        assert data["totals"]["llm_calls"] == 17

    def test_patches_digest_block(self, tmp_path):
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps({"schema_version": 2}))
        update_run_meta_telemetry(
            tmp_path,
            digest=DigestMeta(
                title="Title",
                filename="f.md",
                word_count=2469,
                reading_time_minutes=12,
                article_count=169,
                section_counts={"ai": 47},
                watch_item_count=3,
            ),
        )
        data = json.loads(meta_path.read_text())
        assert data["digest"]["word_count"] == 2469
        assert data["digest"]["section_counts"] == {"ai": 47}

    def test_unspecified_blocks_left_alone(self, tmp_path):
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "stages": {"writer": {"duration_ms": 999}},
                    "totals": {"llm_calls": 99},
                }
            )
        )
        # Patch only stages; totals must remain. New stage entries merge per
        # key into the existing block so concurrent api.summarize and api.label
        # writes don't clobber each other.
        update_run_meta_telemetry(tmp_path, stages={"editor": StageMeta(duration_ms=50)})
        data = json.loads(meta_path.read_text())
        # Existing 'writer' entry preserved; new 'editor' entry added.
        assert data["stages"]["writer"] == {"duration_ms": 999}
        assert data["stages"]["editor"]["duration_ms"] == 50
        assert data["totals"]["llm_calls"] == 99

    def test_stages_merge_per_key(self, tmp_path):
        """Successive stages-patches add new keys instead of replacing the block."""
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(
            json.dumps({"schema_version": 2, "stages": {"summarizer": {"duration_ms": 100}}})
        )
        update_run_meta_telemetry(tmp_path, stages={"writer": StageMeta(duration_ms=200)})
        data = json.loads(meta_path.read_text())
        assert "summarizer" in data["stages"]
        assert "writer" in data["stages"]

    def test_sampling_merges_per_key(self, tmp_path):
        """Successive sampling-patches add new keys instead of replacing the block."""
        meta_path = tmp_path / "meta.json"
        meta_path.write_text(json.dumps({"schema_version": 2}))
        update_run_meta_telemetry(
            tmp_path, sampling={"summarizer": SamplingMeta(seed_source="random")}
        )
        update_run_meta_telemetry(tmp_path, sampling={"writer": SamplingMeta(seed_source="cli")})
        data = json.loads(meta_path.read_text())
        assert data["sampling"]["summarizer"]["seed_source"] == "random"
        assert data["sampling"]["writer"]["seed_source"] == "cli"

    def test_creates_stub_when_meta_json_absent(self, tmp_path):
        # The per-stage CLI workflow can leave meta.json absent. The
        # orchestrator's terminal _persist_telemetry call must create the
        # file rather than crash with FileNotFoundError.
        meta_path = tmp_path / "meta.json"
        assert not meta_path.exists()
        update_run_meta_telemetry(
            tmp_path,
            stages={"writer": StageMeta(duration_ms=500, llm_calls=3)},
            totals={
                "llm_calls": 3,
                "prompt_tokens": 1500,
                "completion_tokens": 600,
                "llm_duration_ms": 500,
            },
        )
        data = json.loads(meta_path.read_text())
        assert data["schema_version"] == 2
        assert data["stages"]["writer"]["duration_ms"] == 500
        assert data["totals"]["prompt_tokens"] == 1500
