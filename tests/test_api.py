"""Tests for digest_generator/api.py: the public programmatic entry points.

Covers ``fetch``, ``summarize``, ``label``, ``digest``, and ``run``:
resume behavior, per-stage mocking, in-memory join correctness, and
composition wiring. Run setup (``run_dir``, ``run_context``,
``llm_telemetry``) is the caller's responsibility, and
``digest_generator.cli`` owns it for the command-line path.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from http.client import IncompleteRead
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from digest_generator.api import (
    _load_digest_input,
    digest,
    fetch,
    label,
    render_audio,
    run,
    summarize,
)
from digest_generator.core.label.io import (
    labeled_path,
    load_labeled,
    save_labeled,
)
from digest_generator.core.summary.io import save_summarized
from digest_generator.core.types import ContentType, Entry, Filter, Label, Summary, TopicType
from digest_generator.sources.rss.io import save_entries
from digest_generator.sources.rss.types import Feed

# =============================================================================
# Per-stage primitives: fetch, summarize, label
# =============================================================================


@pytest.fixture
def now():
    return datetime.now(tz=UTC)


@pytest.fixture
def feed():
    return Feed(
        name="openai-news",
        url="https://example.com/feed",
        content_type=ContentType.AI,
    )


def _entry(now, *, url="https://example.com/1", title="GPT-5", content_type=ContentType.AI):
    """Test entry fixture. ``content_type`` defaults to AI so iter_fetched
    in the flat layout can determine the section bucket from entry data
    without a directory tier. ``origin`` is the feed slug."""
    return Entry(
        title=title,
        url=url,
        origin="openai-news",
        published=now,
        description="d",
        content="c",
        content_head="head",
        content_type=content_type,
        fetched_at=now,
    )


class TestFetch:
    """api.fetch: per-feed fanout with cache-skip."""

    def test_fetches_and_persists(self, tmp_path, feed, now):
        fetcher = MagicMock()
        fetcher.fetch_entries = AsyncMock(return_value=[_entry(now)])

        asyncio.run(fetch([feed], Filter.resolve(days_back=7), run_dir=tmp_path, fetcher=fetcher))

        target = tmp_path / "source-fetched" / "openai-news.json"
        assert target.exists()
        fetcher.fetch_entries.assert_awaited_once()

    def test_resume_skips_cached_feed(self, tmp_path, feed, now):
        """Pre-populate the cache, then run fetch; fetcher must not be called."""
        save_entries(tmp_path, "openai-news", [_entry(now)])

        fetcher = MagicMock()
        fetcher.fetch_entries = AsyncMock(return_value=[])

        asyncio.run(fetch([feed], Filter.resolve(days_back=7), run_dir=tmp_path, fetcher=fetcher))

        fetcher.fetch_entries.assert_not_awaited()

    def test_one_failing_feed_does_not_kill_run(self, tmp_path, now):
        """A per-feed exception (e.g. ``IncompleteRead``, SSL handshake error)
        must surface as a warning, not propagate through ``asyncio.TaskGroup``
        and cancel sibling tasks. Without a per-feed try/except, a single feed
        returning a truncated response can cancel every other feed and kill the
        whole pipeline.
        """
        good_feed = Feed(
            name="good",
            url="https://example.com/good",
            content_type=ContentType.AI,
        )
        bad_feed = Feed(
            name="bad",
            url="https://example.com/bad",
            content_type=ContentType.SECURITY,
        )

        async def side_effect(feed, _filter):
            if feed.name == "bad":
                raise IncompleteRead(b"partial", 1000)  # noqa: EM101 (args, not message)
            return [
                _entry(now, url=f"https://example.com/{feed.name}", content_type=feed.content_type)
            ]

        fetcher = MagicMock()
        fetcher.fetch_entries = AsyncMock(side_effect=side_effect)

        # Must not raise; the bad feed is logged and skipped.
        asyncio.run(
            fetch(
                [good_feed, bad_feed],
                Filter.resolve(days_back=7),
                run_dir=tmp_path,
                fetcher=fetcher,
            )
        )

        # Both feeds were attempted (TaskGroup didn't cancel the good one).
        assert fetcher.fetch_entries.await_count == 2
        # The good feed persisted; the bad feed didn't (cache miss preserved).
        assert (tmp_path / "source-fetched" / "good.json").exists()
        assert not (tmp_path / "source-fetched" / "bad.json").exists()

    def test_partial_resume(self, tmp_path, now):
        """Mixed cache state: only the uncached feed is fetched."""
        cached_feed = Feed(
            name="cached",
            url="https://example.com/a",
            content_type=ContentType.AI,
        )
        new_feed = Feed(
            name="new",
            url="https://example.com/b",
            content_type=ContentType.SECURITY,
        )
        save_entries(tmp_path, "cached", [_entry(now)])

        fetcher = MagicMock()
        fetcher.fetch_entries = AsyncMock(return_value=[_entry(now, url="https://example.com/2")])

        asyncio.run(
            fetch(
                [cached_feed, new_feed],
                Filter.resolve(days_back=7),
                run_dir=tmp_path,
                fetcher=fetcher,
            )
        )

        # Only the new feed triggered a fetch.
        assert fetcher.fetch_entries.await_count == 1
        # Both feeds now have cache files.
        assert (tmp_path / "source-fetched" / "cached.json").exists()
        assert (tmp_path / "source-fetched" / "new.json").exists()


class TestSummarize:
    """api.summarize: iterate fetched, summarize, persist with cache-skip."""

    def test_summarizes_and_persists(self, tmp_path, now):
        save_entries(tmp_path, "openai-news", [_entry(now)])

        summarizer = MagicMock()
        summarizer.summarize_entries = AsyncMock(
            return_value=[
                Summary(entry=_entry(now), summary="extracted fact", length=14, topics=[])
            ]
        )

        asyncio.run(summarize(run_dir=tmp_path, summarizer=summarizer))

        target = tmp_path / "source-summarized" / "openai-news.json"
        assert target.exists()
        summarizer.summarize_entries.assert_awaited_once()

    def test_resume_skips_cached_feed(self, tmp_path, now):
        save_entries(tmp_path, "openai-news", [_entry(now)])
        save_summarized(
            tmp_path,
            "openai-news",
            [Summary(entry=_entry(now), summary="prior", length=5, topics=[])],
        )

        summarizer = MagicMock()
        summarizer.summarize_entries = AsyncMock(return_value=[])

        asyncio.run(summarize(run_dir=tmp_path, summarizer=summarizer))

        summarizer.summarize_entries.assert_not_awaited()

    def test_no_fetched_yields_no_calls(self, tmp_path):
        """Empty source-fetched/ means summarize is a no-op."""
        summarizer = MagicMock()
        summarizer.summarize_entries = AsyncMock(return_value=[])

        asyncio.run(summarize(run_dir=tmp_path, summarizer=summarizer))

        summarizer.summarize_entries.assert_not_awaited()


class TestLabel:
    """api.label: iterate fetched, classify, persist labeled with cache-skip."""

    def test_classifies_and_persists(self, tmp_path, now):
        save_entries(tmp_path, "openai-news", [_entry(now)])

        classifier = MagicMock()
        classifier.classify_entries = MagicMock(
            return_value=[[Label(value=TopicType.LLM, confidence=0.92)]]
        )

        asyncio.run(label(run_dir=tmp_path, classifier=classifier))

        target = tmp_path / "source-labeled" / "openai-news.json"
        assert target.exists()
        classifier.classify_entries.assert_called_once()

    def test_resume_skips_cached_feed(self, tmp_path, now):
        save_entries(tmp_path, "openai-news", [_entry(now)])
        # Pre-populate labeled cache.
        save_labeled(
            tmp_path,
            "openai-news",
            urls=["https://example.com/1"],
            labels_per_entry=[[Label(value=TopicType.LLM, confidence=0.5)]],
        )

        classifier = MagicMock()
        classifier.classify_entries = MagicMock(return_value=[])

        asyncio.run(label(run_dir=tmp_path, classifier=classifier))

        classifier.classify_entries.assert_not_called()

    def test_persists_url_keyed_labels(self, tmp_path, now):
        """The labeled payload is keyed by URL (not index)."""
        e1 = _entry(now, url="https://example.com/1", title="A")
        e2 = _entry(now, url="https://example.com/2", title="B")
        save_entries(tmp_path, "openai-news", [e1, e2])

        classifier = MagicMock()
        classifier.classify_entries = MagicMock(
            return_value=[
                [Label(value=TopicType.LLM, confidence=0.92)],
                [Label(value=TopicType.SAFETY, confidence=0.71)],
            ]
        )

        asyncio.run(label(run_dir=tmp_path, classifier=classifier))

        result = load_labeled(tmp_path, "openai-news")
        assert result is not None
        assert result["https://example.com/1"][0].value == TopicType.LLM
        assert result["https://example.com/2"][0].value == TopicType.SAFETY


# =============================================================================
# In-memory digest input: _load_digest_input joins summarized, labeled, fetched
# =============================================================================


class TestLoadDigestInput:
    """_load_digest_input joins source-summarized/ + source-labeled/ + source-fetched/ in memory."""

    def test_joins_summary_labels_and_content_type(self, tmp_path, now):
        entry = _entry(now, url="https://example.com/1", title="A")
        save_entries(tmp_path, "openai-news", [entry])
        save_summarized(
            tmp_path,
            "openai-news",
            [Summary(entry=entry, summary="extracted", length=9, topics=[])],
        )
        save_labeled(
            tmp_path,
            "openai-news",
            urls=["https://example.com/1"],
            labels_per_entry=[[Label(value=TopicType.LLM, confidence=0.92)]],
        )

        results = _load_digest_input(tmp_path)

        assert "openai-news" in results
        article = results["openai-news"][0]
        assert article["title"] == "A"
        assert article["summary"] == "extracted"
        assert article["content_type"] == "ai"
        assert article["topics"] == {"large-language-models": 0.92}

    def test_url_keyed_join_robust_to_reordering(self, tmp_path, now):
        """Labels lookup is by URL, not index, so reordering between stages is fine."""
        e1 = _entry(now, url="https://example.com/1", title="A")
        e2 = _entry(now, url="https://example.com/2", title="B")
        save_entries(tmp_path, "openai-news", [e1, e2])
        save_summarized(
            tmp_path,
            "openai-news",
            [
                Summary(entry=e1, summary="sum1", length=4, topics=[]),
                Summary(entry=e2, summary="sum2", length=4, topics=[]),
            ],
        )
        # Classifier persisted in REVERSED order; URL keying makes it harmless.
        save_labeled(
            tmp_path,
            "openai-news",
            urls=["https://example.com/2", "https://example.com/1"],
            labels_per_entry=[
                [Label(value=TopicType.SAFETY, confidence=0.71)],
                [Label(value=TopicType.LLM, confidence=0.92)],
            ],
        )

        results = _load_digest_input(tmp_path)

        by_url = {a["url"]: a for a in results["openai-news"]}
        assert by_url["https://example.com/1"]["topics"] == {"large-language-models": 0.92}
        assert by_url["https://example.com/2"]["topics"] == {"safety": 0.71}

    def test_missing_labeled_yields_empty_topics(self, tmp_path, now):
        """A summarized feed without labels still surfaces; articles get empty topics."""
        entry = _entry(now)
        save_entries(tmp_path, "openai-news", [entry])
        save_summarized(
            tmp_path,
            "openai-news",
            [Summary(entry=entry, summary="s", length=1, topics=[])],
        )
        # source-labeled/ deliberately not written.

        results = _load_digest_input(tmp_path)

        assert results["openai-news"][0]["topics"] == {}
        assert results["openai-news"][0]["content_type"] == "ai"


# =============================================================================
# api.run: composition (fetch, then summarize alongside label, then digest)
# =============================================================================


class TestRunComposition:
    def test_run_produces_summarized_and_labeled(self, tmp_path, feed, now):
        """Full corpus build (with_digest=False) writes both stage outputs to disk."""
        fetcher = MagicMock()
        fetcher.fetch_entries = AsyncMock(return_value=[_entry(now)])
        summarizer = MagicMock()
        summarizer.summarize_entries = AsyncMock(
            return_value=[Summary(entry=_entry(now), summary="s", length=1, topics=[])]
        )
        classifier = MagicMock()
        classifier.classify_entries = MagicMock(
            return_value=[[Label(value=TopicType.LLM, confidence=0.5)]]
        )

        run(
            [feed],
            Filter.resolve(days_back=7),
            run_dir=tmp_path,
            fetcher=fetcher,
            summarizer=summarizer,
            classifier=classifier,
            with_digest=False,
        )

        assert (tmp_path / "source-summarized" / "openai-news.json").exists()
        assert labeled_path(tmp_path, "openai-news").exists()

    def test_run_skips_digest_when_with_digest_false(self, tmp_path, feed, now):
        fetcher = MagicMock()
        fetcher.fetch_entries = AsyncMock(return_value=[_entry(now)])
        summarizer = MagicMock()
        summarizer.summarize_entries = AsyncMock(
            return_value=[Summary(entry=_entry(now), summary="s", length=1, topics=[])]
        )
        classifier = MagicMock()
        classifier.classify_entries = MagicMock(
            return_value=[[Label(value=TopicType.LLM, confidence=0.5)]]
        )

        with patch("digest_generator.api.digest") as digest_spy:
            run(
                [feed],
                Filter.resolve(days_back=7),
                run_dir=tmp_path,
                fetcher=fetcher,
                summarizer=summarizer,
                classifier=classifier,
                with_digest=False,
            )

        digest_spy.assert_not_called()

    def test_run_calls_digest_when_with_digest_true(self, tmp_path, feed, now):
        fetcher = MagicMock()
        fetcher.fetch_entries = AsyncMock(return_value=[_entry(now)])
        summarizer = MagicMock()
        summarizer.summarize_entries = AsyncMock(
            return_value=[Summary(entry=_entry(now), summary="s", length=1, topics=[])]
        )
        classifier = MagicMock()
        classifier.classify_entries = MagicMock(
            return_value=[[Label(value=TopicType.LLM, confidence=0.5)]]
        )

        with patch("digest_generator.api.digest") as digest_spy:
            run(
                [feed],
                Filter.resolve(days_back=7),
                run_dir=tmp_path,
                fetcher=fetcher,
                summarizer=summarizer,
                classifier=classifier,
                with_digest=True,
            )

        digest_spy.assert_called_once()
        assert digest_spy.call_args.kwargs["run_dir"] == tmp_path


# =============================================================================
# api.digest: joins source-summarized/ and source-labeled/ in memory
# =============================================================================


class TestDigestEntryPoint:
    def test_returns_none_when_summarized_empty(self, tmp_path):
        """No source-summarized/ tree means digest returns None and doesn't call run_digest_from_json."""
        with patch("digest_generator.core.digest.orchestrator.run_digest_from_json") as runner:
            result = digest(run_dir=tmp_path)
        assert result is None
        runner.assert_not_called()

    def test_calls_run_digest_from_json_with_joined_payload(self, tmp_path, now):
        """When source-summarized/ has data, digest joins labels and forwards the merged shape."""
        entry = _entry(now)
        save_entries(tmp_path, "openai-news", [entry])
        save_summarized(
            tmp_path,
            "openai-news",
            [Summary(entry=entry, summary="s", length=1, topics=[])],
        )
        save_labeled(
            tmp_path,
            "openai-news",
            urls=[entry.url],
            labels_per_entry=[[Label(value=TopicType.LLM, confidence=0.5)]],
        )

        with patch("digest_generator.core.digest.orchestrator.run_digest_from_json") as runner:
            runner.return_value = MagicMock()
            digest(run_dir=tmp_path)

        runner.assert_called_once()
        forwarded = runner.call_args.args[0]
        assert "openai-news" in forwarded
        article = forwarded["openai-news"][0]
        assert article["topics"] == {"large-language-models": 0.5}
        assert article["content_type"] == "ai"


class TestRenderAudio:
    """``api.render_audio`` discovers the digest .md, runs the renderer,
    harvests telemetry into meta.json's ``stages.audio`` + ``models.audio``."""

    def _make_digest(self, run_dir):
        md = run_dir / "2026-05-11-weekly.md"
        md.write_text("# Weekly\n\nBody.\n")
        return md

    def _make_meta(self, run_dir):
        (run_dir / "meta.json").write_text('{"schema_version": 2, "stages": {}, "models": {}}')

    def test_returns_opus_path_from_artifact(self, tmp_path):
        self._make_digest(tmp_path)
        mock_renderer = MagicMock()
        mock_renderer.render.return_value = MagicMock(
            opus_path=tmp_path / "audio" / "2026-05-11-weekly.opus",
            voice_id="en_US-amy-medium",
            bitrate_kbps=24,
            narration_chars=100,
            audio_bytes=50000,
            audio_duration_s=16.7,
            cached=False,
        )
        result = render_audio(run_dir=tmp_path, renderer=mock_renderer)
        assert result == tmp_path / "audio" / "2026-05-11-weekly.opus"
        mock_renderer.render.assert_called_once()

    def test_writes_stage_telemetry_to_meta(self, tmp_path):
        self._make_digest(tmp_path)
        self._make_meta(tmp_path)
        # A tiny sleep inside render makes ``duration_ms > 0`` so the
        # real_time_factor computation in the api harvest actually fires.
        artifact = MagicMock(
            opus_path=tmp_path / "audio" / "x.opus",
            voice_id="en_US-amy-medium",
            bitrate_kbps=24,
            narration_chars=8421,
            audio_bytes=1_238_400,
            audio_duration_s=412.3,
            cached=False,
        )

        def render_with_delay(_run_dir, _md_path):
            time.sleep(0.002)
            return artifact

        mock_renderer = MagicMock()
        mock_renderer.render.side_effect = render_with_delay
        render_audio(run_dir=tmp_path, renderer=mock_renderer)

        meta = json.loads((tmp_path / "meta.json").read_text())
        audio = meta["stages"]["audio"]
        assert audio["voice"] == "en_US-amy-medium"
        assert audio["bitrate_kbps"] == 24
        assert audio["narration_chars"] == 8421
        assert audio["audio_bytes"] == 1_238_400
        assert audio["audio_duration_s"] == 412.3
        assert audio["cached"] is False
        assert audio["duration_ms"] >= 0
        # real_time_factor present on fresh renders.
        assert "real_time_factor" in audio
        # models map updated with piper:<voice_id>.
        assert meta["models"]["audio"] == "piper:en_US-amy-medium"

    def test_cache_hit_omits_real_time_factor(self, tmp_path):
        self._make_digest(tmp_path)
        self._make_meta(tmp_path)
        mock_renderer = MagicMock()
        mock_renderer.render.return_value = MagicMock(
            opus_path=tmp_path / "audio" / "x.opus",
            voice_id="en_US-amy-medium",
            bitrate_kbps=24,
            narration_chars=0,
            audio_bytes=1_238_400,
            audio_duration_s=412.3,
            cached=True,
        )
        render_audio(run_dir=tmp_path, renderer=mock_renderer)

        meta = json.loads((tmp_path / "meta.json").read_text())
        audio = meta["stages"]["audio"]
        assert audio["cached"] is True
        # Cache hit: synthesis didn't run, RTF would be meaningless.
        assert "real_time_factor" not in audio

    def test_no_meta_json_skips_telemetry_silently(self, tmp_path):
        """Programmatic callers without meta.json must still get the .opus path."""
        self._make_digest(tmp_path)
        mock_renderer = MagicMock()
        mock_renderer.render.return_value = MagicMock(
            opus_path=tmp_path / "audio" / "x.opus",
            voice_id="en_US-amy-medium",
            bitrate_kbps=24,
            narration_chars=10,
            audio_bytes=100,
            audio_duration_s=0.05,
            cached=False,
        )
        # Must not raise even though meta.json is absent.
        result = render_audio(run_dir=tmp_path, renderer=mock_renderer)
        assert result == tmp_path / "audio" / "x.opus"
        assert not (tmp_path / "meta.json").exists()

    def test_missing_digest_md_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no digest markdown"):
            render_audio(run_dir=tmp_path)

    def test_bitrate_override_passed_to_renderer(self, tmp_path):
        """Caller-supplied bitrate flows when no renderer is injected."""
        self._make_digest(tmp_path)
        mock_voice = MagicMock(voice_id="en_US-amy-medium")
        # Mock the registry + AudioRenderer construction.
        with (
            patch("digest_generator.shared.tts.registry.voice_registry") as reg,
            patch("digest_generator.core.audio.renderer.AudioRenderer") as cls,
        ):
            reg.default = mock_voice
            instance = cls.return_value
            instance.render.return_value = MagicMock(
                opus_path=tmp_path / "audio" / "x.opus",
                voice_id="en_US-amy-medium",
                bitrate_kbps=48,
                narration_chars=10,
                audio_bytes=100,
                audio_duration_s=0.05,
                cached=False,
            )
            render_audio(run_dir=tmp_path, bitrate_kbps=48)

        cls.assert_called_once()
        kwargs = cls.call_args.kwargs
        assert kwargs["bitrate_kbps"] == 48

    def test_run_with_audio_invokes_render_audio(self, tmp_path, feed):
        """``api.run(with_audio=True)`` chains digest then audio."""
        # Mock the underlying corpus build + digest.
        with (
            patch("digest_generator.api.fetch", new_callable=AsyncMock) as mock_fetch,
            patch("digest_generator.api.summarize", new_callable=AsyncMock) as mock_sum,
            patch("digest_generator.api.label", new_callable=AsyncMock) as mock_lbl,
            patch("digest_generator.api.digest") as mock_digest,
            patch("digest_generator.api.render_audio") as mock_audio,
        ):
            mock_audio.return_value = tmp_path / "audio" / "x.opus"
            run([feed], Filter.resolve(days_back=7), run_dir=tmp_path, with_audio=True)

        mock_fetch.assert_called_once()
        mock_sum.assert_called_once()
        mock_lbl.assert_called_once()
        mock_digest.assert_called_once()
        mock_audio.assert_called_once_with(run_dir=tmp_path)

    def test_with_audio_requires_with_digest(self, tmp_path, feed):
        """Audio needs the digest .md; with_digest=False + with_audio=True is invalid."""
        with (
            patch("digest_generator.api.fetch", new_callable=AsyncMock),
            patch("digest_generator.api.summarize", new_callable=AsyncMock),
            patch("digest_generator.api.label", new_callable=AsyncMock),
            pytest.raises(ValueError, match="with_audio=True requires with_digest=True"),
        ):
            run(
                [feed],
                Filter.resolve(days_back=7),
                run_dir=tmp_path,
                with_digest=False,
                with_audio=True,
            )
