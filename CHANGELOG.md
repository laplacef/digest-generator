# Changelog

The format is based on [Common Changelog](https://common-changelog.org/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-20

Initial public release.

### Added

- RSS fetch stage with concurrent feed retrieval, content extraction, and quality filtering
- Per-article summarization via local or cloud Ollama LLMs
- Zero-shot topic classification using BART-MNLI
- Six-stage digest pipeline (cluster, write, edit, frame, watch, compose) producing markdown digests
- Opt-in audio rendering of the digest to Opus via Piper TTS + ffmpeg
- Per-stage on-disk persistence so re-running a pipeline skips completed work
- User-defined feeds and sections via `feeds.yaml`, with `digest-generator init` to scaffold a starter
- Generic baseline prompts for every stage with a user override layer, so the tool runs on any topic out of the box
- `digest-generator` CLI with `init`, `run`, `fetch`, `summarize`, `label`, `digest`, `audio`, and `feeds` subcommands
- Programmatic API exposed through `digest_generator.api`
- Centralized configuration via environment variables / `.env`
- Structured logging with per-run log files and secret redaction

[0.1.0]: https://github.com/laplacef/digest-generator/releases/tag/v0.1.0
