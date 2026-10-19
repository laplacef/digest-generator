# Changelog

The format is based on [Common Changelog](https://common-changelog.org/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-28

_Digest output is now checked by a deterministic lint that runs on every write._

### Changed

- Normalize digest output deterministically, folding locale, scale, and notation drift ([`ab9e612`](https://github.com/laplacef/digest-generator/commit/ab9e612))
- Draft sections concurrently ([`4fe48b3`](https://github.com/laplacef/digest-generator/commit/4fe48b3))
- Batch topic inference across articles ([`d9cfd2c`](https://github.com/laplacef/digest-generator/commit/d9cfd2c))
- Diversify title framing shapes ([`4b0a7eb`](https://github.com/laplacef/digest-generator/commit/4b0a7eb))
- Rewrite vague thesis-shaped paragraph openers ([`3d99305`](https://github.com/laplacef/digest-generator/commit/3d99305))
- Normalize currency scale letters, so `$100 M` becomes `$100 million` ([`263142f`](https://github.com/laplacef/digest-generator/commit/263142f))

### Added

- `lint` command that checks a digest file and exits non-zero on error findings ([`73f960f`](https://github.com/laplacef/digest-generator/commit/73f960f))
- Automatic lint reporting after every digest write ([`73f960f`](https://github.com/laplacef/digest-generator/commit/73f960f))
- Importable golden-output lint over assembled digest markdown ([`b309ba7`](https://github.com/laplacef/digest-generator/commit/b309ba7))
- Link-target validation against a run's fetched corpus, catching invented or mutated citation URLs ([`91da782`](https://github.com/laplacef/digest-generator/commit/91da782))

### Fixed

- Emit the frontmatter summary as a whole sentence instead of a truncated fragment ([`5252db1`](https://github.com/laplacef/digest-generator/commit/5252db1))
- Prevent internal citation keys leaking into prose and require per-section citations ([`45cdbe4`](https://github.com/laplacef/digest-generator/commit/45cdbe4))

## [0.1.0] - 2026-06-22

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

[0.2.0]: https://github.com/laplacef/digest-generator/releases/tag/v0.2.0
[0.1.0]: https://github.com/laplacef/digest-generator/releases/tag/v0.1.0
