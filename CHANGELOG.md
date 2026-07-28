# Changelog

The format is based on [Common Changelog](https://common-changelog.org/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-28

_Digest output is now checked by a deterministic lint that runs on every write._

### Changed

- Normalize digest output deterministically, folding locale, scale, and notation drift ([`484f495`](https://github.com/laplacef/digest-generator/commit/484f495))
- Draft sections concurrently ([`f858ffb`](https://github.com/laplacef/digest-generator/commit/f858ffb))
- Batch topic inference across articles ([`3f26140`](https://github.com/laplacef/digest-generator/commit/3f26140))
- Diversify title framing shapes ([`7193117`](https://github.com/laplacef/digest-generator/commit/7193117))
- Rewrite vague thesis-shaped paragraph openers ([`a671938`](https://github.com/laplacef/digest-generator/commit/a671938))
- Normalize currency scale letters, so `$100 M` becomes `$100 million` ([`5d8e19b`](https://github.com/laplacef/digest-generator/commit/5d8e19b))

### Added

- `lint` command that checks a digest file and exits non-zero on error findings ([`284b5c1`](https://github.com/laplacef/digest-generator/commit/284b5c1))
- Automatic lint reporting after every digest write ([`284b5c1`](https://github.com/laplacef/digest-generator/commit/284b5c1))
- Importable golden-output lint over assembled digest markdown ([`1103abb`](https://github.com/laplacef/digest-generator/commit/1103abb))
- Link-target validation against a run's fetched corpus, catching invented or mutated citation URLs ([`10fe62d`](https://github.com/laplacef/digest-generator/commit/10fe62d))

### Fixed

- Emit the frontmatter summary as a whole sentence instead of a truncated fragment ([`5762033`](https://github.com/laplacef/digest-generator/commit/5762033))
- Prevent internal citation keys leaking into prose and require per-section citations ([`efa0fa2`](https://github.com/laplacef/digest-generator/commit/efa0fa2))

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

[0.2.0]: https://github.com/laplacef/digest-generator/releases/tag/v0.2.0
[0.1.0]: https://github.com/laplacef/digest-generator/releases/tag/v0.1.0
