# Changelog

The format is based on [Common Changelog](https://common-changelog.org/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-10-19

_This release resolves configuration from a single directory and fixes the crashes, hangs, and lost work found since 0.2.0._

### Changed

- **Breaking:** Resolve `feeds.yaml` and prompt overrides from one config directory (`--config` / `DIGEST_CONFIG`, else `~/.config/digest-generator`); `./digest-generator/` in the working directory is no longer read, and prompt overrides no longer layer across directories ([#32](https://github.com/laplacef/digest-generator/pull/32))
- **Breaking:** Report a missing `--feeds` or `--config` target as an error instead of falling back to another location ([#32](https://github.com/laplacef/digest-generator/pull/32))
- Apply `--feeds` and `--config` to the digest sections and prompt overrides as well as the feed list ([#32](https://github.com/laplacef/digest-generator/pull/32))
- Reject generic trend verbs such as Reshape and Accelerate in titles, which the lint reports as errors ([`aab956d`](https://github.com/laplacef/digest-generator/commit/aab956d))
- Flag utilize and leverage as abstract verbs ([`e53212f`](https://github.com/laplacef/digest-generator/commit/e53212f))

### Added

- `--feeds` and `--config` options on `digest`, and `--config` on `summarize` ([#32](https://github.com/laplacef/digest-generator/pull/32))
- `write_digest` in the programmatic API ([#32](https://github.com/laplacef/digest-generator/pull/32))
- `title-missing` lint error for an empty frontmatter title ([#32](https://github.com/laplacef/digest-generator/pull/32))
- Author link in the package metadata ([`90751db`](https://github.com/laplacef/digest-generator/commit/90751db))

### Fixed

- Run the topic classifier on `cuda` and `mps` devices ([#32](https://github.com/laplacef/digest-generator/pull/32))
- Bound the feed fallback fetch by `FETCH_TIMEOUT`, so an unresponsive host no longer hangs the run ([#32](https://github.com/laplacef/digest-generator/pull/32))
- Write `<date>.md` from `api.run`, so `with_audio=True` finds the digest ([#32](https://github.com/laplacef/digest-generator/pull/32))
- Keep one failing article or section from discarding the rest of its batch ([#32](https://github.com/laplacef/digest-generator/pull/32))
- Keep the first title when the framer's retry comes back empty ([#32](https://github.com/laplacef/digest-generator/pull/32))
- Quote frontmatter values as YAML, so backslashes no longer break site builds ([#32](https://github.com/laplacef/digest-generator/pull/32))
- Keep section order and names when a run's section caches reload ([#32](https://github.com/laplacef/digest-generator/pull/32))
- Load a copied `.env.example` without reading comments as values ([#32](https://github.com/laplacef/digest-generator/pull/32))

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

[0.3.0]: https://github.com/laplacef/digest-generator/releases/tag/v0.3.0
[0.2.0]: https://github.com/laplacef/digest-generator/releases/tag/v0.2.0
[0.1.0]: https://github.com/laplacef/digest-generator/releases/tag/v0.1.0
