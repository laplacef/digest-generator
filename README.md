# Digest Generator

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/laplacef/digest-generator/blob/main/LICENSE.md)

Digest Generator is a Python pipeline that aggregates articles from RSS feeds you define, generates fact-dense per-article summaries via an LLM, classifies them with zero-shot NLI, and produces a Markdown digest via Ollama. Feeds, sections, and prompts are all user-supplied; the tool ships generic baselines so it runs on any topic out of the box.

## How It Works

A run has two halves that share one run directory:

1. **Corpus.** Fetch every feed, then summarize each article with an LLM and classify its topics with BART-MNLI, in parallel.
2. **Digest.** Cluster articles into stories, draft each section, edit it, write the title and intro, pick what to watch, and compose `<date>.md`.

`digest-generator run` does both. `run --no-digest` stops after the corpus, and `digest <run_dir>` turns an existing corpus into a digest. Each stage caches its output, so re-running against the same directory skips finished work. The stage-by-stage view is in [`docs/usage.md`](https://github.com/laplacef/digest-generator/blob/main/docs/usage.md).

## Requirements

- **Python.** Version 3.13 or later.
- **Ollama.** The summarizer and every digest stage call [Ollama](https://ollama.com). The default models (`gemma4:31b-cloud`, `gpt-oss:120b-cloud`) are Ollama cloud models: set `OLLAMA_API_KEY` to call ollama.com directly, or use a local Ollama signed in to an Ollama account. To stay fully local, pull local models and set the six `*_MODEL` variables listed in [`.env.example`](https://github.com/laplacef/digest-generator/blob/main/.env.example).
- **HuggingFace token.** `run` and `label` log in to the HuggingFace Hub before the topic classifier loads, so they fail without `HF_TOKEN` ([create a token](https://huggingface.co/settings/tokens)). The classifier model (BART-MNLI, about 1.6 GB) downloads on first use.
- **Optional extras.** Piper and ffmpeg for audio rendering, and an NVIDIA or Apple Silicon GPU for the classifier. Both are covered in [`docs/setup.md`](https://github.com/laplacef/digest-generator/blob/main/docs/setup.md).

## Installation

```bash
uv tool install digest-generator      # or: pipx install digest-generator
digest-generator init                 # write a starter feeds.yaml
```

Inside an activated virtual environment, `pip install digest-generator` works too. `init` creates `~/.config/digest-generator/feeds.yaml` from a starter template. Edit it to define your own sections (`categories:`) and the feeds in each, then run `digest-generator feeds` to check it. Put `HF_TOKEN` (and `OLLAMA_API_KEY` for cloud Ollama) in a `.env` file in the directory you run from, or export them in your shell. [`.env.example`](https://github.com/laplacef/digest-generator/blob/main/.env.example) lists every other setting with its default.

A small first run keeps the LLM cost low:

```bash
digest-generator run -d 1 --limit 3   # 1 day back, at most 3 articles per feed
```

The digest lands at `output/<run_id>/<date>.md`. To work from a clone instead, follow [`docs/setup.md`](https://github.com/laplacef/digest-generator/blob/main/docs/setup.md), which also covers audio, GPU, and troubleshooting.

## Usage

```bash
digest-generator run                  # full pipeline (fetch + summarize + classify + digest)
digest-generator run --no-digest      # corpus build only (skip digest generation)
digest-generator run --audio          # full pipeline + Piper TTS rendition
digest-generator digest <run_dir>     # regenerate the digest from an existing run directory
digest-generator audio <run_dir>      # render audio for an existing digest (no LLM cost)
digest-generator lint <digest.md>     # check a digest before publishing (exits 1 on errors)
digest-generator feeds                # list available feeds
```

See [`docs/usage.md`](https://github.com/laplacef/digest-generator/blob/main/docs/usage.md) for the full CLI reference, caching, the programmatic API, and the output layout.

## Contributing

Bug reports, feature requests, and pull requests are welcome; [CONTRIBUTING.md](https://github.com/laplacef/digest-generator/blob/main/CONTRIBUTING.md) covers development setup and the workflow, and participation follows the [Code of Conduct](https://github.com/laplacef/digest-generator/blob/main/CODE_OF_CONDUCT.md).

## License

This project is licensed under the [Apache License 2.0](https://github.com/laplacef/digest-generator/blob/main/LICENSE.md). You are free to use, modify, and distribute this project, provided you keep the license and the [NOTICE](https://github.com/laplacef/digest-generator/blob/main/NOTICE.md) file with any redistribution.
