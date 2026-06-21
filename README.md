# Digest Generator

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit-archive)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](./LICENSE.md)

## Overview

Digest Generator is a Python pipeline that aggregates articles from 75 RSS feeds across AI, business, engineering, infrastructure, and security. It generates fact-dense per-article summaries via an LLM, classifies them with zero-shot NLI, and produces a weekly Markdown digest via Ollama.

## How It Works

```mermaid
flowchart LR
    Feeds[("RSS")]:::storage

    subgraph S["run --no-digest"]
        direction TB
        Fetch:::compute --> Summarize:::compute --> Classify:::compute
    end

    JSON[("JSON")]:::storage

    subgraph D["digest"]
        direction TB
        Writer:::compute --> Editor:::compute
        Editor --> Framer:::compute --> Watcher:::compute --> Composer:::compute
        Editor --> Composer
        Framer --> Composer
    end

    MD[("digest.md")]:::storage

    Feeds --> Fetch
    Classify --> JSON
    JSON --> Writer
    Composer --> MD

    classDef compute fill:#fed7aa,stroke:#9a3412,color:#0f172a
    classDef storage fill:#e5e7eb,stroke:#374151,color:#0f172a
```

`digest-generator run` does both halves end to end, writing the JSON corpus and the final Markdown digest into the same run directory. You can also run the halves separately: `run --no-digest` stops after building the corpus, and `digest <run_dir>` turns an existing corpus into a digest.

For full usage details, see [`docs/usage.md`](docs/usage.md).

## Installation

```bash
git clone https://github.com/laplacef/digest-generator.git
cd digest-generator
uv sync --extra dev
cp .env.example .env
# edit .env, set HF_TOKEN
```

Full step-by-step (including prerequisites, optional audio rendering, optional GPU acceleration) in [`docs/setup.md`](docs/setup.md).

## Usage

```bash
digest-generator run                  # full pipeline (fetch + summarize + classify + digest)
digest-generator run --no-digest      # corpus build only (skip digest generation)
digest-generator run --audio          # full pipeline + Piper TTS rendition
digest-generator digest <run_dir>     # regenerate the digest from an existing run directory
digest-generator audio <run_dir>      # render audio for an existing digest (no LLM cost)
digest-generator feeds                # list available feeds
```

Each run lands in its own timestamped directory under `output/`, containing the per-stage caches, the final Markdown digest, run metadata, and a log of the run. See [`docs/usage.md`](docs/usage.md) for the full CLI reference, programmatic API, and output layout.

## Contributing

Bug reports, feature requests, and pull requests are all welcome. See [CONTRIBUTING.md](./CONTRIBUTING.md) for development setup, coding standards, and the contribution workflow.

This project follows a [Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you are expected to uphold it.

## License

This project is licensed under the [Apache License 2.0](./LICENSE.md). You are free to use, modify, and distribute this project, provided you include proper attribution. See the [NOTICE](./NOTICE.md) file for details.
