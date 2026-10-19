# Setup Guide

One-time setup from a clone: required steps first, then the optional audio and GPU sections. For a tool install (`uv tool install digest-generator`), skip steps 1 and 2, drop the `uv run` prefix, and keep your `.env` in the directory you run `digest-generator` from. Day-to-day CLI usage is in [`usage.md`](./usage.md).

## Prerequisites

- **Python.** Version 3.13 or later.
- **uv.** The package manager ([install](https://docs.astral.sh/uv/getting-started/installation/)).
- **Ollama.** The summarizer and every digest stage call [Ollama](https://ollama.com), and the default models are Ollama cloud models. Step 4 covers the three ways to connect.
- **HuggingFace token.** `run` and `label` log in to the HuggingFace Hub before the BART-MNLI classifier loads, so they fail without `HF_TOKEN` ([create a token](https://huggingface.co/settings/tokens) with a free account). Audio rendering does not need a token; Piper voices live in a public HF repo.

## Installation

### 1. Clone

```bash
git clone https://github.com/laplacef/digest-generator.git
cd digest-generator
```

### 2. Install dependencies

```bash
uv sync --extra dev
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set `HF_TOKEN`. Keep any comment on its own line: a comment after an empty value (`KEY=   # note`) is read as the value itself. Every other setting has a sensible default, so override only what you need. `.env.example` lists each setting with its default.

### 4. Connect to Ollama

The default models are `gemma4:31b-cloud` (summarizer and writer) and `gpt-oss:120b-cloud` (editor, framer, watcher, and clusterer). Pick one of three setups:

- **Cloud, direct.** Set `OLLAMA_API_KEY` in `.env`. Every stage then calls ollama.com, and no local Ollama is needed.
- **Local Ollama with cloud models.** Install Ollama, run `ollama signin`, and leave `OLLAMA_API_KEY` unset. The local server at `OLLAMA_HOST` (default `http://localhost:11434`) runs the cloud models through your account.
- **Fully local.** Pull local models with `ollama pull <model>` and set `SUMMARIZER_MODEL`, `WRITER_MODEL`, `EDITORIAL_MODEL`, `FRAMER_MODEL`, `WATCHER_MODEL`, and `CLUSTERER_MODEL` in `.env`. Raise `OLLAMA_CONCURRENCY` (default `3`) if your GPU serves parallel requests.

### 5. Configure feeds

The tool ships with no feeds. Write a starter and edit it:

```bash
uv run digest-generator init   # writes ~/.config/digest-generator/feeds.yaml
```

The file has two blocks. `categories:` defines your digest sections (each a stable `id` and a display `title`), in the order they appear in the digest. `feeds:` lists each feed's `name`, `url`, and `category` (which must match one of your category ids). [`usage.md`](./usage.md#feed-configuration) covers where the loader looks for the file and how to point at another one.

The bundled prompts are generic and topic-neutral, so no prompt setup is required. To tune the voice for your topic, drop override templates into a `prompts/` directory; see [`usage.md`](./usage.md#prompt-overrides) for the search order and template names.

### 6. Verify the core install

```bash
uv run digest-generator --help
uv run digest-generator feeds | head      # lists the feeds from your feeds.yaml
uv run pytest -q                          # full test suite (all I/O mocked, no network)
```

If `--help` lists the subcommands and the test suite passes, the core install is done. With a `feeds.yaml` in place you can smoke-test against real feeds with `uv run digest-generator run -c ai --limit 3 --no-digest` (cap entries, skip digest). That run calls both Ollama (summarizer) and the HuggingFace Hub (classifier), so it confirms steps 3 and 4.

---

## Optional: Audio rendering

Required only when using `--audio` on `digest-generator run` / `digest-generator digest`, or the standalone `digest-generator audio <run_dir>` command. Skip this section if you don't need TTS rendition.

### Install Piper TTS

```bash
uv tool install piper-tts==1.4.2
```

Piper is a standalone CLI that the renderer calls as a subprocess, so it installs separately rather than as a Python dependency.

### Install ffmpeg

Install ffmpeg with your system package manager:

```bash
sudo apt install ffmpeg   # Debian/Ubuntu
brew install ffmpeg       # macOS
```

With [mise](https://mise.jdx.dev), `mise use -g ffmpeg@latest` also works.

### Verify the audio toolchain

```bash
which piper ffmpeg
piper --help | head -3
ffmpeg -version | head -1
```

### Verify against an existing digest

The audio renderer needs only the digest `.md` at the run directory root, so stage caches are irrelevant. Any existing run directory works:

```bash
uv run digest-generator audio output/<some-run-dir>
```

What happens on first run:

- Downloads ~63 MB of Piper voice files to `~/.cache/digest_generator/piper-voices/` (pinned to the `rhasspy/piper-voices` `v1.0.0` tag via `AUDIO_VOICE_REVISION`).
- Runs the narration pre-pass, then Piper synthesis, then ffmpeg encoding.
- Writes `output/<run-dir>/audio/<date>.opus` + `audio/cache_key.txt`.
- Echoes the `.opus` path on stdout.

Later runs with the same digest markdown, voice, bitrate, and sentence-silence setting are sub-second cache hits.

Listen to the result with any audio player (`mpv`, `ffplay`, VLC). In a clone, you can fix mispronunciations by adding entries to `digest_generator/core/audio/narration_overrides.yaml` and re-rendering with `rm -rf <run>/audio && uv run digest-generator audio <run>` (the cache key does not cover the overrides file); an installed package has no user-level overrides file. The optional audio settings (voice, bitrate, sentence silence, and the rest) are in the Audio block of [`.env.example`](../.env.example).

---

## Optional: GPU acceleration

Required only when running the BART-MNLI topic classifier on GPU. CPU is the default and sufficient for a typical feed corpus.

### Prerequisites

- An NVIDIA driver that supports CUDA 13 (Linux/Windows). In a clone, `uv sync` installs the CUDA 13.0 (`cu130`) torch wheels on Linux and Windows; these bundle the CUDA runtime. A tool or pip install gets PyPI's default torch build instead.
- Or Apple Silicon (Mac) for MPS

### Configure

```bash
# In .env
DEVICE=cuda          # or "mps" on Apple Silicon
```

Override per invocation with `digest-generator run --device cuda` or `digest-generator label <run_dir> --device cuda`. The summarizer and digest stages don't run on GPU (they use Ollama), so `DEVICE` affects only the classifier.

---

## Troubleshooting

### Core pipeline

**`Illegal header value b'Bearer '` or a HuggingFace 401 at startup.** `run` and `label` log in to the HuggingFace Hub before classifying. The first message means `HF_TOKEN` is unset; a 401 means the token is invalid. Set a valid token in `.env`.

**Ollama connection refused.** With `OLLAMA_API_KEY` unset, every LLM call goes to `OLLAMA_HOST` (default `http://localhost:11434`). Start Ollama, or point `OLLAMA_HOST` at the server you use.

**Ollama 401 or 403.** A 401 means the cloud credentials are missing or invalid: check `OLLAMA_API_KEY`, or run `ollama signin` for a local server. A 403 means your account cannot use that model; choose one it can access and set the matching `*_MODEL` variable. A 429 is a rate limit, which the client retries with backoff; lowering `OLLAMA_CONCURRENCY` makes it rarer.

**`No feeds.yaml found`.** The loader found no file in its search path (the error lists each path it checked). Run `digest-generator init`, or pass `--feeds <file>` or `--config <dir>`.

**`Unknown content type '...'`.** The `-c` flag takes the `id` values under `categories:` in your `feeds.yaml`, and the error lists the valid ones. The starter file defines `ai`, `engineering`, and `security`.

### Audio

**`piper` binary not found on `PATH`.** The `uv tool install piper-tts` step succeeded, but its directory isn't on `$PATH`. Run `uv tool update-shell` and reopen your shell, or find the directory with `uv tool dir --bin` and add it to `$PATH`.

**`ffmpeg` binary not found on `PATH`.** `which ffmpeg` returns nothing. Re-run the install. On macOS, Homebrew sometimes installs to `/opt/homebrew/bin`, which may not be on `$PATH` for non-interactive shells, so check `echo $PATH`.

**HuggingFace download fails on first audio render.** The voice file lives in the public `rhasspy/piper-voices` repo and needs no auth. If `digest-generator audio` errors during download, check that `huggingface.co` is reachable. The download is one-time; later runs use the local cache.

**Audio cache hit after the narration changed.** The cache key covers the digest markdown, voice, bitrate, and sentence-silence setting, so edits to `narration_overrides.yaml` are not detected. Delete `<run>/audio/` to force a fresh render.
