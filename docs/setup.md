# Setup Guide

One-time setup for Digest Generator. Required prerequisites come first, then optional sections (audio rendering, GPU). Install only what you need.

For day-to-day CLI usage, see [`usage.md`](./usage.md).

## Prerequisites

- **Python 3.13** or higher
- **uv** package manager ([install](https://docs.astral.sh/uv/getting-started/installation/))
- **HuggingFace account** ([sign up free](https://huggingface.co/join)) for the BART-MNLI classifier. Audio rendering does not need a token; Piper voices live in a public HF repo.

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

Edit `.env` and set `HF_TOKEN`. Every other setting has a sensible default, so override only what you need. The full list of values lives in `digest_generator/shared/settings.py`, and audio-specific settings are covered in the optional section below.

### 4. Configure feeds

The tool ships with no feeds. Write a starter and edit it:

```bash
uv run digest-generator init   # writes ~/.config/digest-generator/feeds.yaml
```

The file has two blocks. `categories:` defines your digest sections (each a stable `id` and a display `title`), in the order they appear in the digest. `feeds:` lists each feed's `name`, `url`, and `category` (which must match one of your category ids). The loader searches, in order: a path passed via `--feeds`, then `<DIGEST_CONFIG>/feeds.yaml`, then `./digest-generator/feeds.yaml`, then `~/.config/digest-generator/feeds.yaml`. Point at any file directly with `digest-generator run --feeds path/to/feeds.yaml`.

The bundled prompts are generic and topic-neutral, so no prompt setup is required. To tune the voice for your topic, drop override templates into a `prompts/` directory; see [`usage.md`](./usage.md#prompt-overrides) for the search order and template names.

### 5. Verify the core install

```bash
uv run digest-generator --help
uv run digest-generator feeds | head      # lists the feeds from your feeds.yaml
uv run pytest -q                  # full test suite (all I/O mocked, no network)
```

If `digest-generator --help` shows the subcommands (`init`, `run`, `fetch`, `summarize`, `label`, `digest`, `audio`, `feeds`) and the test suite passes, the core install is done. With a `feeds.yaml` in place you can smoke-test against real feeds with `digest-generator run -c ai --limit 3 --no-digest` (cap entries, skip digest).

---

## Optional: Audio rendering

Required only when using `--audio` on `digest-generator run` / `digest-generator digest`, or the standalone `digest-generator audio <run_dir>` command. Skip this section if you don't need TTS rendition.

### Install Piper TTS

```bash
uv tool install piper-tts==1.4.2
```

Piper is a standalone CLI that the renderer calls as a subprocess, so it installs separately rather than as a Python dependency.

### Install ffmpeg

```bash
mise use -g ffmpeg@latest
```

Installs a static ffmpeg binary and puts it on your `PATH`.

If `mise` isn't available, install ffmpeg with your system package manager: `sudo apt install ffmpeg` (Debian/Ubuntu) or `brew install ffmpeg` (macOS).

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
- Writes `output/<run-dir>/audio/<date>-<slug>.opus` + `audio/cache_key.txt`.
- Echoes the `.opus` path on stdout.

Subsequent runs against the same `(md, voice, bitrate)` triple are sub-second cache hits.

Listen to the result with any audio player (`mpv`, `ffplay`, VLC). If you notice mispronunciations, add entries to `digest_generator/core/audio/narration_overrides.yaml` and re-render with `rm -rf <run>/audio && digest-generator audio <run>` (the cache key doesn't see overrides-file edits).

### Audio-related env vars (`.env`)

All optional with defaults from `digest_generator/shared/settings.py`:

```bash
# AUDIO_VOICE_MODEL=en_US-amy-medium     # Piper voice id
# AUDIO_VOICE_REVISION=375a0fe641dea077c2a47b4e9a056d6da521eed3
# AUDIO_BITRATE_KBPS=24                  # Opus encode bitrate
# AUDIO_SAMPLE_RATE=22050                # Piper native sample rate
# AUDIO_VOICE_CACHE=~/.cache/digest_generator/piper-voices
# AUDIO_FFMPEG_PATH=ffmpeg               # Override ffmpeg binary path
```

---

## Optional: GPU acceleration

Required only when running the BART-MNLI topic classifier on GPU. CPU is the default and sufficient for a typical feed corpus.

### Prerequisites

- CUDA 13.0+ runtime (Linux/Windows). `torch` installs with CUDA 13.0 wheels automatically.
- Or Apple Silicon (Mac) for MPS

### Configure

```bash
# In .env
DEVICE=cuda          # or "mps" on Apple Silicon
```

Override per invocation with `digest-generator run --device cuda` or `digest-generator label <run_dir> --device cuda`. The summarizer and digest stages don't run on GPU (they use Ollama), so `DEVICE` affects only the classifier.

---

## Troubleshooting

**`piper` binary not found on `PATH`.** The `uv tool install piper-tts` step succeeded, but its directory isn't on `$PATH`. Run `uv tool update-shell` and reopen your shell, or find the directory with `uv tool dir --bin` and add it to `$PATH`.

**`ffmpeg` binary not found on `PATH`.** `which ffmpeg` returns nothing. Re-run the install. On macOS, Homebrew sometimes installs to `/opt/homebrew/bin`, which may not be on `$PATH` for non-interactive shells, so check `echo $PATH`.

**HuggingFace download fails on first audio render.** The voice file lives in the public `rhasspy/piper-voices` repo and needs no auth. If `digest-generator audio` errors during download, check that `huggingface.co` is reachable. The download is one-time; later runs use the local cache.

**Audio cache hit after the narration changed.** The cache key hashes only `(md_bytes, voice_id, bitrate_kbps)`, so edits to `narration_overrides.yaml` are not detected. Delete `<run>/audio/` to force a fresh render.
