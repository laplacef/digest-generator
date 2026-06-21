"""Run-lifecycle infrastructure: meta.json + run-output directories.

Sibling to ``shared/transformers/`` (in-process HF), ``shared/llm/``
(Ollama HTTP), and ``shared/tts/`` (Piper subprocess). Where those
cluster engine-specific model-serving infrastructure, ``runtime/``
clusters things that exist *around* a pipeline run rather than during
model execution: the run directory itself and the meta.json record
describing it.

- ``meta``: ``RunMeta`` dataclass + ``write_run_meta`` /
  ``update_run_meta_digest`` for the meta.json lifecycle.
- ``dirs``: ``create_run_dir``, a timestamped run-output directory with
  a hex collision suffix.

The CLI is the only writer of meta.json (the library-first design has
``api.run`` return data and lets the caller decide what to persist), but
the dataclass and helpers are reusable for any programmatic caller.
"""
