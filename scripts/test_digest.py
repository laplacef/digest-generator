from pathlib import Path

from digest_generator.api import digest
from digest_generator.core.digest.io import build_digest_filename, build_digest_markdown
from digest_generator.shared.logging import logger

run_dir = Path("output/test-output-1")

digest_result = digest(run_dir=run_dir)

if not digest_result or not digest_result.content:
    logger.error("Digest generation returned empty — check LLM logs above")
    raise SystemExit(1)

digest_filename = build_digest_filename(digest_result)
digest_path = run_dir / digest_filename
markdown = build_digest_markdown(digest_result)
digest_path.write_text(markdown, encoding="utf-8")

logger.info(
    "Digest written to {} ({} words, {} chars)",
    digest_path,
    digest_result.word_count,
    len(digest_result.content),
)
