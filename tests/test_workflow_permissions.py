"""Tests for the permissions and pins of the GitHub workflows.

Each test reads the workflow files themselves, so the text is what gets checked.
"""

import re
from pathlib import Path
from typing import Any

import yaml

_WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
_COMMIT_SHA = re.compile(r"@[0-9a-f]{40}$")


def _load(name: str) -> dict[str, Any]:
    return yaml.safe_load((_WORKFLOWS / name).read_text())


class TestCiPermissions:
    def test_token_is_read_only(self):
        assert _load("ci.yml")["permissions"] == {"contents": "read"}


class TestReleaseOrder:
    def test_github_release_waits_for_the_pypi_upload(self):
        """A failed upload must not leave a GitHub release pointing at nothing."""
        needs = _load("release.yml")["jobs"]["github-release"]["needs"]
        assert "pypi-publish" in ([needs] if isinstance(needs, str) else needs)


class TestPublishingPins:
    def test_actions_holding_the_oidc_token_are_pinned_to_a_commit(self):
        """A tag or branch can be moved; a job that can publish runs only pinned code."""
        jobs = _load("release.yml")["jobs"]
        publishing = {
            name: job
            for name, job in jobs.items()
            if job.get("permissions", {}).get("id-token") == "write"
        }
        assert publishing
        for name, job in publishing.items():
            third_party = [
                step["uses"]
                for step in job["steps"]
                if "uses" in step and not step["uses"].startswith("actions/")
            ]
            assert third_party, name
            for uses in third_party:
                assert _COMMIT_SHA.search(uses), (name, uses)
