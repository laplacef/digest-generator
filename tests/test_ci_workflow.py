"""Tests for how the CI workflow installs the project.

Each test reads ``ci.yml`` itself, so the workflow text is what gets checked.
"""

from pathlib import Path

import yaml

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _install_commands() -> dict[str, list[str]]:
    """Map each job to the ``uv sync`` commands its steps run."""
    workflow = yaml.safe_load(_WORKFLOW.read_text())
    return {
        name: [str(step["run"]) for step in job["steps"] if "uv sync" in str(step.get("run", ""))]
        for name, job in workflow["jobs"].items()
    }


class TestLockedInstall:
    def test_every_job_installs_the_project(self):
        commands = _install_commands()
        assert commands
        assert all(commands.values()), commands

    def test_install_fails_on_a_stale_lock(self):
        # --locked compares uv.lock with pyproject.toml and exits non-zero on
        # a mismatch. --frozen installs from the lock without comparing.
        for job, commands in _install_commands().items():
            for command in commands:
                assert "--locked" in command, (job, command)
                assert "--frozen" not in command, (job, command)
