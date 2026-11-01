"""Tests for the tag handling in the release workflow's shell steps.

Each test lifts a step's ``run:`` script out of ``release.yml`` and runs it
under bash in a temp directory, so the workflow text itself is what gets
checked.
"""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"

_CHANGELOG = """\
# Changelog

## [0.4.0] - 2026-11-15

Final notes.

## [0.4.0-rc.1] - 2026-11-01

Candidate notes.

## [0.3.0] - 2026-10-19

Older notes.
"""


def _step_script(job: str, step: str) -> str:
    workflow = yaml.safe_load(_WORKFLOW.read_text())
    for entry in workflow["jobs"][job]["steps"]:
        if entry.get("name") == step:
            return str(entry["run"])
    msg = f"no step {step!r} in job {job!r}"
    raise KeyError(msg)


def _run(script: str, cwd: Path, **env: str) -> subprocess.CompletedProcess[str]:
    # GitHub runs a step's script as `bash -e {0}`
    return subprocess.run(
        ["bash", "-e", "-c", script],
        cwd=cwd,
        env={"PATH": os.environ["PATH"], **env},
        capture_output=True,
        text=True,
        check=False,
    )


class TestVersionCheck:
    @staticmethod
    def _check(tmp_path: Path, tag: str, version: str) -> subprocess.CompletedProcess[str]:
        (tmp_path / "pyproject.toml").write_text(
            f'[project]\nname = "digest-generator"\nversion = "{version}"\n'
        )
        script = _step_script("build", "Verify tag matches package version")
        return _run(script, tmp_path, GITHUB_REF_NAME=tag)

    @pytest.mark.parametrize(
        ("tag", "version"),
        [
            ("v0.4.0", "0.4.0"),
            ("v0.4.0-rc.1", "0.4.0rc1"),
            ("v0.4.0-rc.12", "0.4.0rc12"),
            ("v0.4.0-beta.2", "0.4.0b2"),
            ("v0.4.0-alpha.3", "0.4.0a3"),
        ],
    )
    def test_tag_matching_version_passes(self, tmp_path: Path, tag: str, version: str) -> None:
        result = self._check(tmp_path, tag, version)
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize(
        ("tag", "version"),
        [
            ("v0.4.0-rc.1", "0.4.0rc2"),
            ("v0.4.0-rc.1", "0.4.0"),
            ("v0.4.0", "0.4.0rc1"),
            ("v0.4.0-beta.2", "0.4.0rc2"),
            # SemVer puts a dot before the pre-release number
            ("v0.4.0-rc1", "0.4.0rc1"),
        ],
    )
    def test_mismatched_pair_fails(self, tmp_path: Path, tag: str, version: str) -> None:
        result = self._check(tmp_path, tag, version)
        assert result.returncode != 0
        assert "does not match" in result.stderr


class TestChangelogExtraction:
    @pytest.mark.parametrize(
        ("tag", "expected", "excluded"),
        [
            ("v0.4.0-rc.1", "Candidate notes.", ["Final notes.", "Older notes."]),
            ("v0.4.0", "Final notes.", ["Candidate notes.", "Older notes."]),
        ],
    )
    def test_extracts_only_the_tagged_entry(
        self, tmp_path: Path, tag: str, expected: str, excluded: list[str]
    ) -> None:
        (tmp_path / "CHANGELOG.md").write_text(_CHANGELOG)
        script = _step_script("github-release", "Extract changelog entry")

        result = _run(script, tmp_path, TAG=tag)

        assert result.returncode == 0, result.stderr
        notes = (tmp_path / "release_notes.md").read_text()
        assert expected in notes
        for text in excluded:
            assert text not in notes


class TestGitHubRelease:
    @staticmethod
    def _release_args(tmp_path: Path, tag: str) -> list[str]:
        # A stub gh on PATH records the arguments the step passes
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        args_file = tmp_path / "gh-args"
        stub = bin_dir / "gh"
        stub.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{args_file}"\n')
        stub.chmod(0o755)
        (tmp_path / "dist").mkdir()
        (tmp_path / "dist" / "digest_generator-0.4.0.tar.gz").write_text("")
        (tmp_path / "release_notes.md").write_text("Notes.\n")
        script = _step_script("github-release", "Create GitHub Release")

        result = _run(script, tmp_path, TAG=tag, PATH=f"{bin_dir}:{os.environ['PATH']}")

        assert result.returncode == 0, result.stderr
        return args_file.read_text().splitlines()

    @pytest.mark.parametrize("tag", ["v0.4.0-rc.1", "v0.4.0-beta.2"])
    def test_pre_release_tag_marks_a_pre_release(self, tmp_path: Path, tag: str) -> None:
        args = self._release_args(tmp_path, tag)
        assert args[:3] == ["release", "create", tag]
        assert "--prerelease" in args

    def test_final_tag_is_a_full_release(self, tmp_path: Path) -> None:
        args = self._release_args(tmp_path, "v0.4.0")
        assert args[:3] == ["release", "create", "v0.4.0"]
        assert "--prerelease" not in args
        assert "dist/digest_generator-0.4.0.tar.gz" in args
