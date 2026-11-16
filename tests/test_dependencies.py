"""Tests that the declared dependencies match what the package imports.

The checks read ``pyproject.toml`` and the package source, so a library used
without being declared, or declared without being used, fails here.
"""

import ast
import re
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE = _ROOT / "digest_generator"


def _normalize(name: str) -> str:
    """Return the PEP 503 form of a distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared() -> set[str]:
    """Return the distribution names in ``[project] dependencies``."""
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"]
    names = set()
    for requirement in project["dependencies"]:
        match = re.match(r"[A-Za-z0-9._-]+", requirement)
        assert match, requirement
        names.add(_normalize(match.group()))
    return names


def _imported() -> set[str]:
    """Return the third-party top-level modules the package imports."""
    modules = set()
    for path in _PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module.split(".")[0])
    return modules - sys.stdlib_module_names - {_PACKAGE.name}


def _providers() -> dict[str, set[str]]:
    """Map each importable module to the distributions that provide it."""
    return {
        module: {_normalize(name) for name in names}
        for module, names in packages_distributions().items()
    }


class TestDeclaredDependencies:
    def test_finds_the_package_imports(self):
        # Guards the two checks below against passing on an empty scan.
        assert {"ollama", "typer", "yaml"} <= _imported()

    def test_every_imported_library_is_declared(self):
        declared = _declared()
        providers = _providers()
        undeclared = sorted(
            module for module in _imported() if not providers.get(module, set()) & declared
        )
        assert undeclared == []

    def test_every_declared_library_is_imported(self):
        providers = _providers()
        used = set().union(*(providers.get(module, set()) for module in _imported()))
        assert sorted(_declared() - used) == []
