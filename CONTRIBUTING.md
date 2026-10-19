# Contributing

Thanks for your interest in contributing. Bug reports, feature requests, and pull requests are welcome from everyone, and participation follows the [Code of Conduct](./CODE_OF_CONDUCT.md).

## How contributions work

This project is maintained by a single author, so the most reliable way to contribute is to **start with an issue**: report a bug or propose a change, and a maintainer will help align on scope before any code is written. Small, obvious fixes (typos, documentation, a one-line bug fix) are welcome directly as pull requests. For anything larger, wait for a maintainer to confirm the approach before investing time in a PR.

## Reporting issues

- Search [existing issues](https://github.com/laplacef/digest-generator/issues) first so you don't file a duplicate.
- Open a new issue using the relevant template; the bug template asks for the environment details needed to reproduce.
- For security vulnerabilities, **do not open a public issue**. Follow [SECURITY.md](./SECURITY.md) instead.

## Development setup

Outside contributors work from a fork. The project uses [`uv`](https://docs.astral.sh/uv/) and Python 3.13 or later (`.python-version` pins 3.13, which CI tests).

```bash
git clone https://github.com/<your-username>/digest-generator.git
cd digest-generator
uv sync --extra dev               # dependencies and dev tooling
uv run pre-commit install         # lint, format, type, security, and secret checks on commit
uv run pytest                     # baseline: all I/O is mocked, so no tokens or network needed
```

Real runs need a `.env` (`cp .env.example .env`, then set `HF_TOKEN`); [`docs/setup.md`](./docs/setup.md) covers Ollama and the optional audio and GPU setup.

## Making changes

1. Create a branch on your fork: `<type>/<short-description>` (e.g. `feat/add-reddit-source`, `fix/fetcher-timeout`).
2. Make small, atomic commits, one logical change each, leaving the test suite green at every step.
3. Add or update tests for new behavior and bug fixes.
4. Run the checks locally before pushing: `uv run pre-commit run --all-files` and `uv run pytest`. CI runs the same checks on every pull request, so don't bypass them with `--no-verify`.
5. Push to your fork, open a pull request against `main`, and complete the pull request template.

## Commit messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

Types: `feat`, `fix`, `perf`, `refactor`, `docs`, `style`, `test`, `build`, `ci`, `chore`. Subject in imperative mood, under 50 characters, no trailing period. Use the body to explain **why**; the diff already shows what changed.

## License

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](./LICENSE.md).
