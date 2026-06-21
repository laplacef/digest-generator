# Contributing

Thanks for your interest in contributing. Digest Generator welcomes bug reports, feature requests, and pull requests from everyone.

By participating you agree to abide by our [Code of Conduct](./CODE_OF_CONDUCT.md).

## How contributions work

This project is maintained by a single author. The most reliable way to contribute is to **start with an issue**: report a bug or propose a change, and a maintainer will help align on scope before any code is written. Small, obvious fixes (typos, documentation, a one-line bug fix) are welcome directly as pull requests. For anything larger, open an issue and wait for a maintainer to confirm the approach before investing time in a PR.

## Ways to contribute

- Report a bug or request a feature through Issues (see below).
- Improve documentation, examples, or test coverage.
- Submit a pull request for an open issue.

## Reporting issues

- Search [existing issues](https://github.com/laplacef/digest-generator/issues) first so you don't file a duplicate.
- Open a new issue using the relevant template. For bugs, include a minimal reproduction, expected vs. actual behavior, and your environment (OS, Python version).
- For security vulnerabilities, **do not open a public issue**. Follow [SECURITY.md](./SECURITY.md) instead.

## Development setup

Outside contributors work from a fork. The project uses [`uv`](https://docs.astral.sh/uv/) and Python 3.14.

1. **Fork** this repository, then clone your fork:
   ```bash
   git clone https://github.com/<your-username>/digest-generator.git
   cd digest-generator
   ```
2. **Install dependencies and dev tooling**:
   ```bash
   uv sync --extra dev
   ```
3. **Set up environment**:
   ```bash
   cp .env.example .env
   # add your HF_TOKEN to .env
   ```
4. **Install the pre-commit hooks**:
   ```bash
   uv run pre-commit install
   ```
5. **Verify the baseline**:
   ```bash
   uv run pytest
   ```

## Making changes

This project uses a fork-and-pull-request workflow. For non-trivial changes, confirm an issue exists and a maintainer has agreed on the approach before you open a PR.

1. Create a branch on your fork: `<type>/<short-description>` (e.g. `feat/add-reddit-source`, `fix/fetcher-timeout`).
2. Make small, atomic commits, one logical change each, in a working state at every step.
3. Run the checks locally before pushing:
   ```bash
   uv run pre-commit run --all-files
   uv run pytest
   ```
4. Push to your fork and open a pull request against this repository's `main`.
5. Complete the pull request template and link any related issue.

## Commit messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`. Subject in imperative mood, under 50 characters, no trailing period. Explain **why** in the body, not just what.

## Coding standards

- Code is linted, formatted, and type-checked with ruff, mypy, and bandit, enforced by pre-commit and CI.
- Add or update tests for new behavior and bug fixes.
- Keep `main` releasable: every commit should leave the test suite green.

CI runs the same checks on every pull request. Don't bypass them with `--no-verify`.

## Review process

- A maintainer will review your pull request. Address each comment or explain why a suggestion doesn't apply.
- Reviews check correctness, scope, and alignment with existing conventions.
- Be patient and respectful.

## License

By contributing, you agree that your contributions are licensed under the [Apache License 2.0](./LICENSE.md).
