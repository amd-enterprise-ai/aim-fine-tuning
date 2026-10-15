<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Contributing to AIM Fine-tuning

Thank you for your interest in contributing. This document explains how to get started, how work is tracked, and what to expect from the review process.

## Code of Conduct

By participating in this project you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Issue tracking

Bug reports and feature requests are tracked as [GitHub Issues](https://github.com/amd-enterprise-ai/aim-fine-tuning/issues). Before opening a new issue, search existing ones to avoid duplicates.

## Development setup

Install the pre-commit hooks in `.pre-commit-config.yaml`.

### Workspace layout

| Path | Description |
| --- | --- |
| `aimfttk` | A fine-tuning engine |
| `aim_fine_tune` | The standard unified interface |

### Test commands

To test `aimfttk`, install the package locally and use pytest.
```bash
cd aimfttk
pytest tests
```

To test `aim_fine_tune`, [install uv](https://docs.astral.sh/uv/getting-started/installation/) and run:
```bash
cd aim_fine_tune
uv run --locked pytest tests/ -v
```

## Making changes

1. **Fork** the repository and create a branch from `main`.
2. **Make your changes.** Keep commits focused — one logical change per commit.
3. **Add or update tests** for any new behavior.
4. **Run the relevant test suite** before opening a PR.
5. **Open a pull request** against `main` with a clear title and description.

### Commit signing and sign-off

Commits must be both cryptographically **signed** and carry a Developer Certificate of Origin (DCO) **`Signed-off-by`** trailer. Use `git commit -s` to add the trailer automatically. This is enforced by a GitHub policy.

### What reviewers look for

- Tests cover the new behavior
- No secrets, credentials, or internal hostnames in committed files

## Reporting security issues

Do **not** open a public GitHub Issue for security vulnerabilities. See [SECURITY.md](SECURITY.md) for the responsible disclosure process.

## License

By contributing you agree that your contributions will be licensed under the [MIT License](LICENSE).
