<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Contributing to AIM Fine-tuning

Thank you for your interest in contributing. AIM Fine-tuning is currently released here as a public reference. We are not looking for direct code contributions, but we welcome issue reports.

## Code of Conduct

By participating in this project you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Issue tracking

Bug reports and feature requests are tracked as [GitHub Issues](https://github.com/amd-enterprise-ai/aim-fine-tuning/issues). Before opening a new issue, search existing ones to avoid duplicates.

## Reporting security issues

Do **not** open a public GitHub Issue for security vulnerabilities. See [SECURITY.md](SECURITY.md) for the responsible disclosure process.

## Developer setup

If you wish to e.g. extend this codebase for your own use, this section has some pointers for getting started with setup.

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

### Linting

We recommend installing the pre-commit hooks in `.pre-commit-config.yaml` to maintain the same formatting as used through the repository.

## License

By contributing you agree that your contributions will be licensed under the [MIT License](LICENSE).
