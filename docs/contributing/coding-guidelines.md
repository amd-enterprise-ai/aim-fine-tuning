<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Coding guidelines

Follow these conventions when contributing to AIM Fine-tuning. Keep changes
focused, preserve existing package boundaries, and include tests for behavioral
changes.

## Python style

- Target Python 3.12 for the top-level project and the `aim_fine_tune` CLI.
- Format Python with Black using a 120-character line length.
- Organize imports with isort's Black profile.
- Write clear, typed interfaces at module boundaries. Run MyPy on changed Python
  code and resolve new type errors before submitting a change.
- Use descriptive names and keep functions focused on a single responsibility.
- Add comments only when they explain a non-obvious decision or constraint.

## Tests

- Add or update focused tests when changing behavior, especially CLI inputs,
  recipe selection, configuration serialization, or training setup.
- Run the affected package's tests locally. For the CLI package:

  ```bash
  cd aim_fine_tune
  uv run pytest
  ```

- Run the full package test suite when a change crosses module or package
  boundaries.

## Formatting and validation

The repository pre-commit configuration checks copyright notices, JSON, YAML,
trailing whitespace, requirements files, Black formatting, Flake8, isort, and
MyPy. Install and run the hooks before opening a review:

```bash
pre-commit install
pre-commit run --all-files
```

Flake8 uses a 120-character line limit and is configured to ignore `E203`,
`E501`, `F401`, and `F841`; do not rely on those exceptions to make code harder
to read.

## Files and documentation

- Include the AMD copyright and MIT SPDX header in new source and documentation
  files, following the surrounding file format.
- Keep public command, configuration, and environment-variable documentation in
  sync with the implementation.
- Use the external documentation TOC in `_toc.yml.in`; do not add inline MyST
  `{toctree}` directives to pages built by this site.
- Do not commit generated documentation output, virtual environments, or local
  build artifacts.

## Change review

Describe the user-visible behavior, the tests run, and any deployment or
compatibility considerations in the pull request. Keep unrelated formatting or
refactoring out of functional changes so reviews remain focused.
