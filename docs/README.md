<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Fine-tuning documentation

The published docs live under [`docs/aim-fine-tuning/`](aim-fine-tuning/) and are built with [Sphinx](https://www.sphinx-doc.org) using Markdown (via [MyST](https://myst-parser.readthedocs.io)) and [rocm-docs-core](https://github.com/ROCm/rocm-docs-core) — AMD's official docs theme (pydata-sphinx-theme + sphinx-book-theme) — so local previews match the published umbrella site at <https://enterprise-ai.docs.amd.com>.

## Layout

| Path | Purpose |
| --- | --- |
| `docs/aim-fine-tuning/` | Sphinx source root (Markdown pages, `conf.py`, `_static/`). |
| `docs/aim-fine-tuning/README.md` | Site landing page (the `_toc.yml` root). |
| `docs/aim-fine-tuning/_toc.yml.in` | Table of contents template. rocm-docs-core generates `_toc.yml` from it at build time. |
| `docs/aim-fine-tuning/<section>/<section>.md` | Per-section index pages (a bullet list of the section's links, for GitHub browsing). |
| `docs/requirements.txt` | Python build dependencies. |

## Generate docs locally

Install the build dependencies, then build from the repository root:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/aim-fine-tuning docs/_build/html
```

For a local preview that rebuilds on change:

```bash
sphinx-autobuild docs/aim-fine-tuning docs/_build/html
```

## Multi-stack / umbrella site

These docs are designed to build standalone and to be aggregated into the umbrella `AMDEnterpriseAISuiteDocs` site. [`.github/workflows/sync-docs-to-central-repo.yml`](../.github/workflows/sync-docs-to-central-repo.yml) syncs `docs/aim-fine-tuning/` into that repository under `docs/aim-fine-tuning/`. The umbrella build uses its own root `conf.py` and excludes this tree's `conf.py`.

To keep both modes working:

- Use relative `.md` links between pages. Avoid absolute paths and the `aim-fine-tuning/` prefix, because the umbrella adds that prefix when it nests this stack.
- Express navigation in `docs/aim-fine-tuning/_toc.yml.in` (sphinx-external-toc), and give each section landing page a plain bullet list of the same links so GitHub browsing stays navigable.
- Keep page-local anchors unique per document. Nesting under `aim-fine-tuning/` in the umbrella namespaces everything else.
