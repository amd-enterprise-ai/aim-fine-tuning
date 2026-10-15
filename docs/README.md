<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Fine-tuning documentation

The published docs live under [`docs/`](docs/) and are built with
[Sphinx](https://www.sphinx-doc.org) using Markdown (via
[MyST](https://myst-parser.readthedocs.io)) and
[rocm-docs-core](https://github.com/ROCm/rocm-docs-core) — AMD's official docs
theme (pydata-sphinx-theme + sphinx-book-theme) — so local previews match the
published umbrella site at <https://enterprise-ai.docs.amd.com>. Diagrams are
authored in [D2](https://d2lang.com) and committed as rendered SVGs.

## Layout

| Path | Purpose |
| --- | --- |
| `docs/docs/` | Sphinx source root (Markdown pages, `conf.py`, `_static/`). |
| `docs/docs/README.md` | Site landing page (the `_toc.yml` root). |
| `docs/docs/_toc.yml.in` | Table of contents template; rocm-docs-core generates `_toc.yml` from it at build time. |
| `docs/docs/<section>/<section>.md` | Per-section index pages (a bullet list of the section's links, for GitHub browsing). |
| `docs/docs/assets/diagrams/*.svg` | Committed, rendered diagrams (do not edit by hand). |
| `docs/diagrams/*.d2` | D2 diagram sources. |
| `docs/diagrams/_theme.d2` | Central style shared by every diagram. |
| `docs/requirements.txt` | Python build dependencies. |

## Generate docs locally

Run the following commands in virtual environment with required dependencies installed (see `docs/requirements.txt`):

```bash
 pip install -r docs/requirements.txt
```

Then build the docs:

```bash
 cd docs
 sphinx-build -b html aim-fine-tuning _build/html

```bash
 cd ..
 sphinx-autobuild docs/aim-fine-tuning _build/html
```

## Diagrams (D2)

1. Edit or add a source file under `docs/diagrams/`. Every diagram starts with
   `...@_theme` to inherit the shared palette, classes, layout engine, and theme
   from `docs/diagrams/_theme.d2`. Restyle all graphs by editing that one file.
2. Render the committed SVGs, then commit both the
   `.d2` source and the `.svg` output.
3. Reference a diagram from a page with a normal relative image so it renders in
   both Sphinx and plain GitHub browsing:

   ```markdown
   ![Alt text describing the diagram](../assets/diagrams/<name>.svg)
   ```

> GitHub renders Mermaid natively but not D2. The committed-SVG workflow is what
> keeps diagrams visible when browsing `.md` files directly on GitHub.

### Render diagrams

Install d2 (see Installation [`https://d2lang.com/tour/install/`](https://d2lang.com/tour/install/).) and run the following command from the `docs` directory:

```bash
 d2 docs/diagrams/*.d2 docs/docs/assets/diagrams/
```

## Multi-stack / umbrella site

These docs are designed to build standalone *and* to be aggregated into the
umbrella `AMDEnterpriseAISuiteDocs` site, where the contents of `docs/docs` are
synced under `docs/aim-fine-tuning/` (see
[`.github/workflows/sync-docs-to-central-repo.yml`](../.github/workflows/sync-docs-to-central-repo.yml))
and built by that repo's own root `conf.py`.

To keep both modes working:

- Use **relative** `.md` links between pages — never absolute paths and never the
  `aim-fine-tuning/` prefix (the umbrella adds that when nesting this stack).
- Express navigation in `docs/docs/_toc.yml.in` (sphinx-external-toc), and give
  each section landing page a plain bullet list of the same links so GitHub
  browsing stays navigable.
- Keep page-local anchors unique per document; nesting under `aim-fine-tuning/` in the
  umbrella namespaces everything else.
