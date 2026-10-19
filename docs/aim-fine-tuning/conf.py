# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
"""Sphinx configuration for the AIM Fine-tuning documentation.

Built with rocm-docs-core (AMD's official docs theme: pydata-sphinx-theme +
sphinx-book-theme + MyST + sphinx-design + sphinx-external-toc), so the local
preview matches the published umbrella site at
https://enterprise-ai.docs.amd.com.

This project also builds standalone and is aggregated into the umbrella
AMDEnterpriseAISuiteDocs site. ``.github/workflows/sync-docs-to-central-repo.yml``
rsyncs ``docs/aim-fine-tuning/`` into that repository under
``docs/aim-fine-tuning/``, excluding this ``conf.py``. The umbrella build uses
its own root config. To keep both modes working: all internal links are
relative ``.md`` links, and the table of contents lives in ``_toc.yml.in``.
"""

import os
from datetime import datetime

# -- Project information ------------------------------------------------------
# Parameterized via env vars so the umbrella build can override per stack.
project = os.environ.get("DOCS_PROJECT", "AIM Fine-Tuning")
author = "Advanced Micro Devices, Inc."
copyright = f"{datetime.now():%Y}, {author}"
version = os.environ.get("DOCS_VERSION", "")
release = version

# -- rocm-docs-core ----------------------------------------------------------
extensions = ["rocm_docs"]
html_theme = "rocm_docs_theme"
# "generic" gives the AMD-branded theme without the ROCm-product header/banner
# (and avoids the ROCm version banner network fetch). Other flavors: rocm,
# instinct, local, ...
html_theme_options = {
    "flavor": "generic",
    # The generic-flavor header shows "GitHub" and "Support" links; without a
    # repository_url they fall back to "#". Point them at the repo and its
    # new-issue page.
    "repository_url": "https://github.com/amdenterpriseai/aim-fine-tuning",
}

# Table of contents (sphinx-external-toc, configured by rocm-docs-core). The
# template is committed; rocm-docs-core renders the ignored output before
# sphinx-external-toc consumes it.
external_toc_path = "_toc.yml"
external_toc_template_path = "_toc.yml.in"

# Don't fetch the shared ROCm projects.yaml over the network at build time; this
# project doesn't use cross-project intersphinx.
external_projects_remote_repository = ""
external_projects = []

# -- MyST --------------------------------------------------------------------
# rocm-docs-core already enables colon_fence, substitution, etc.; this list is
# unioned with those. Deep heading anchors so the generated CRD API reference's
# in-page anchor links resolve.
# A set, because rocm-docs-core unions it with the extensions it already enables
# (colon_fence, substitution, html_image, ...).
myst_enable_extensions = {
    "linkify",
    "deflist",
    "tasklist",
    "attrs_inline",
    "attrs_block",
}
myst_heading_anchors = 6

# The generated CRD API reference (reference/api/*.md) links between types with
# same-page anchors that MyST renders as working links but still flags as
# false-positive xref_missing warnings. Suppress that category so ``-W`` builds
# stay meaningful for real problems.
suppress_warnings = ["myst.xref_missing"]

# -- HTML output -------------------------------------------------------------
html_title = project

# The legacy inline navigation page is replaced by the external TOC.
exclude_patterns = ["toctree.md"]


def setup(app):
    import logging

    # Filter rocm-docs-core's benign "current project not found in projects"
    # warning so it doesn't fail strict (-W) builds. This project isn't a member
    # of the shared ROCm projects.yaml and doesn't use cross-project intersphinx.
    class _DropProjectsWarning(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "not found in projects" not in record.getMessage()

    logging.getLogger("sphinx.rocm_docs.projects").addFilter(_DropProjectsWarning())
