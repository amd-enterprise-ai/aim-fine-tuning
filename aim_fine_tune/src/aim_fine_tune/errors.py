# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations


class AimFineTuneError(ValueError):
    """Top-level CLI/runtime validation and execution error."""


class DownloadError(ValueError):
    """Download and dataset preparation error."""
