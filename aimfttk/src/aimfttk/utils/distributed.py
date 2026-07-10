# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Utilities for distributed work"""

import os


def is_fsdp():
    return os.environ.get("ACCELERATE_USE_FSDP", "no").lower() == "true"
