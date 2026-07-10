# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Utilities for serialization like YAML dumping"""

import enum

import yaml


class ConfigDumper(yaml.SafeDumper):
    """A SafeDumper that can handle Enum values"""

    def represent_data(self, data):
        if isinstance(data, enum.Enum):
            return self.represent_data(data.value)
        return super().represent_data(data)
