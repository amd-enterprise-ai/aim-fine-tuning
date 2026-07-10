#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
import json
from textwrap import indent

from pydantic import Field

from aimfttk.config import DPOExperimentConfig, SFTExperimentConfig
from aimfttk.config.base import BaseConfig


# This is used so that we can discriminate between SFT and DPO experiments.
# In requests, the aimfttk config is anyway embedded under a specific key, not "flat"
# like it's passed on to the aimfttk program itself.
class MasterConfig(BaseConfig):
    aimfttk_config: SFTExperimentConfig | DPOExperimentConfig = Field(discriminator="method")


schema = MasterConfig.model_json_schema()

# Make total_train_batch_size optional for schema.json (will be set to num GPUS)
del schema["$defs"]["BatchsizeConfig"]["properties"]["total_train_batch_size"]["type"]
schema["$defs"]["BatchsizeConfig"]["properties"]["total_train_batch_size"]["anyOf"] = [
    {"type": "integer"},
    {"type": "null"},
]

print("NOTE: only $defs, add these to the schema.json in the appropriate place")
print("---")
for line in indent(json.dumps({"$defs": schema["$defs"]}, indent=2), 2 * " ").splitlines()[1:-2]:
    print(line)
