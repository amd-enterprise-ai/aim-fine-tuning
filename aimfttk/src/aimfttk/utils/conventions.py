# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""This module codifies conventions used in the package"""

import os
import pathlib

# In containers running in workflow we define CHECKPOINTS_PATH env var to keep it the same
# in all containers whether they are handling data movement or do the actual finetuning job.
# In order for this to work properly, aimfttk templates must not define `exp_conf.training_args.output_dir`
# and let it be set by default
local_checkpoints_dir = pathlib.Path(os.getenv("CHECKPOINTS_PATH", "./checkpoints"))
local_logs_dir = pathlib.Path(os.getenv("LOGS_PATH", "./logs"))


def ensure_tensorboard_logging_dir():
    if os.getenv("TENSORBOARD_LOGGING_DIR") is not None:
        return
    os.environ["TENSORBOARD_LOGGING_DIR"] = local_logs_dir.as_posix()
