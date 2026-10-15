# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from ..metrics import TrainMetricsMessage
from ..reports import TrainingReport
from .base import parameter_field
from .common import CommonParameters


class SftParameters(CommonParameters):
    method_name = "SFT"

    max_context_length: int | None = parameter_field(
        None,
        env="AIMFT_MAX_CONTEXT_LENGTH",
        alias="maxContextLength",
        ge=1,
    )


class SftTrainingReport(TrainingReport):
    pass


class SftTrainMetricsMessage(TrainMetricsMessage):
    pass
