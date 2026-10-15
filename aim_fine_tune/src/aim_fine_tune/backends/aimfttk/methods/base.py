# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any, Protocol

from aim_fine_tune.models.training import TrainRuntimeParameters


class AimfttkTrainingMethod(Protocol):
    name: str
    command_name: str

    def extend_training_args(
        self,
        training_args: dict[str, Any],
        defaults: TrainRuntimeParameters,
    ) -> None:
        pass

    def config_sections(self, defaults: TrainRuntimeParameters) -> dict[str, Any]:
        pass
