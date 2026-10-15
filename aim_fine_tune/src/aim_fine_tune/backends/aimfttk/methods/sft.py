# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any

from aim_fine_tune.models.training import TrainRuntimeParameters


class AimfttkSftMethod:
    name = "SFT"
    command_name = "sft"

    def extend_training_args(
        self,
        training_args: dict[str, Any],
        defaults: TrainRuntimeParameters,
    ) -> None:
        return None

    def config_sections(self, defaults: TrainRuntimeParameters) -> dict[str, Any]:
        return {"sft_args": {"max_seq_length": defaults.max_context_length or 2048, "length_handling": "truncate"}}
