# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Self

from .base import BaseParameters, parameter_field


class CommonParameters(BaseParameters):
    epochs: int = parameter_field(1, env="AIMFT_EPOCHS", ge=1)
    seed: int = parameter_field(42, env="AIMFT_SEED")
    evaluation_interval: int = parameter_field(
        1000,
        env="AIMFT_EVALUATION_INTERVAL",
        alias="evaluationInterval",
        ge=1,
    )
    logging_interval: int = parameter_field(25, env="AIMFT_LOGGING_INTERVAL", alias="loggingInterval", ge=1)
    learning_rate: float | None = parameter_field(None, env="AIMFT_LEARNING_RATE", alias="learningRate")
    learning_rate_warmup_steps: int | None = parameter_field(
        None,
        env="AIMFT_LEARNING_RATE_WARMUP_STEPS",
        alias="learningRateWarmupSteps",
        ge=0,
    )
    per_accelerator_batch_size: int | None = parameter_field(
        None,
        env="AIMFT_PER_ACCELERATOR_BATCH_SIZE",
        alias="perAcceleratorBatchSize",
        ge=1,
    )
    gradient_accumulation: int | None = parameter_field(
        None,
        env="AIMFT_GRADIENT_ACCUMULATION",
        alias="gradientAccumulation",
        ge=1,
    )
    lora_alpha: int | None = parameter_field(None, env="AIMFT_LORA_ALPHA", alias="loraAlpha", ge=1)
    lora_dropout: float | None = parameter_field(
        None,
        env="AIMFT_LORA_DROPOUT",
        alias="loraDropout",
        ge=0,
        le=1,
    )
    lora_rank: int | None = parameter_field(None, env="AIMFT_LORA_RANK", alias="loraRank", ge=1)
    save_interval: int = parameter_field(500, env="AIMFT_SAVE_INTERVAL", alias="saveInterval")

    def apply_adapter_defaults(self, use_adapter: bool) -> Self:
        if not use_adapter:
            return self

        defaults: dict[str, float | int] = {}
        if self.lora_dropout is None:
            defaults["lora_dropout"] = 0.0
        if self.lora_rank is None:
            defaults["lora_rank"] = 64
        return self.model_copy(update=defaults) if defaults else self
