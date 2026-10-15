# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, ConfigDict

from aim_fine_tune.models.training import TrainRuntimeParameters

DEFAULT_LEARNING_RATE_WARMUP_STEPS = 100


class CommonTrainingArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    max_grad_norm: float
    weight_decay: float
    optim: str
    num_train_epochs: int
    output_dir: str
    lr_scheduler_type: str
    logging_strategy: str
    logging_steps: int
    logging_first_step: bool
    save_strategy: str
    save_steps: int | None
    seed: int
    bf16: bool
    report_to: list[str]
    push_to_hub: bool
    gradient_checkpointing: bool
    include_num_input_tokens_seen: str
    warmup_steps: int

    @classmethod
    def from_runtime_parameters(
        cls,
        defaults: TrainRuntimeParameters,
        *,
        is_adapter: bool,
        checkpoints_dir: PurePosixPath,
    ) -> "CommonTrainingArgs":
        return cls(
            per_device_train_batch_size=defaults.per_accelerator_batch_size or 1,
            per_device_eval_batch_size=defaults.per_accelerator_batch_size or 1,
            gradient_accumulation_steps=defaults.gradient_accumulation or 1,
            learning_rate=defaults.learning_rate if defaults.learning_rate is not None else 0.00005,
            max_grad_norm=7.0,
            weight_decay=0.000001 if is_adapter else 0.00001,
            optim="adamw_torch",
            num_train_epochs=defaults.epochs,
            output_dir=str(checkpoints_dir),
            lr_scheduler_type="cosine",
            logging_strategy="steps",
            logging_steps=defaults.logging_interval,
            logging_first_step=True,
            save_strategy="steps" if defaults.save_interval != -1 else "no",
            save_steps=defaults.save_interval if defaults.save_interval != -1 else None,
            seed=defaults.seed,
            bf16=True,
            report_to=["none"],
            push_to_hub=False,
            gradient_checkpointing=False,
            include_num_input_tokens_seen="non_padding",
            warmup_steps=(
                defaults.learning_rate_warmup_steps
                if defaults.learning_rate_warmup_steps is not None
                else DEFAULT_LEARNING_RATE_WARMUP_STEPS
            ),
        )

    def render(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)
