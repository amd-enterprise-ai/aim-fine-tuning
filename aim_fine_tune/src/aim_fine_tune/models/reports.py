# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from .build import EngineImage


class TrainingReport(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    backend: str
    accelerators: list[str] = Field(default_factory=list)
    initial_train_loss: float | None = Field(default=None, alias="initialTrainLoss")
    final_train_loss: float | None = Field(default=None, alias="finalTrainLoss")
    initial_validation_loss: float | None = Field(default=None, alias="initialValidationLoss")
    final_validation_loss: float | None = Field(default=None, alias="finalValidationLoss")
    duration_seconds: float | None = Field(default=None, alias="durationSeconds")
    average_tokens_per_gpu_per_second: float | None = Field(default=None, alias="averageTokensPerGpuPerSecond")
    incomplete: list[str] = Field(default_factory=list)
    ran_to_completion: bool = Field(default=False, alias="ranToCompletion")
    finished_at: str | None = Field(default=None, alias="finishedAt")
    engine_image: EngineImage | None = Field(default=None, alias="engineImage")
    aim_fine_tune_version: str | None = Field(default=None, alias="aimFineTuneVersion")
    recipe_id: str | None = Field(default=None, alias="recipeId")
    recipe_digest: str | None = Field(default=None, alias="recipeDigest")
    train_data_digest: str | None = Field(default=None, alias="trainDataDigest")
    validation_data_digest: str | None = Field(default=None, alias="validationDataDigest")
    request: dict[str, Any] | None = None
    report_metric_names: ClassVar[tuple[str, ...]] = ()

    def to_jsonable(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)

    def with_incomplete(self, *, expect_validation: bool) -> TrainingReport:
        incomplete: list[str] = []
        if not self.accelerators:
            incomplete.append("accelerators")
        if self.initial_train_loss is None:
            incomplete.append("initialTrainLoss")
        if self.final_train_loss is None:
            incomplete.append("finalTrainLoss")
        if expect_validation and self.initial_validation_loss is None:
            incomplete.append("initialValidationLoss")
        if expect_validation and self.final_validation_loss is None:
            incomplete.append("finalValidationLoss")
        if self.duration_seconds is None:
            incomplete.append("durationSeconds")
        if self.average_tokens_per_gpu_per_second is None:
            incomplete.append("averageTokensPerGpuPerSecond")
        return self.model_copy(update={"incomplete": incomplete})


@dataclass(frozen=True)
class TrainReportMessage:
    report: TrainingReport
    event: Literal["train_report"] = "train_report"

    def to_jsonable(self) -> dict[str, object]:
        # model_dump_json serializes NaN and infinite losses as null, which strict JSON output requires.
        return {"event": self.event, "report": json.loads(self.report.model_dump_json(by_alias=True))}
