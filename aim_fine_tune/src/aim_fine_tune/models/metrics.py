# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TrainMetricsMessage:
    train_loss: float | None
    step: int
    elapsed_seconds: float | None
    tokens_seen: int | None = None
    event: Literal["train_metrics"] = "train_metrics"

    def to_jsonable(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "event": self.event,
            "step": self.step,
        }
        if self.train_loss is not None:
            payload["train_loss"] = self.train_loss
        if self.tokens_seen is not None:
            payload["tokens_seen"] = self.tokens_seen
        if self.elapsed_seconds is not None:
            payload["elapsed_seconds"] = self.elapsed_seconds
        return payload

    def to_report_metrics(self) -> dict[str, float]:
        return {}
