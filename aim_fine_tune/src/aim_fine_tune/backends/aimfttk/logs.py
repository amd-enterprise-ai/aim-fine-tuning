# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

from aim_fine_tune.backends.base import (
    EventSubscription,
    StdoutLine,
)
from aim_fine_tune.models.methods.sft import SftTrainMetricsMessage


class SftOutputObserver:
    """Convert AIMFTTK's structured sft stdout lines into standard runtime events."""

    def subscriptions(self) -> tuple[EventSubscription[StdoutLine], ...]:
        return (EventSubscription(StdoutLine, self.extract_train_metrics),)

    def extract_train_metrics(
        self,
        event: StdoutLine,
    ) -> SftTrainMetricsMessage | None:
        # AIMFTTK emits one JSON object per training-loss log:
        # {"event":"train_metrics","train_loss":...,"step":...,"elapsed_seconds":...,"tokens_seen":...}
        try:
            payload = json.loads(event.text)
            if payload["event"] != "train_metrics":
                return None
            return SftTrainMetricsMessage(
                train_loss=payload.get("train_loss"),
                step=payload["step"],
                elapsed_seconds=payload.get("elapsed_seconds"),
                tokens_seen=payload.get("tokens_seen"),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return None
