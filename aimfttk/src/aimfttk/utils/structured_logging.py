# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Stable, machine-readable progress events for AIM Fine-tuning runtimes."""

import json
import math
import sys
from typing import Any

from transformers import TrainerCallback

TRAIN_METRICS_EVENT = "train_metrics"


class StructuredTrainingLogCallback(TrainerCallback):
    """Write one JSON object for each Hugging Face training-loss log."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not isinstance(logs, dict):
            return

        event = _training_event(logs, step=state.global_step)
        if event is None:
            return
        # The leading newline starts a fresh line even when a progress bar has left the cursor
        # mid-line, since the bar redraws on stderr with a carriage return and no newline.
        # One write keeps the whole event atomic on the pipe that every rank shares under accelerate;
        # `print` would send the trailing newline separately and let another rank interleave.
        sys.stdout.write("\n" + json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
        sys.stdout.flush()


def _training_event(logs: dict[str, Any], *, step: int) -> dict[str, int | float | str] | None:
    loss = _number(logs.get("loss"))
    elapsed_seconds = _number(logs.get("train_runtime"))
    tokens_seen = logs.get("num_input_tokens_seen")
    dpo_metrics = {
        key: value
        for key in ("rewards/chosen", "rewards/rejected", "rewards/accuracies", "rewards/margins")
        if (value := _number(logs.get(key))) is not None
    }
    if loss is None and not dpo_metrics:
        return None
    if tokens_seen is not None and (not isinstance(tokens_seen, int) or isinstance(tokens_seen, bool)):
        return None
    event: dict[str, int | float | str] = {
        "event": TRAIN_METRICS_EVENT,
        "step": step,
    }
    if loss is not None:
        event["train_loss"] = loss
    if elapsed_seconds is not None:
        event["elapsed_seconds"] = elapsed_seconds
    if tokens_seen is not None:
        event["tokens_seen"] = tokens_seen
    event.update({key.replace("/", "_"): value for key, value in dpo_metrics.items()})
    return event


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value):
        return None
    return value
