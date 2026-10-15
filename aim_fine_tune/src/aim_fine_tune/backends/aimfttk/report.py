# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

from aim_fine_tune.backends.base import (
    StdoutLine,
)
from aim_fine_tune.execution import RUNTIME_DIR_NAME, TRAINING_STDOUT_LOG_NAME
from aim_fine_tune.models.methods.sft import SftTrainingReport, SftTrainMetricsMessage
from aim_fine_tune.models.metrics import TrainMetricsMessage
from aim_fine_tune.models.reports import TrainingReport
from aim_fine_tune.models.training import TrainRequest

REPORT_TYPES: dict[str, tuple[type[TrainingReport], type[TrainMetricsMessage]]] = {
    "SFT": (SftTrainingReport, SftTrainMetricsMessage),
}


def extract_train_metrics(
    event: StdoutLine,
    message_type: type[TrainMetricsMessage],
) -> TrainMetricsMessage | None:
    try:
        payload = json.loads(event.text)
        if payload["event"] != "train_metrics":
            return None
        if message_type is SftTrainMetricsMessage:
            return SftTrainMetricsMessage(
                train_loss=payload.get("train_loss"),
                step=payload["step"],
                elapsed_seconds=payload.get("elapsed_seconds"),
                tokens_seen=payload.get("tokens_seen"),
            )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    return None


def parse_aimfttk_run_report(
    *,
    request: TrainRequest,
    duration_seconds: float,
    accelerators: tuple[str, ...] = (),
) -> TrainingReport:
    # Intermediate progress is the JSON Lines stream on stdout. Final figures come from
    # the last numbered checkpoint's trainer_state.json when checkpointing is enabled.
    # With save_interval=-1 there are no numbered checkpoints, so train loss and throughput
    # fall back to train_metrics events in the captured stdout log. checkpoint-final is a
    # save_model() dump and does not include trainer_state.json. Eval events are not written
    # to that stream, so validation losses stay incomplete on the fallback path.
    resolved_accelerators = list(accelerators)
    gpu_count = request.metadata.accelerator_count or len(resolved_accelerators) or 1
    output_path = Path(request.data.output_path)
    method = request.metadata.method.upper()
    if method not in REPORT_TYPES:
        supported_methods = ", ".join(sorted(REPORT_TYPES))
        raise ValueError(
            f"No AIMFTTK training report is registered for method '{method}'. "
            f"Supported report methods: {supported_methods}."
        )
    report_type, metric_type = REPORT_TYPES[method]
    history = _log_history(output_path, metric_type)

    train_losses: list[float] = []
    eval_losses: list[float] = []
    validation_seconds = 0.0
    elapsed_seconds: float | None = None
    validation_before_step = 0.0
    tokens_per_second: float | None = None
    tokens_seen: float | None = None
    for record in history:
        if not isinstance(record, dict):
            continue
        if record.get("eval_loss") is not None:
            eval_losses.append(float(record["eval_loss"]))
        if record.get("eval_runtime") is not None:
            validation_seconds += float(record["eval_runtime"])
        if record.get("train_runtime") is not None:
            elapsed_seconds = float(record["train_runtime"])
            validation_before_step = validation_seconds
        if record.get("train_tokens_per_second") is not None:
            tokens_per_second = float(record["train_tokens_per_second"])
        if record.get("num_input_tokens_seen") is not None:
            tokens_seen = float(record["num_input_tokens_seen"])
        if record.get("loss") is None:
            continue
        train_losses.append(float(record["loss"]))

    throughput = None
    if elapsed_seconds is not None and gpu_count > 0:
        training_seconds = elapsed_seconds - validation_before_step
        session_tokens = tokens_per_second * elapsed_seconds if tokens_per_second is not None else tokens_seen
        if training_seconds > 0 and session_tokens is not None:
            throughput = session_tokens / training_seconds / gpu_count

    return report_type(
        backend="aimfttk",
        accelerators=resolved_accelerators,
        initial_train_loss=train_losses[0] if train_losses else None,
        final_train_loss=train_losses[-1] if train_losses else None,
        initial_validation_loss=eval_losses[0] if eval_losses else None,
        final_validation_loss=eval_losses[-1] if eval_losses else None,
        duration_seconds=duration_seconds,
        average_tokens_per_gpu_per_second=throughput,
    ).with_incomplete(expect_validation=request.data.validation_data_path is not None)


def _log_history(output_path: Path, metric_type: type[TrainMetricsMessage]) -> list[object]:
    trainer_states = {
        int(step): checkpoint_dir / "trainer_state.json"
        for checkpoint_dir in (output_path / "checkpoints").glob("checkpoint-*")
        if (step := checkpoint_dir.name.removeprefix("checkpoint-")).isdigit()
        and (checkpoint_dir / "trainer_state.json").is_file()
    }
    if trainer_states:
        return json.loads(trainer_states[max(trainer_states)].read_text(encoding="utf-8"))["log_history"]

    stdout_log_path = output_path / RUNTIME_DIR_NAME / TRAINING_STDOUT_LOG_NAME
    history: list[object] = []
    if stdout_log_path.is_file():
        for line in stdout_log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            message = extract_train_metrics(StdoutLine(line), metric_type)
            if message is not None:
                history.append(
                    {
                        "loss": message.train_loss,
                        "step": message.step,
                        "train_runtime": message.elapsed_seconds,
                        "num_input_tokens_seen": message.tokens_seen,
                        **message.to_report_metrics(),
                    }
                )
    if not history:
        raise FileNotFoundError(
            f"No numbered checkpoint with trainer_state.json under {output_path / 'checkpoints'} "
            f"and no train_metrics events in {stdout_log_path}"
        )
    return history
