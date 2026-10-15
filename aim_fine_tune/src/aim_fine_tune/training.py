# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

from aim_fine_tune.backends import get_backend_adapter
from aim_fine_tune.backends.base import (
    BackendAdapter,
    EventActor,
    LaunchSpec,
    LogMessage,
    RawStderrChunk,
    RawStdoutChunk,
    StderrClosed,
    StdoutClosed,
)
from aim_fine_tune.errors import AimFineTuneError
from aim_fine_tune.execution import (
    RUNTIME_DIR_NAME,
    TRAINING_REPORT_NAME,
    TRAINING_STATUS_NAME,
    TRAINING_STDERR_LOG_NAME,
    TRAINING_STDOUT_LOG_NAME,
    JsonStdoutPrinter,
    OutputPipeline,
    PlainLogPrinter,
    RawStderrPrinter,
    RawStdoutPrinter,
    TrainingStatusHeartbeat,
    TrainLogWriter,
    TrainReportPrinter,
    build_and_write_training_report,
    detected_accelerator_ids,
    raise_if_failed,
    run_captured_command,
)
from aim_fine_tune.models.reports import TrainReportMessage
from aim_fine_tune.models.training import TrainRequest


def run_training(
    request_factory: Callable[[], TrainRequest],
    *,
    dry_run: bool,
    json_logs: bool,
    render_dir: Path | None,
) -> None:
    printers: tuple[EventActor, ...] = (
        (JsonStdoutPrinter(),)
        if json_logs
        else (RawStdoutPrinter(), RawStderrPrinter(), PlainLogPrinter(), TrainReportPrinter())
    )
    with OutputPipeline(printers) as output_pipeline:
        try:
            request = request_factory()
            adapter = get_backend_adapter(request.metadata.backend)
            output_pipeline.add_actors(adapter.create_output_observers(request))
            launch_spec = adapter.prepare(request)
            resolved_render_dir = resolve_render_dir(request.data.output_path, render_dir)
            write_render_workspace(resolved_render_dir, request, launch_spec)
            output_pipeline.publish(
                LogMessage(message=f"Rendered {request.metadata.backend} launch workspace to {resolved_render_dir}")
            )

            if not dry_run:
                ensure_output_path(request.data.output_path)
                execute_launch_spec(
                    launch_spec,
                    resolved_render_dir,
                    request=request,
                    adapter=adapter,
                    output_pipeline=output_pipeline,
                )
                output_pipeline.publish(LogMessage(message="Execution completed successfully."))
        except (AimFineTuneError, ValueError) as exc:
            output_pipeline.publish(LogMessage(message=str(exc), level="error"))
            raise


def resolve_render_dir(output_path: Path, render_dir: Path | None) -> Path:
    if render_dir is not None:
        return render_dir.expanduser()
    return output_path


def write_render_workspace(render_dir: Path, request: TrainRequest, launch_spec: LaunchSpec) -> None:
    render_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, content in launch_spec.files.items():
        target_path = render_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")

    runtime_dir = render_dir / RUNTIME_DIR_NAME
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "train-request.json").write_text(
        json.dumps(request.to_jsonable(), indent=2) + "\n",
        encoding="utf-8",
    )
    (runtime_dir / "launch-spec.json").write_text(
        json.dumps(launch_spec.to_jsonable(), indent=2) + "\n",
        encoding="utf-8",
    )
    (runtime_dir / "run.sh").write_text(render_shell_script(launch_spec), encoding="utf-8")


def build_execution_command(launch_spec: LaunchSpec) -> list[str]:
    return list(launch_spec.command)


def execute_launch_spec(
    launch_spec: LaunchSpec,
    render_dir: Path,
    *,
    request: TrainRequest,
    adapter: BackendAdapter,
    output_pipeline: OutputPipeline,
) -> None:
    status_path = render_dir / RUNTIME_DIR_NAME / TRAINING_STATUS_NAME
    with TrainingStatusHeartbeat(status_path):
        stage_runtime_configs(launch_spec, render_dir)
        validate_execution_paths(launch_spec)
        execution_env = build_execution_environment(launch_spec)
        command = build_execution_command(launch_spec)
        accelerators = detected_accelerator_ids()
        stdout_log_path = render_dir / RUNTIME_DIR_NAME / TRAINING_STDOUT_LOG_NAME
        stderr_log_path = render_dir / RUNTIME_DIR_NAME / TRAINING_STDERR_LOG_NAME
        with (
            TrainLogWriter(stdout_log_path, RawStdoutChunk, StdoutClosed) as stdout_log,
            TrainLogWriter(stderr_log_path, RawStderrChunk, StderrClosed) as stderr_log,
        ):
            output_pipeline.add_actors((stdout_log, stderr_log))
            captured = run_captured_command(command, execution_env, output_pipeline=output_pipeline)
        report = build_and_write_training_report(
            adapter=adapter,
            request=request,
            launch_spec=launch_spec,
            render_dir=render_dir,
            captured=captured,
            accelerators=accelerators,
        )
        output_pipeline.publish(TrainReportMessage(report))
        output_pipeline.publish(
            LogMessage(message=f"Wrote training report to {render_dir / RUNTIME_DIR_NAME / TRAINING_REPORT_NAME}")
        )
        if report.incomplete:
            formatted = ", ".join(report.incomplete)
            output_pipeline.publish(
                LogMessage(message=f"Training report incomplete fields: {formatted}", level="warning")
            )
        raise_if_failed(captured)


def build_execution_environment(
    launch_spec: LaunchSpec,
    *,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    execution_env = dict(os.environ if base_env is None else base_env)
    execution_env.update(launch_spec.env)
    return execution_env


def validate_execution_paths(launch_spec: LaunchSpec) -> None:
    runtime_configs_dir = Path(str(launch_spec.config_mount_destination))
    for relative_path in launch_spec.files:
        if not relative_path.startswith("configs/"):
            continue
        staged_path = runtime_configs_dir / Path(relative_path).name
        if not staged_path.exists():
            raise AimFineTuneError(f"Staged config file is missing: '{staged_path}'.")

    command = build_execution_command(launch_spec)
    if len(command) >= 2:
        command_target = Path(command[1])
        if command_target.is_absolute() and not command_target.exists():
            raise AimFineTuneError(f"Backend command path does not exist: '{command_target}'.")


def stage_runtime_configs(launch_spec: LaunchSpec, render_dir: Path) -> None:
    runtime_configs_dir = Path(str(launch_spec.config_mount_destination))
    runtime_configs_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in launch_spec.files:
        if not relative_path.startswith("configs/"):
            continue
        target_path = runtime_configs_dir / Path(relative_path).name
        source_path = render_dir / relative_path
        if source_path.resolve() == target_path.resolve():
            continue
        target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")


def ensure_output_path(output_path: Path) -> None:
    if output_path.exists() and not output_path.is_dir():
        raise AimFineTuneError(f"AIMFT_OUTPUT_PATH must point to a directory, got '{output_path}'.")
    output_path.mkdir(parents=True, exist_ok=True)


def render_shell_script(launch_spec: LaunchSpec) -> str:
    export_lines = "\n".join(f"export {key}={_quote_shell(value)}" for key, value in sorted(launch_spec.env.items()))
    command = " ".join(_quote_shell(argument) for argument in launch_spec.command)
    return f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- \"$(dirname -- \"${{BASH_SOURCE[0]}}\")\" && pwd)
mkdir -p {launch_spec.config_mount_destination}
cp \"$SCRIPT_DIR/configs/\"* {launch_spec.config_mount_destination}/
{export_lines}
exec {command}
"""


def _quote_shell(argument: str) -> str:
    return "'" + argument.replace("'", "'\"'\"'") + "'"
