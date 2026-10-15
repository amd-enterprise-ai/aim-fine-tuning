# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import codecs
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from importlib.metadata import version
from pathlib import Path
from typing import Any, BinaryIO, TypeAlias

from aim_runtime.accelerator_detector import AcceleratorDetector

from aim_fine_tune.backends.base import (
    BackendAdapter,
    EventActor,
    EventSubscription,
    LaunchSpec,
    LogMessage,
    RawStderrChunk,
    RawStdoutChunk,
    StderrClosed,
    StderrLine,
    StdoutClosed,
    StdoutLine,
)
from aim_fine_tune.errors import AimFineTuneError
from aim_fine_tune.models.build import EngineImage
from aim_fine_tune.models.metrics import TrainMetricsMessage
from aim_fine_tune.models.recipes import RecipeData
from aim_fine_tune.models.reports import TrainingReport, TrainReportMessage
from aim_fine_tune.models.training import TrainRequest

logger = logging.getLogger(__name__)

BUILD_INFO_PATH = Path("/opt/aim_fine_tune/build-info.json")
RUNTIME_DIR_NAME = "runtime"
TRAINING_STDOUT_LOG_NAME = "train.stdout.log"
TRAINING_STDERR_LOG_NAME = "train.stderr.log"
TRAINING_REPORT_NAME = "train-report.json"
TRAINING_STATUS_NAME = "train-status.json"
STATUS_HEARTBEAT_SECONDS = 10.0


class JsonStdoutPrinter:
    def subscriptions(self) -> tuple[EventSubscription[Any], ...]:
        return (
            EventSubscription(TrainMetricsMessage, self.print_message),
            EventSubscription(LogMessage, self.print_message),
            EventSubscription(TrainReportMessage, self.print_message),
        )

    def print_message(self, message: TrainMetricsMessage | LogMessage | TrainReportMessage) -> None:
        _write_json_line(message.to_jsonable())


class TrainReportPrinter:
    """Print the training report as a JSON line when engine output is relayed as-is."""

    def subscriptions(self) -> tuple[EventSubscription[Any], ...]:
        return (EventSubscription(TrainReportMessage, self.print_message),)

    def print_message(self, message: TrainReportMessage) -> None:
        _write_json_line(message.to_jsonable())


class RawStdoutPrinter:
    def subscriptions(self) -> tuple[EventSubscription[Any], ...]:
        return (EventSubscription(RawStdoutChunk, self.print_chunk),)

    def print_chunk(self, chunk: RawStdoutChunk) -> None:
        _relay_bytes(chunk.data)


class RawStderrPrinter:
    def subscriptions(self) -> tuple[EventSubscription[Any], ...]:
        return (EventSubscription(RawStderrChunk, self.print_chunk),)

    def print_chunk(self, chunk: RawStderrChunk) -> None:
        _relay_bytes(chunk.data)


class PlainLogPrinter:
    def subscriptions(self) -> tuple[EventSubscription[Any], ...]:
        return (EventSubscription(LogMessage, self.print_message),)

    def print_message(self, message: LogMessage) -> None:
        sys.stdout.write(message.message + "\n")
        sys.stdout.flush()


RawStreamChunk: TypeAlias = RawStdoutChunk | RawStderrChunk
StreamClosed: TypeAlias = StdoutClosed | StderrClosed


class TrainLogWriter:
    """Append one captured engine stream as chunks are published."""

    def __init__(
        self,
        path: Path,
        chunk_type: type[RawStdoutChunk] | type[RawStderrChunk],
        closed_type: type[StdoutClosed] | type[StderrClosed],
    ) -> None:
        self._path = path
        self._chunk_type = chunk_type
        self._closed_type = closed_type
        self._file: BinaryIO | None = None

    def __enter__(self) -> TrainLogWriter:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("wb")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def subscriptions(self) -> tuple[EventSubscription[Any], ...]:
        return (
            EventSubscription[Any](self._chunk_type, self.write_chunk),
            EventSubscription[Any](self._closed_type, self.finish),
        )

    def write_chunk(self, chunk: RawStreamChunk) -> None:
        if self._file is None:
            raise RuntimeError("The training log writer is not open.")
        self._file.write(chunk.data)
        self._file.flush()

    def finish(self, _event: StreamClosed) -> None:
        self.close()

    def close(self) -> None:
        if self._file is None:
            return
        self._file.flush()
        self._file.close()
        self._file = None


class _LineObserver:
    def __init__(
        self,
        chunk_type: type[RawStdoutChunk] | type[RawStderrChunk],
        closed_type: type[StdoutClosed] | type[StderrClosed],
        line_type: type[StdoutLine] | type[StderrLine],
    ) -> None:
        self._chunk_type = chunk_type
        self._closed_type = closed_type
        self._line_type = line_type
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._buffer = ""

    def subscriptions(self) -> tuple[EventSubscription[Any], ...]:
        return (
            EventSubscription[Any](self._chunk_type, self.feed),
            EventSubscription[Any](self._closed_type, self.finish),
        )

    def feed(self, chunk: RawStreamChunk) -> tuple[object, ...]:
        return self._lines(self._decoder.decode(chunk.data).replace("\r", "\n"))

    def finish(self, _event: StreamClosed) -> tuple[object, ...]:
        events = list(self._lines(self._decoder.decode(b"", final=True).replace("\r", "\n")))
        if self._buffer:
            events.append(self._line_type(self._buffer))
            self._buffer = ""
        return tuple(events)

    def _lines(self, text: str) -> tuple[object, ...]:
        self._buffer += text
        lines: list[object] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            lines.append(self._line_type(line))
        return tuple(lines)


class OutputPipeline:
    """Synchronously route source events and actor-produced events."""

    def __init__(self, actors: tuple[EventActor, ...]) -> None:
        self._actors = [
            _LineObserver(RawStdoutChunk, StdoutClosed, StdoutLine),
            _LineObserver(RawStderrChunk, StderrClosed, StderrLine),
            *actors,
        ]
        self._events: deque[object] = deque()
        self._lock = threading.RLock()
        self._dispatching = False
        self._started = False
        self._closed = False

    def __enter__(self) -> OutputPipeline:
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def start(self) -> None:
        if self._started:
            return
        self._started = True

    def add_actors(self, actors: tuple[EventActor, ...]) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("The output pipeline is closed.")
            self._actors.extend(actors)

    def publish(self, event: object) -> None:
        with self._lock:
            if not self._started or self._closed:
                raise RuntimeError("The output pipeline is not running.")
            self._events.append(event)
            if self._dispatching:
                return
            self._dispatching = True
            try:
                while self._events:
                    self._dispatch(self._events.popleft())
            finally:
                self._dispatching = False

    def flush(self) -> None:
        # Acquiring the lock waits for a source currently dispatching events.
        with self._lock:
            if self._events:
                raise RuntimeError("The output pipeline has undispatched events.")

    def close(self) -> None:
        with self._lock:
            if not self._started or self._closed:
                return
            self._closed = True

    def _dispatch(self, event: object) -> None:
        for actor in tuple(self._actors):
            for subscription in actor.subscriptions():
                if not isinstance(event, subscription.event_type):
                    continue
                try:
                    published = subscription.callback(event)
                    if published is None:
                        continue
                    self._events.extend(published if isinstance(published, tuple) else (published,))
                except Exception:
                    logger.exception("Output actor %s failed.", type(actor).__name__)


class TrainingStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class TrainingStatusHeartbeat:
    def __init__(self, path: Path, *, interval_seconds: float = STATUS_HEARTBEAT_SECONDS):
        self.path = path
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> TrainingStatusHeartbeat:
        self.write(TrainingStatus.RUNNING)
        self._thread = threading.Thread(target=self._heartbeat, name="training-status-heartbeat", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        self.write(TrainingStatus.FAILED if exc_type is not None else TrainingStatus.COMPLETED)

    def write(self, status: TrainingStatus) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": status.value,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        temporary_path = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary_path, self.path)

    def _heartbeat(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                self.write(TrainingStatus.RUNNING)
            except OSError:
                logger.exception("Failed to update training status heartbeat at %s.", self.path)


@dataclass(frozen=True)
class CapturedRun:
    returncode: int
    duration_seconds: float


def run_captured_command(
    command: list[str],
    env: dict[str, str],
    *,
    output_pipeline: OutputPipeline,
) -> CapturedRun:
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = process.stdout
    stderr = process.stderr
    if stdout is None or stderr is None:
        raise RuntimeError("Failed to capture stdout and stderr from the subprocess.")

    source_errors: list[Exception] = []
    errors_lock = threading.Lock()

    def publish_stream(
        stream: BinaryIO,
        chunk_type: type[RawStdoutChunk] | type[RawStderrChunk],
        closed_type: type[StdoutClosed] | type[StderrClosed],
    ) -> None:
        try:
            while chunk := stream.read(4096):
                # NOTE: output_pipeline.publish is a blocking call. If an engine produces a ton of output,
                # this may cause a lot of backpressure.
                # Consider making output_pipeline.publish non-blocking if this becomes a problem.
                output_pipeline.publish(chunk_type(chunk))
        except Exception as exc:
            with errors_lock:
                source_errors.append(exc)
        finally:
            stream.close()
            output_pipeline.publish(closed_type())

    sources = (
        threading.Thread(
            target=publish_stream,
            args=(stdout, RawStdoutChunk, StdoutClosed),
            name="training-stdout-source",
        ),
        threading.Thread(
            target=publish_stream,
            args=(stderr, RawStderrChunk, StderrClosed),
            name="training-stderr-source",
        ),
    )
    for source in sources:
        source.start()
    returncode = process.wait()
    for source in sources:
        source.join()
    output_pipeline.flush()
    if source_errors:
        raise source_errors[0]

    return CapturedRun(
        returncode=returncode,
        duration_seconds=time.perf_counter() - started,
    )


def _write_json_line(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n")
    sys.stdout.flush()


def _relay_bytes(chunk: bytes) -> None:
    # Prefer the binary buffer so engine bytes are copied through unchanged.
    # stdout.buffer is missing when stdout is a text-only wrapper (StringIO,
    # redirect_stdout, some log captures); those accept str, not bytes.
    buffer = getattr(sys.stdout, "buffer", None)
    if buffer is not None:
        buffer.write(chunk)
        buffer.flush()
        return
    sys.stdout.write(chunk.decode("utf-8", errors="replace"))
    sys.stdout.flush()


def detected_accelerator_ids() -> tuple[str, ...]:
    try:
        detection = AcceleratorDetector().detect()
    except Exception:
        logger.exception("Failed to detect accelerators for the training report.")
        return ()
    return tuple(RecipeData.normalize_accelerator_id(gpu.device_id) for gpu in detection.gpu_info)


def write_training_artifacts(runtime_dir: Path, *, report: TrainingReport) -> None:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / TRAINING_REPORT_NAME).write_text(
        report.model_dump_json(by_alias=True, indent=2) + "\n",
        encoding="utf-8",
    )


def build_and_write_training_report(
    *,
    adapter: BackendAdapter,
    request: TrainRequest,
    launch_spec: LaunchSpec,
    render_dir: Path,
    captured: CapturedRun,
    accelerators: tuple[str, ...],
) -> TrainingReport:
    try:
        report = adapter.parse_run_report(
            request=request,
            duration_seconds=captured.duration_seconds,
            accelerators=accelerators,
        )
    except Exception:
        logger.exception("Failed to parse %s training report.", launch_spec.backend)
        report = TrainingReport(
            backend=launch_spec.backend,
            accelerators=list(accelerators),
            duration_seconds=captured.duration_seconds,
        ).with_incomplete(expect_validation=request.data.validation_data_path is not None)

    validation_data_path = request.data.validation_data_path
    report = report.model_copy(
        update={
            "ran_to_completion": captured.returncode == 0,
            "finished_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "engine_image": read_engine_image(),
            "aim_fine_tune_version": version("aim_fine_tune"),
            "recipe_id": request.metadata.recipe_id,
            "recipe_digest": request.metadata.recipe_digest,
            "train_data_digest": file_digest(request.data.train_data_path),
            "validation_data_digest": file_digest(validation_data_path) if validation_data_path else None,
            "request": request.to_jsonable(),
        }
    )
    write_training_artifacts(render_dir / RUNTIME_DIR_NAME, report=report)
    return report


def read_engine_image() -> EngineImage | None:
    if not BUILD_INFO_PATH.is_file():
        return None
    build_info = json.loads(BUILD_INFO_PATH.read_text(encoding="utf-8"))
    return EngineImage.parse(build_info["engineImage"])


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def raise_if_failed(captured: CapturedRun) -> None:
    if captured.returncode != 0:
        raise AimFineTuneError(f"Backend execution failed with exit code {captured.returncode}.")
