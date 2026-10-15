# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import TestCase, main

from aim_fine_tune.backends.aimfttk.logs import SftOutputObserver
from aim_fine_tune.backends.aimfttk.report import parse_aimfttk_run_report
from aim_fine_tune.backends.base import (
    EventActor,
    EventSubscription,
    LogMessage,
    RawStderrChunk,
    RawStdoutChunk,
    StderrClosed,
    StderrLine,
    StdoutClosed,
    StdoutLine,
)
from aim_fine_tune.execution import (
    JsonStdoutPrinter,
    OutputPipeline,
    PlainLogPrinter,
    RawStderrPrinter,
    RawStdoutPrinter,
    TrainingStatusHeartbeat,
    TrainLogWriter,
    run_captured_command,
)
from aim_fine_tune.models import (
    TrainRequest,
    TrainRequestData,
    TrainRequestMetadata,
    TrainRuntimeParameters,
)
from aim_fine_tune.models.methods.sft import SftTrainMetricsMessage
from aim_fine_tune.models.metrics import TrainMetricsMessage

FIXTURE_MODEL_ID = "google/gemma-3-27b-it"


def _request(
    *,
    validation: bool = True,
    accelerator_count: int = 2,
    output_path: Path = Path("/outputs"),
) -> TrainRequest:
    return TrainRequest(
        metadata=TrainRequestMetadata(
            recipe_id=None,
            backend="aimfttk",
            model_id=FIXTURE_MODEL_ID,
            aim_id=FIXTURE_MODEL_ID,
            method="SFT",
            is_adapter=True,
            accelerator_count=accelerator_count,
        ),
        data=TrainRequestData(
            model_path=Path("/model"),
            train_data_path=Path("/train.jsonl"),
            validation_data_path=Path("/valid.jsonl") if validation else None,
            output_path=output_path,
        ),
        defaults=TrainRuntimeParameters(
            epochs=1,
            seed=42,
            evaluation_interval=1000,
            logging_interval=25,
        ),
    )


class AimfttkReportParserTests(TestCase):
    def test_reads_final_figures_from_the_last_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory)
            _write_trainer_state(output_path, step=50, log_history=_STATE_HISTORY)
            report = parse_aimfttk_run_report(
                request=_request(output_path=output_path),
                duration_seconds=60.0,
                accelerators=("0x74a1", "0x74a1"),
            )

        self.assertEqual(report.accelerators, ["0x74a1", "0x74a1"])
        self.assertEqual(report.initial_train_loss, 2.4123456789)
        self.assertEqual(report.final_train_loss, 1.087654321)
        self.assertEqual(report.initial_validation_loss, 2.3712345678)
        self.assertEqual(report.final_validation_loss, 1.2187654321)
        self.assertEqual(report.duration_seconds, 60.0)
        # 32768 tokens over 20s, 4s of it validation before the last training step: 32768 / 16 / 2 GPUs.
        self.assertEqual(report.average_tokens_per_gpu_per_second, 1024.0)
        self.assertEqual(report.incomplete, [])

    def test_throughput_ignores_validation_frequency(self) -> None:
        # Both runs do identical training: 32768 tokens over 16 seconds of training across 2 GPUs.
        def _history(eval_seconds: float) -> list[dict[str, float]]:
            wall = 16.0 + eval_seconds
            history: list[dict[str, float]] = [
                {
                    "loss": 2.4,
                    "epoch": 0.1,
                    "num_input_tokens_seen": 4096,
                    "train_runtime": 5.0,
                    "train_tokens_per_second": 819.2,
                }
            ]
            for _ in range(int(eval_seconds // 8)):
                history.append(
                    {
                        "eval_loss": 1.5,
                        "eval_runtime": 8.0,
                        "num_input_tokens_seen": 4096,
                    }
                )
            history.append(
                {
                    "loss": 1.1,
                    "epoch": 1.0,
                    "num_input_tokens_seen": 32768,
                    "train_runtime": wall,
                    "train_tokens_per_second": 32768 / wall,
                }
            )
            return history

        rates = []
        for eval_seconds in (8.0, 32.0):
            with tempfile.TemporaryDirectory() as directory:
                output_path = Path(directory)
                _write_trainer_state(output_path, step=50, log_history=_history(eval_seconds))
                report = parse_aimfttk_run_report(
                    request=_request(output_path=output_path),
                    duration_seconds=120.0,
                    accelerators=("0x74a1", "0x74a1"),
                )
            rates.append(report.average_tokens_per_gpu_per_second)

        self.assertEqual(rates[0], rates[1])
        self.assertAlmostEqual(rates[0], 1024.0)

    def test_does_not_treat_eval_loss_as_train_loss(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory)
            _write_trainer_state(
                output_path,
                step=10,
                log_history=[
                    {"eval_loss": 3.0, "epoch": 0.5},
                    {"loss": 2.0, "epoch": 1.0},
                ],
            )
            report = parse_aimfttk_run_report(
                request=_request(output_path=output_path),
                duration_seconds=1.0,
                accelerators=("0x74a1",),
            )
        self.assertEqual(report.initial_train_loss, 2.0)
        self.assertEqual(report.final_train_loss, 2.0)
        self.assertEqual(report.initial_validation_loss, 3.0)
        self.assertEqual(report.final_validation_loss, 3.0)

    def test_train_only_omits_validation_from_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory)
            _write_trainer_state(
                output_path,
                step=10,
                log_history=[{"loss": 2.0, "epoch": 0.1}, {"loss": 1.0, "epoch": 1.0}],
            )
            report = parse_aimfttk_run_report(
                request=_request(validation=False, accelerator_count=1, output_path=output_path),
                duration_seconds=5.0,
                accelerators=("0x74a1",),
            )
        self.assertIsNone(report.initial_validation_loss)
        self.assertIsNone(report.final_validation_loss)
        self.assertEqual(report.incomplete, ["averageTokensPerGpuPerSecond"])

    def test_uses_the_highest_numbered_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory)
            _write_trainer_state(output_path, step=25, log_history=_STATE_HISTORY[:2])
            _write_trainer_state(output_path, step=50, log_history=_STATE_HISTORY)
            stale_final = output_path / "checkpoint-final"
            stale_final.mkdir(parents=True)
            (stale_final / "trainer_state.json").write_text(
                json.dumps({"global_step": 1, "log_history": [{"loss": 9.0, "eval_loss": 9.0}]}),
                encoding="utf-8",
            )
            report = parse_aimfttk_run_report(
                request=_request(output_path=output_path),
                duration_seconds=60.0,
                accelerators=("0x74a1", "0x74a1"),
            )

        self.assertEqual(report.final_train_loss, 1.087654321)
        self.assertEqual(report.final_validation_loss, 1.2187654321)

    def test_uses_tokens_per_second_when_token_counts_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory)
            _write_trainer_state(
                output_path,
                step=10,
                log_history=[
                    {
                        "loss": 1.0,
                        "epoch": 1.0,
                        "train_runtime": 10.0,
                        "train_tokens_per_second": 400.0,
                    }
                ],
            )
            report = parse_aimfttk_run_report(
                request=_request(validation=False, accelerator_count=2, output_path=output_path),
                duration_seconds=20.0,
                accelerators=("0x74a1", "0x74a1"),
            )
        self.assertEqual(report.average_tokens_per_gpu_per_second, 200.0)

    def test_falls_back_to_stdout_train_metrics_when_checkpoints_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory)
            stdout_log_path = output_path / "runtime" / "train.stdout.log"
            stdout_log_path.parent.mkdir(parents=True)
            stdout_log_path.write_text(
                "engine startup\n"
                '{"event":"train_metrics","train_loss":2.4123456789,"step":1,"elapsed_seconds":5.0,"tokens_seen":4096}\n'
                '{"event":"unrelated","message":"ignore"}\n'
                '{"event":"train_metrics","train_loss":1.087654321,"step":50,"elapsed_seconds":20.0,"tokens_seen":32768}\n',
                encoding="utf-8",
            )
            report = parse_aimfttk_run_report(
                request=_request(output_path=output_path),
                duration_seconds=60.0,
                accelerators=("0x74a1", "0x74a1"),
            )

        self.assertEqual(report.initial_train_loss, 2.4123456789)
        self.assertEqual(report.final_train_loss, 1.087654321)
        self.assertIsNone(report.initial_validation_loss)
        self.assertIsNone(report.final_validation_loss)
        # Stdout events do not include eval runtime, so 32768 tokens / 20s / 2 GPUs.
        self.assertEqual(report.average_tokens_per_gpu_per_second, 819.2)
        self.assertEqual(report.incomplete, ["initialValidationLoss", "finalValidationLoss"])

    def test_prefers_checkpoint_trainer_state_over_stdout_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory)
            _write_trainer_state(output_path, step=50, log_history=_STATE_HISTORY)
            stdout_log_path = output_path / "runtime" / "train.stdout.log"
            stdout_log_path.parent.mkdir(parents=True)
            stdout_log_path.write_text(
                '{"event":"train_metrics","train_loss":9.0,"step":1,"elapsed_seconds":1.0,"tokens_seen":1}\n',
                encoding="utf-8",
            )
            report = parse_aimfttk_run_report(
                request=_request(output_path=output_path),
                duration_seconds=60.0,
                accelerators=("0x74a1", "0x74a1"),
            )

        self.assertEqual(report.final_train_loss, 1.087654321)
        self.assertEqual(report.final_validation_loss, 1.2187654321)
        self.assertEqual(report.incomplete, [])


def _write_trainer_state(output_path: Path, *, step: int, log_history: list[dict[str, float]]) -> None:
    checkpoint = output_path / "checkpoints" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step, "log_history": log_history}),
        encoding="utf-8",
    )


# Full precision entries, exactly as Trainer.log appends them before any display formatting.
_STATE_HISTORY = [
    {
        "loss": 2.4123456789,
        "epoch": 0.1,
        "num_input_tokens_seen": 4096,
        "step": 1,
        "train_runtime": 5.0,
        "train_tokens_per_second": 819.2,
    },
    {"eval_loss": 2.3712345678, "eval_runtime": 4.0, "epoch": 0.5, "step": 25},
    {
        "loss": 1.087654321,
        "epoch": 1.0,
        "num_input_tokens_seen": 32768,
        "step": 50,
        "train_runtime": 20.0,
        "train_tokens_per_second": 1638.4,
    },
    {"eval_loss": 1.2187654321, "eval_runtime": 4.0, "epoch": 1.0, "step": 50},
]


def _run_captured(command: list[str], *actors: EventActor):
    relayed = StringIO()
    with tempfile.TemporaryDirectory() as directory:
        stdout_log_path = Path(directory) / "train.stdout.log"
        stderr_log_path = Path(directory) / "train.stderr.log"
        with (
            redirect_stdout(relayed),
            TrainLogWriter(stdout_log_path, RawStdoutChunk, StdoutClosed) as stdout_writer,
            TrainLogWriter(stderr_log_path, RawStderrChunk, StderrClosed) as stderr_writer,
            OutputPipeline((stdout_writer, stderr_writer, *actors)) as output_pipeline,
        ):
            captured = run_captured_command(command, dict(os.environ), output_pipeline=output_pipeline)
        stdout_log = stdout_log_path.read_text(encoding="utf-8", errors="replace")
        stderr_log = stderr_log_path.read_text(encoding="utf-8", errors="replace")
    return captured, stdout_log, stderr_log, relayed.getvalue()


class CapturedCommandTests(TestCase):
    def test_relays_both_streams_to_stdout_and_captures_them_separately(self) -> None:
        program = """
import sys
print("hello-stdout")
print("hello-stderr", file=sys.stderr)
"""
        captured, stdout_log, stderr_log, relayed = _run_captured(
            [sys.executable, "-c", program],
            RawStdoutPrinter(),
            RawStderrPrinter(),
        )
        self.assertEqual(captured.returncode, 0)
        self.assertIn("hello-stdout", stdout_log)
        self.assertNotIn("hello-stderr", stdout_log)
        self.assertIn("hello-stderr", stderr_log)
        self.assertNotIn("hello-stdout", stderr_log)
        self.assertIn("hello-stdout", relayed)
        self.assertIn("hello-stderr", relayed)
        self.assertGreaterEqual(captured.duration_seconds, 0.0)

    def test_tee_mode_still_observes_stdout(self) -> None:
        messages: list[TrainMetricsMessage] = []

        class RecordingPrinter:
            def subscriptions(self):
                return (EventSubscription(TrainMetricsMessage, self.print_message),)

            def print_message(self, message: TrainMetricsMessage) -> None:
                messages.append(message)

        program = """
print("hello-training")
print('{"event":"train_metrics","train_loss":1.5,"step":1,"elapsed_seconds":2.0,"tokens_seen":3}')
"""
        _captured, stdout_log, _stderr_log, relayed = _run_captured(
            [sys.executable, "-c", program],
            RawStdoutPrinter(),
            RecordingPrinter(),
            SftOutputObserver(),
        )

        self.assertIn("hello-training", relayed)
        self.assertIn("train_metrics", relayed)
        self.assertIn("hello-training", stdout_log)
        self.assertEqual(
            messages,
            [SftTrainMetricsMessage(train_loss=1.5, step=1, elapsed_seconds=2.0, tokens_seen=3)],
        )

    def test_json_logs_survive_progress_bars_and_multi_rank_noise(self) -> None:
        # A progress bar redraws with a carriage return and no newline, so the cursor sits mid-line
        # when the next event arrives, and under accelerate every rank shares this one stream.
        program = r"""
import sys
def bar(pct, step):
    sys.stderr.write("\r%3d%%|%-10s| %d/400 [00:30<07:30]" % (pct, "#" * (pct // 10), step))
    sys.stderr.flush()
def event(loss, step, elapsed, tokens):
    sys.stdout.write(
        '\n{"event":"train_metrics","train_loss":%r,"step":%d,"elapsed_seconds":%r,"tokens_seen":%d}\n'
        % (loss, step, elapsed, tokens)
    )
    sys.stdout.flush()

bar(6, 25)
event(2.4123456, 25, 30.5, 1024)
sys.stderr.write("[rank1] deepspeed: step=25, skipped=0, lr=[5e-05]\n")
bar(12, 50)
event(2.1987654, 50, 61.0, 2048)
sys.stdout.write("[rank2] partial line with no newline")
bar(100, 400)
event(1.0812345, 400, 480.0, 16384)
"""
        _captured, _stdout_log, stderr_log, relayed = _run_captured(
            [sys.executable, "-c", program],
            JsonStdoutPrinter(),
            SftOutputObserver(),
        )

        events = [json.loads(line) for line in relayed.splitlines() if line.strip()]
        self.assertEqual([event["step"] for event in events], [25, 50, 400])
        self.assertEqual(events[0]["train_loss"], 2.4123456)
        self.assertNotIn("%|", relayed)
        self.assertNotIn("rank", relayed)
        # Everything the engine emitted is still on disk, including what stdout suppressed.
        self.assertIn("%|", stderr_log)
        self.assertIn("[rank1]", stderr_log)

    def test_json_logs_capture_everything_and_relay_only_training_events(self) -> None:
        program = """
print("engine startup noise")
print('{"event":"train_metrics","train_loss":2.5,"step":1,"elapsed_seconds":3.0,"tokens_seen":1024}')
print('{"event":"unrelated","message":"ignore me"}')
print('{"event":"train_metrics","train_loss":1.5,"step":25,"elapsed_seconds":30.0,"tokens_seen":8192}')
print("engine shutdown noise")
"""
        _captured, stdout_log, _stderr_log, relayed = _run_captured(
            [sys.executable, "-c", program],
            JsonStdoutPrinter(),
            SftOutputObserver(),
        )

        self.assertIn("engine startup noise", stdout_log)
        self.assertIn("engine shutdown noise", stdout_log)
        events = [json.loads(line) for line in relayed.splitlines()]
        self.assertEqual(
            events,
            [
                {
                    "event": "train_metrics",
                    "train_loss": 2.5,
                    "step": 1,
                    "elapsed_seconds": 3.0,
                    "tokens_seen": 1024,
                },
                {
                    "event": "train_metrics",
                    "train_loss": 1.5,
                    "step": 25,
                    "elapsed_seconds": 30.0,
                    "tokens_seen": 8192,
                },
            ],
        )

    def test_stdout_and_stderr_lines_are_distinct_events(self) -> None:
        stderr_lines: list[str] = []

        class StderrObserver:
            def subscriptions(self):
                return (EventSubscription(StderrLine, self.observe),)

            def observe(self, event: StderrLine) -> None:
                stderr_lines.append(event.text)

        program = 'import sys; print("stdout line"); print("stderr line", file=sys.stderr)'
        _run_captured([sys.executable, "-c", program], StderrObserver())

        self.assertEqual(stderr_lines, ["stderr line"])

    def test_train_metrics_tokens_seen_is_optional(self) -> None:
        observer = SftOutputObserver()
        message = observer.extract_train_metrics(
            StdoutLine('{"event":"train_metrics","train_loss":1.5,"step":25,"elapsed_seconds":3.0}')
        )
        self.assertEqual(
            message,
            SftTrainMetricsMessage(train_loss=1.5, step=25, elapsed_seconds=3.0),
        )
        assert message is not None
        self.assertNotIn("tokens_seen", message.to_jsonable())

    def test_observer_can_convert_plain_engine_output(self) -> None:
        class PlainOutputObserver:
            def subscriptions(self):
                return (EventSubscription(StdoutLine, self.observe),)

            def observe(self, event: StdoutLine) -> TrainMetricsMessage | None:
                if not event.text.startswith("step="):
                    return None
                step, loss = event.text.split()
                return TrainMetricsMessage(
                    train_loss=float(loss.removeprefix("loss=")),
                    step=int(step.removeprefix("step=")),
                    elapsed_seconds=0.0,
                )

        program = 'print("engine startup"); import sys; sys.stdout.write("step=25 loss=1.5")'
        _captured, stdout_log, _stderr_log, relayed = _run_captured(
            [sys.executable, "-c", program],
            JsonStdoutPrinter(),
            PlainOutputObserver(),
        )

        self.assertEqual(
            json.loads(relayed),
            {
                "event": "train_metrics",
                "train_loss": 1.5,
                "step": 25,
                "elapsed_seconds": 0.0,
            },
        )
        self.assertIn("engine startup", stdout_log)

    def test_pipeline_runs_actors_synchronously_and_fans_out(self) -> None:
        caller_thread = threading.current_thread().name
        observer_threads: list[str] = []
        printer_threads: list[str] = []
        first_messages: list[LogMessage | TrainMetricsMessage] = []
        second_messages: list[LogMessage | TrainMetricsMessage] = []

        class Observer:
            def subscriptions(self):
                return (EventSubscription(StdoutLine, self.observe),)

            def observe(self, event: StdoutLine) -> TrainMetricsMessage:
                observer_threads.append(threading.current_thread().name)
                return TrainMetricsMessage(train_loss=1.0, step=1, elapsed_seconds=2.0, tokens_seen=3)

        class RecordingPrinter:
            def __init__(self, messages: list[LogMessage | TrainMetricsMessage]) -> None:
                self.messages = messages

            def print_message(self, message: LogMessage | TrainMetricsMessage) -> None:
                printer_threads.append(threading.current_thread().name)
                self.messages.append(message)

            def subscriptions(self):
                return (
                    EventSubscription(LogMessage, self.print_message),
                    EventSubscription(TrainMetricsMessage, self.print_message),
                )

        with OutputPipeline(
            (RecordingPrinter(first_messages), RecordingPrinter(second_messages), Observer()),
        ) as output_pipeline:
            output_pipeline.publish(RawStdoutChunk(b"line\n"))
            output_pipeline.publish(LogMessage(message="runtime"))

        expected = [
            LogMessage(message="runtime"),
            TrainMetricsMessage(train_loss=1.0, step=1, elapsed_seconds=2.0, tokens_seen=3),
        ]
        self.assertCountEqual(first_messages, expected)
        self.assertCountEqual(second_messages, expected)
        self.assertEqual(observer_threads, [caller_thread])
        self.assertEqual(printer_threads, [caller_thread] * 4)

    def test_actor_runs_on_the_publishing_source_thread(self) -> None:
        actor_threads: list[str] = []

        class Actor:
            def subscriptions(self):
                return (EventSubscription(LogMessage, self.observe),)

            def observe(self, message: LogMessage) -> None:
                actor_threads.append(threading.current_thread().name)

        with OutputPipeline((Actor(),)) as output_pipeline:
            source = threading.Thread(
                target=lambda: output_pipeline.publish(LogMessage(message="from source")),
                name="tensorboard-source",
            )
            source.start()
            source.join()

        self.assertEqual(actor_threads, ["tensorboard-source"])

    def test_pipeline_isolates_a_failed_printer(self) -> None:
        messages: list[LogMessage] = []

        class FailedPrinter:
            def subscriptions(self):
                return (EventSubscription(LogMessage, self.print_message),)

            def print_message(self, message: LogMessage) -> None:
                raise RuntimeError("printer unavailable")

        class RecordingPrinter:
            def subscriptions(self):
                return (EventSubscription(LogMessage, self.print_message),)

            def print_message(self, message: LogMessage) -> None:
                messages.append(message)

        with OutputPipeline((FailedPrinter(), RecordingPrinter())) as output_pipeline:
            output_pipeline.publish(LogMessage(message="still delivered"))

        self.assertEqual(messages, [LogMessage(message="still delivered")])


class TrainLogWriterTests(TestCase):
    def test_writes_stream_log_incrementally_from_published_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "train.stdout.log"
            with TrainLogWriter(log_path, RawStdoutChunk, StdoutClosed) as writer:
                with OutputPipeline((writer,)) as output_pipeline:
                    output_pipeline.publish(RawStdoutChunk(b"first\n"))
                    self.assertEqual(log_path.read_bytes(), b"first\n")
                    output_pipeline.publish(RawStdoutChunk(b"second\n"))
                    self.assertEqual(log_path.read_bytes(), b"first\nsecond\n")


class RuntimeLogTests(TestCase):
    def test_json_logs_emit_structured_runtime_messages_on_stdout(self) -> None:
        relayed = StringIO()
        with redirect_stdout(relayed):
            with OutputPipeline((JsonStdoutPrinter(),)) as output_pipeline:
                output_pipeline.publish(LogMessage(message="Rendered workspace"))
                output_pipeline.publish(
                    LogMessage(message="Training report incomplete fields: durationSeconds", level="warning")
                )

        events = [json.loads(line) for line in relayed.getvalue().splitlines()]
        self.assertEqual(
            events,
            [
                {"event": "log", "level": "info", "message": "Rendered workspace"},
                {
                    "event": "log",
                    "level": "warning",
                    "message": "Training report incomplete fields: durationSeconds",
                },
            ],
        )

    def test_plain_runtime_messages_stay_text(self) -> None:
        relayed = StringIO()
        with redirect_stdout(relayed):
            with OutputPipeline((PlainLogPrinter(),)) as output_pipeline:
                output_pipeline.publish(LogMessage(message="Rendered workspace"))

        self.assertEqual(relayed.getvalue(), "Rendered workspace\n")


class TrainingStatusHeartbeatTests(TestCase):
    def test_writes_running_heartbeats_and_completed_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            status_path = Path(directory) / "train-status.json"
            with TrainingStatusHeartbeat(status_path, interval_seconds=0.01):
                first_timestamp = json.loads(status_path.read_text(encoding="utf-8"))["timestamp"]
                time.sleep(0.03)
                running = json.loads(status_path.read_text(encoding="utf-8"))

            completed = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(running["status"], "RUNNING")
        self.assertNotEqual(running["timestamp"], first_timestamp)
        self.assertEqual(completed["status"], "COMPLETED")
        self.assertTrue(completed["timestamp"].endswith("Z"))

    def test_writes_failed_status_when_execution_raises(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            status_path = Path(directory) / "train-status.json"
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with TrainingStatusHeartbeat(status_path, interval_seconds=0.01):
                    raise RuntimeError("training failed")
            failed = json.loads(status_path.read_text(encoding="utf-8"))

        self.assertEqual(failed["status"], "FAILED")


if __name__ == "__main__":
    main()
