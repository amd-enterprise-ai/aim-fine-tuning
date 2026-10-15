# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import json
from types import SimpleNamespace

from aimfttk.utils.structured_logging import StructuredTrainingLogCallback


def test_emits_standardized_training_event(capsys):
    state = SimpleNamespace(is_world_process_zero=True, global_step=25)

    StructuredTrainingLogCallback().on_log(
        args=SimpleNamespace(),
        state=state,
        control=SimpleNamespace(),
        logs={
            "loss": 2.412,
            "train_runtime": 30.5,
            "num_input_tokens_seen": 8192,
            "learning_rate": 0.00005,
        },
    )

    assert json.loads(capsys.readouterr().out) == {
        "event": "train_metrics",
        "train_loss": 2.412,
        "step": 25,
        "elapsed_seconds": 30.5,
        "tokens_seen": 8192,
    }


def test_event_starts_its_own_line_so_progress_bars_cannot_glue_onto_it(capsys):
    state = SimpleNamespace(is_world_process_zero=True, global_step=25)

    StructuredTrainingLogCallback().on_log(
        args=SimpleNamespace(),
        state=state,
        control=SimpleNamespace(),
        logs={"loss": 2.412, "train_runtime": 30.5, "num_input_tokens_seen": 8192},
    )

    written = capsys.readouterr().out
    assert written.startswith("\n")
    assert written.endswith("\n")
    assert json.loads(written)["step"] == 25


def test_tokens_seen_is_optional(capsys):
    state = SimpleNamespace(is_world_process_zero=True, global_step=25)

    StructuredTrainingLogCallback().on_log(
        args=SimpleNamespace(),
        state=state,
        control=SimpleNamespace(),
        logs={"loss": 2.412, "train_runtime": 30.5},
    )

    assert json.loads(capsys.readouterr().out) == {
        "event": "train_metrics",
        "train_loss": 2.412,
        "step": 25,
        "elapsed_seconds": 30.5,
    }


def test_only_the_main_rank_emits_events(capsys):
    state = SimpleNamespace(is_world_process_zero=False, global_step=25)

    StructuredTrainingLogCallback().on_log(
        args=SimpleNamespace(),
        state=state,
        control=SimpleNamespace(),
        logs={"loss": 2.412, "train_runtime": 30.5, "num_input_tokens_seen": 8192},
    )

    assert capsys.readouterr().out == ""


def test_skips_logs_without_training_metrics(capsys):
    state = SimpleNamespace(is_world_process_zero=True, global_step=50)

    # Evaluation logs carry no train loss, so they are not training events.
    StructuredTrainingLogCallback().on_log(
        args=SimpleNamespace(),
        state=state,
        control=SimpleNamespace(),
        logs={"eval_loss": 1.5, "eval_runtime": 4.0, "num_input_tokens_seen": 8192},
    )

    assert capsys.readouterr().out == ""
