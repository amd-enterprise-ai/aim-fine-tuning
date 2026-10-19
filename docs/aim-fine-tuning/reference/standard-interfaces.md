<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIMFT Image interface

This page is the parameter reference for an AIMFT Image running supervised fine-tuning (SFT). Jobs are configured with the environment variables below. For the surrounding workflow, see the [recipe schema reference](recipe-schema.md) and [Engines](engines.md). Backend-specific launch details are documented in the [AIM Fine-tuning Toolkit reference](engines/aimfttk.md).

## Output paths

Runtime files are created in `${AIMFT_OUTPUT_PATH}/runtime`:

- `train-request.json` contains the training request metadata and the resolved parameters.
- `launch-spec.json` contains the training execution info, such as the command and generated file locations.
- `run.sh` contains a command that executes the training run.
- During execute, `train-status.json` contains `RUNNING`, `COMPLETED`, or `FAILED` and a UTC `timestamp`. The runtime rewrites it atomically every 10 seconds while training runs.
- During execute, `train.stdout.log` and `train.stderr.log` are written incrementally as their captured streams are published. Both streams are relayed to stdout by default. When `AIMFT_JSON_LOGS` is enabled, stdout is JSON Lines: standardized `train_metrics` events from the engine plus runtime `log` events. Engine output that is not a training metric is kept in the two log files.
- After execute, `train-report.json` records the run. Accelerators are the devices detected at startup, as normalized device IDs such as `0x74a1`. Train losses and throughput are read from the last numbered `checkpoints/checkpoint-*/trainer_state.json`. If no numbered checkpoint exists (`AIMFT_SAVE_INTERVAL=-1`), those fields fall back to `train_metrics` events in `train.stdout.log`. Validation losses are taken from the checkpoint state and are absent from that fallback stream. `averageTokensPerGpuPerSecond` covers the time spent training, with validation excluded when `trainer_state.json` is available. Fields that could not be extracted are listed in `incomplete`.

The `aimfttk` backend also writes its launch files under `${AIMFT_OUTPUT_PATH}/configs`: `accelerate_config.yaml`, `finetuning_config.yaml`, and `entrypoint.sh`.

With adapter training, the final adapter is written to `${AIMFT_OUTPUT_PATH}/checkpoint-final-adapter`. Full-parameter training writes `${AIMFT_OUTPUT_PATH}/checkpoint-final`.

```json
{
  "status": "RUNNING",
  "timestamp": "2026-09-09T13:04:05.123456Z"
}
```

```json
{
  "backend": "aimfttk",
  "accelerators": ["0x74a1"],
  "initialTrainLoss": 2.41,
  "finalTrainLoss": 1.08,
  "initialValidationLoss": 2.37,
  "finalValidationLoss": 1.21,
  "durationSeconds": 3612.4,
  "averageTokensPerGpuPerSecond": 1842.0,
  "incomplete": [],
  "ranToCompletion": true,
  "finishedAt": "2026-09-09T14:04:17.123456Z",
  "recipeId": "aimfttk/sft/google-gemma-3-27b-it/lora/1gpu/default",
  "recipeDigest": "sha256:...",
  "trainDataDigest": "sha256:...",
  "validationDataDigest": null
}
```

The report also includes `engineImage`, `aimFineTuneVersion`, and the resolved `request` when those values are available. Missing validation losses are expected when `AIMFT_VALIDATION_DATA_PATH` is unset.

The standardized intermediate training event has this form:

```json
{"event":"train_metrics","train_loss":2.412,"step":25,"elapsed_seconds":30.5,"tokens_seen":8192}
```

`tokens_seen` is optional. `elapsed_seconds` is the engine's elapsed training-loop wall clock and includes validation and checkpointing.

Runtime progress and errors use the same stream when `AIMFT_JSON_LOGS` is enabled:

```json
{"event":"log","level":"info","message":"Execution completed successfully."}
{"event":"log","level":"warning","message":"Training report incomplete fields: initialValidationLoss"}
{"event":"log","level":"error","message":"Backend execution failed with exit code 1."}
```

At the end of every execute run, including a failed one, `train-report.json` is also printed to stdout as one JSON line:

```json
{"event":"train_report","report":{"backend":"aimfttk","ranToCompletion":true}}
```

## Environment variables

The environment variables use a common prefix of `AIMFT_` to avoid overloading variable names and to indicate which variables correspond to AIM Fine-tuning parameters. Certain other environment variables such as `HF_TOKEN` might be used by AIM Fine-tuning, but derive from other sources and thus do not use the prefix.

### Environment variable value encoding

Values are decoded as JSON when the text is valid JSON, so `true`, `1`, and `0.0001` keep those types. A value that is not valid JSON is used as a string. A variable may also be unset.

## Method specific interfaces

Contracts shared by AIMFT Images and their surrounding execution systems. These interfaces describe how an AIMFT Image receives configuration, data, and output paths.

- [SFT Interface](standard-sft-image-interface.md)

More training methods may be added in the future, each with its own interface specification.
