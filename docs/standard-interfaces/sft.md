<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Fine-tuning Standardized SFT Image Interface

This page is the SFT image parameter reference. For the surrounding workflow,
see the [CLI reference](../aim-fine-tuning/reference/cli.md), [recipe schema
reference](../aim-fine-tuning/reference/recipe-schema.md), and [engine
concepts](../aim-fine-tuning/concepts/engines.md). engine-specific launch
details are documented in the [engine reference](../aim-fine-tuning/reference/engines/aimfttk.md).

## Output paths relative to AIMFT_OUTPUT_PATH

The runtime outputs are created in `${AIMFT_OUTPUT_PATH}/runtime`:

- `train-request.json` contains the training request metadata and the resolved parameters.
- `launch-spec.json` contains the training execution info, such as locations.
- `run.sh` may be included, containing a command that will execute the training run.
- During execute, `train-status.json` contains `RUNNING`, `COMPLETED`, or `FAILED` and a UTC `timestamp`. The runtime rewrites it atomically every 10 seconds while training runs.
- During execute, `train.stdout.log` and `train.stderr.log` are written incrementally as their respective captured streams are published. Both streams are relayed to stdout by default. When `AIMFT_JSON_LOGS` is enabled, stdout is JSON Lines: standardized `train_metrics` events from the engine plus runtime `log` events. Engine chatter that is not a training metric is not relayed live; the two log files still contain all engine output.
- After execute, `train-report.json` contains validation metrics. Accelerators are the devices detected at runtime startup. Remaining fields are read from the last numbered `checkpoints/checkpoint-*/trainer_state.json`. If no numbered checkpoint exists (`AIMFT_SAVE_INTERVAL=-1`), train losses and throughput fall back to `train_metrics` events in `train.stdout.log`; validation losses are not in that stream. Intermediate step metrics are the JSON Lines on stdout. `averageTokensPerGpuPerSecond` covers the time spent training, with validation excluded when `trainer_state.json` is available. Fields that could not be extracted are listed in `incomplete`.

Backend stdout and stderr are published as distinct exact byte-chunk events. Sources own their scheduling, while the central controller synchronously calls every matching actor and republishes derived events. Core actors derive decoded lines for each stream, and backend actors may turn those into standardized messages. New event classes can be published without changing the controller. `AIMFT_JSON_LOGS` only selects the console actors: JSON Lines messages versus both raw streams.

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
  "incomplete": []
}
```

Missing validation losses are expected when `AIMFT_VALIDATION_DATA_PATH` is unset.

The standardized intermediate training event has this form:

```json
{"event":"train_metrics","train_loss":2.412,"step":25,"elapsed_seconds":30.5,"tokens_seen":8192}
```

`tokens_seen` is optional.

Runtime progress and errors use the same stream:

```json
{"event":"log","level":"info","message":"Execution completed successfully."}
{"event":"log","level":"warning","message":"Training report incomplete fields: initialValidationLoss"}
{"event":"log","level":"error","message":"Backend execution failed with exit code 1."}
```

`elapsed_seconds` is the engine's elapsed training-loop wall clock and includes validation and checkpointing.

## Environment Variables

The environment variables use a common prefix of `AIMFT_` to avoid overloading variable names and to indicate which variables correspond to AIM Fine-tuning parameters. Certain other environment variables such as `HF_TOKEN` might be used by AIM Fine-tuning, but derive from other sources and thus do not use the prefix.

### Environment Variable Value Encoding

The values of the environment variables are encoded in JSON. The values can any of the basic types representable in JSON: *string*, *integer*, *float*, *object* (i.e. dictionary mapping), *array* (i.e. list), *boolean*, *null*. Additionally, values may be unset.

### Table of SFT Image Environment Variables

| **Key** | **Type** | **Default value / Required** | **Behaviour** |
|--|--|--|--|
| **Required** ||| *These variables must be set.* |
| `AIMFT_TRAIN_DATA_PATH` | *string* | **Required** | Path inside the container to the training data file, in the SFT Data Format. |
| `AIMFT_MODEL_PATH` | *string* | **Required** | Path inside the container to the directory that contains the model, in the Hugging Face format. |
| `AIM_MODEL_ID` | *string* | **Required** | AIM ModelID, the canonical identifier of the model weight set. |
| `AIMFT_OUTPUT_PATH` | *string* | **Required** | Path inside the container to the directory where outputs are produced. |
|||||
| **Optional** ||| *These additional variables may be set, but are not required. If not set, they take a default value.* |
| `AIMFT_RECIPE_ID` | *string\|unset* | *unset* | If set, manually override recipe selection and attempt to use the recipe corresponding to this ID. |
| `AIMFT_METHOD` | *string* | *"SFT"* | The training algorithm or method to use. In SFT images this is set to "SFT" by default. |
| `AIMFT_USE_ADAPTER` | *string* | *False* | If True, chooses an adapter recipe (typically LoRA). If False, chooses a full model recipe. |
| `AIMFT_VALIDATION_DATA_PATH` | *string\|null* | *null* | Path inside the container to the validation data in the SFT Data Format. A null value indicates no separate validation data file is used and validation is not done. |
| `AIMFT_EPOCHS` | *integer* | *1* | Sets the number of times to iterate over the training data. |
| `AIMFT_SEED` | *integer* | *42* | Sets the random seed. |
| `AIMFT_EVALUATION_INTERVAL` | *integer* | *1000* | Sets the number of optimization steps between evaluations. |
| `AIMFT_LOGGING_INTERVAL` | *integer* | *25* | Sets the number of optimization steps between log outputs. |
| `AIMFT_JSON_LOGS` | *boolean* | *False* | If True, stdout is JSON Lines (`train_metrics` and runtime `log` events) instead of relaying raw backend stdout and stderr. Observation and `runtime/train.stdout.log`/`runtime/train.stderr.log` capture always run. |
| `AIMFT_SAVE_INTERVAL` | *integer* | *500* | Sets the number of steps between intermediate checkpoint outputs. If set to -1, no checkpoints are saved. |
| `AIMFT_LEARNING_RATE` | *float* | *Recipe-dependent* | Sets the learning rate. |
| `AIMFT_LEARNING_RATE_WARMUP_STEPS` | *integer* | *Recipe-dependent* | Sets the number of initial training steps over which learning rate is ramped up. |
| `AIMFT_PER_ACCELERATOR_BATCH_SIZE` | *integer* | *Recipe-dependent* | Sets the batch size per accelerator. Total batch size is `AIMFT_PER_ACCELERATOR_BATCH_SIZE` \* `AIMFT_GRADIENT_ACCUMULATION` \* `The level of data parallelism`. The level of data parallelism is defined by the type of distributed setup, and depends on the recipe. |
| `AIMFT_GRADIENT_ACCUMULATION` | *integer* | *Recipe-dependent* | Sets the number of batches over which gradients are accumulated before taking an optimizer step. |
| `AIMFT_LORA_ALPHA` | *integer* | *Recipe-dependent* | Sets the alpha parameter of the LoRA Adapter. |
| `AIMFT_LORA_DROPOUT` | *float* | *Recipe-dependent; 0.0 if unset and LoRA is enabled* | Sets the dropout probability of the LoRA Adapter. |
| `AIMFT_LORA_RANK` | *integer* | *Recipe-dependent* | Sets the rank parameter of the LoRA Adapter. |
| `AIMFT_MAX_CONTEXT_LENGTH` | *integer* | *Recipe-dependent* | Sets the maximum context length in tokens. Examples longer than this are truncated. |
