<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Fine Tune CLI

`aim_fine_tune` prepares and runs fine-tuning jobs from local inputs.

Today the CLI supports:

- backend: `aimfttk`
- methods: `sft`
- train modes: `dry-run` and `execute`
- local model, dataset, and output paths
- recipe discovery and recipe-based launch generation

This README is focused on how to use the CLI. It avoids backend implementation details and internal packaging notes.

## Quick Start

If you are working from Windows, run the commands from WSL in the `aim_fine_tune/` directory.

```bash
uv sync
uv run aim_fine_tune --help
uv run aim_fine_tune train --help
uv run aim_fine_tune list-recipes --help
```

## Commands

- `aim_fine_tune list-recipes`: show packaged recipes, optionally filtered to compatible ones
- `aim_fine_tune train`: render a backend launch workspace or execute the backend command

The CLI may also permit `debug-download` for fetching a model or dataset during development. It is not a feature for production use. S3 downloads need the optional dependency: `uv sync --extra download`.

## Required Inputs For `train`

| CLI flag | Environment variable | Meaning |
| --- | --- | --- |
| `--model-path` | `AIMFT_MODEL_PATH` | Local model directory |
| `--train-data-path` | `AIMFT_TRAIN_DATA_PATH` | Local training dataset file |
| `--output-path` | `AIMFT_OUTPUT_PATH` | Output directory for generated files and training outputs |
| `--aim-model-id` | `AIM_MODEL_ID` | Model identifier used for recipe selection |

Common optional inputs:

| CLI flag | Environment variable | Meaning |
| --- | --- | --- |
| `--validation-data-path` | `AIMFT_VALIDATION_DATA_PATH` | Optional validation dataset file |
| `--recipe-id` | `AIMFT_RECIPE_ID` | Use one specific recipe by unique recipe ID (`metadata.recipeId`) |
| `--extra-recipes` | `AIMFT_EXTRA_RECIPES` | Directory of extra recipes added to the built-in ones |
| `--recipe-path` | `AIMFT_RECIPE_PATH` | Use this recipe file directly, without the recipe catalog or compatibility checks |

Training tuning and backend options are configured through environment variables such as `AIMFT_USE_ADAPTER`, `AIMFT_EPOCHS`, `AIMFT_SEED`, `AIMFT_EVALUATION_INTERVAL`, `AIMFT_LOGGING_INTERVAL`, `AIMFT_LEARNING_RATE`, `AIMFT_LEARNING_RATE_WARMUP_STEPS`, `AIMFT_PER_ACCELERATOR_BATCH_SIZE`, `AIMFT_GRADIENT_ACCUMULATION`, `AIMFT_LORA_ALPHA`, `AIMFT_LORA_DROPOUT`, `AIMFT_LORA_RANK`, `AIMFT_MAX_CONTEXT_LENGTH`, and `AIMFT_ENGINE_ARGS`.

CLI flags take precedence over environment variables.

For environment variables, string values can be passed directly. Non-string values should be valid JSON literals such as `true`, `false`, `1`, or `0.0001`.

## Typical Workflow

1. Download or stage the model and dataset locally.
2. Inspect available recipes.
3. Run `train` in dry-run mode to render the launch workspace.
4. Run `train --execute` inside the target execution environment when you are ready to launch.

## List Recipes

Use `list-recipes` to see what recipes are available.

```bash
uv run aim_fine_tune list-recipes
uv run aim_fine_tune list-recipes --compatible-only
uv run aim_fine_tune list-recipes --aim-id google/gemma-3-27b-it
uv run aim_fine_tune list-recipes --output-format json
uv run aim_fine_tune list-recipes --extra-recipes /path/to/extra-recipes
```

Use `--recipe-id` with `train` to pin one specific recipe.

The built-in recipes are always read from `/opt/aim_fine_tune/builtin_recipes`, where AIMFT images install them. To add your
own, mount a directory of recipe YAML files with its own `aimft-recipe-compatibility.yaml` and point `--extra-recipes`
or `AIMFT_EXTRA_RECIPES` at it. Extra recipes are selected like built-in ones; one with the same recipe ID replaces the
built-in recipe, and one missing from the compatibility file is never selected.

To use one recipe file as is, for example on a machine without the target accelerators, pass it with `--recipe-path`
or `AIMFT_RECIPE_PATH`. This skips the recipe catalog and the compatibility checks, but the recipe must still match the
requested model, method, and accelerator count:

```bash
uv run aim_fine_tune train \
	--dry-run \
	--recipe-path recipes/aimfttk/sft/google/gemma-3-27b-it.yaml \
	--model-path inputs/model \
	--train-data-path inputs/data.jsonl \
	--output-path outputs \
	--aim-model-id google/gemma-3-27b-it
```

## Render A Launch Workspace

Dry-run is the default. It renders the backend launch workspace without starting the backend process.

```bash
uv run aim_fine_tune train \
	--backend aimfttk \
	--model-path inputs/model \
	--train-data-path inputs/data.jsonl \
	--output-path outputs \
	--aim-model-id amd/your-model-id \
	--recipe-id aimfttk/sft/google-gemma-3-27b-it/lora/1gpu/default \
	--render-dir outputs
```

The rendered launch workspace is written under the output directory. Runtime metadata artifacts are written under `AIMFT_OUTPUT_PATH/runtime` by default:

- `runtime/train-request.json`
- `runtime/launch-spec.json`
- `runtime/run.sh`

After `--execute`, the runtime also writes:

- `runtime/train-status.json`: an atomic, timestamped `RUNNING`, `COMPLETED`, or `FAILED` status, refreshed every 10 seconds while training runs
- `runtime/train.stdout.log`: captured training process stdout, written incrementally, including output hidden by `--json-logs`
- `runtime/train.stderr.log`: captured training process stderr, written incrementally, including output hidden by `--json-logs`
- `runtime/train-report.json`: detected accelerators, initial/final train and validation losses, wall-clock duration, and average tokens/GPU/second over time spent training, excluding validation. Losses and throughput come from the last numbered `checkpoints/checkpoint-*/trainer_state.json` when present, otherwise from `train_metrics` events in `runtime/train.stdout.log`. Intermediate step metrics are the JSON Lines on stdout.

Backend config files remain under the existing config directory:

- `configs/accelerate_config.yaml`
- `configs/finetuning_config.yaml`
- `configs/entrypoint.sh`

Training outputs are written under the output directory, including checkpoint content and final model artifacts.

Pass `--json-logs` (or set `AIMFT_JSON_LOGS=true`) to suppress the engine's human-readable output on
stdout and emit JSON Lines instead. Training updates look like:

```json
{"event":"train_metrics","train_loss":2.412,"step":25,"elapsed_seconds":30.5,"tokens_seen":8192}
```

Runtime messages use the same stream:

```json
{"event":"log","level":"info","message":"Execution completed successfully."}
```

The output is JSON Lines: every event occupies one line. `elapsed_seconds` is Hugging Face's elapsed
training-loop wall clock and therefore includes validation and checkpointing. This option changes only
what is relayed to stdout; `runtime/train.stdout.log` and `runtime/train.stderr.log` always retain the complete
engine output.

At the end of every `--execute` run, including failed ones, the contents of `runtime/train-report.json` are also
printed to stdout as a single JSON line, with or without `--json-logs`, so the report can be collected from
container logs:

```json
{"event":"train_report","report":{"backend":"aimfttk","accelerators":["0x74a1"],"initialTrainLoss":2.5,"finalTrainLoss":1.1,"ranToCompletion":true,"...":"..."}}
```

## Run With Environment Variables

```bash
export AIMFT_MODEL_PATH='inputs/model'
export AIMFT_TRAIN_DATA_PATH='inputs/data.jsonl'
export AIMFT_OUTPUT_PATH='outputs'
export AIM_MODEL_ID='amd/your-model-id'
export AIMFT_RECIPE_ID='aimfttk/sft/google-gemma-3-27b-it/lora/1gpu/default'
export AIMFT_ENGINE_ARGS='{"distributed_type":"auto-deepspeed-stage1","merge_adapter":true}'
export AIMFT_USE_ADAPTER='true'
export AIMFT_EPOCHS='1'
export AIMFT_SEED='42'
export AIMFT_EVALUATION_INTERVAL='1000'
export AIMFT_LOGGING_INTERVAL='25'
export AIMFT_JSON_LOGS='true'
export AIMFT_PER_ACCELERATOR_BATCH_SIZE='1'
export AIMFT_GRADIENT_ACCUMULATION='4'

uv run aim_fine_tune train
```

## Execute A Job

Use `--execute` when the CLI is already running inside the target execution container and the expected mounts already exist. Invoke the installed `aim_fine_tune` console script (for example `/usr/bin/aim_fine_tune`). Do not activate the wrapper virtual environment and do not use `uv run` for execute mode: those prepend the wrapper `bin/` to `PATH`, so engine tools that use `#!/usr/bin/env python3` would run under the wrapper interpreter.

```bash
aim_fine_tune train \
	--execute \
	--backend aimfttk \
	--model-path inputs/model \
	--train-data-path inputs/data.jsonl \
	--output-path outputs \
	--aim-model-id amd/your-model-id
```

Current execute mode assumptions:

- the CLI is running inside the execution container
- the CLI may run from its own virtual environment via the console-script shebang
- `accelerate`, `aimfttk`, `python3`, and backend helper commands must already be on the process `PATH` from the engine image; `--execute` does not rewrite `PATH`
- the model directory, dataset file, and output directory are the paths passed to the CLI
- `aimfttk` writes generated config files under the output directory in `configs/`

## Supported Engine Args

Backend-specific settings are passed through `AIMFT_ENGINE_ARGS` as a JSON object. These args follow the same structure as the `engine_args` field in the recipe and are merged over the recipe defaults. `aimfttk` accepts `distributed_type`, `merge_adapter`, and `finetuning_config`. Other top-level keys are rejected. Nested AIMFTTK settings, including the chat template, belong under `finetuning_config`.
