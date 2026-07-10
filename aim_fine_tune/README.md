<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Fine Tune CLI

`aim_fine_tune` prepares and runs fine-tuning jobs from local inputs.

Today the CLI supports:

- backend: `aimfttk`
- method: `sft`
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
uv run aim_fine_tune download --help
uv run aim_fine_tune list-recipes --help
```

## Commands

- `aim_fine_tune download`: download a model snapshot and a dataset into local paths
- `aim_fine_tune list-recipes`: show packaged recipes, optionally filtered to compatible ones
- `aim_fine_tune train`: render a backend launch workspace or execute the backend command

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
| `--recipes-root` | `AIMFT_RECIPES_ROOT` | Override the default recipe directory |
| `--allow-incompatible-recipe` | `AIMFT_ALLOW_INCOMPATIBLE_RECIPE` | Allow dry-run rendering with an explicit recipe even when the local accelerator setup does not match |

Training tuning and backend options are configured through environment variables such as `AIMFT_USE_ADAPTER`, `AIMFT_EPOCHS`, `AIMFT_SEED`, `AIMFT_EVALUATION_INTERVAL`, `AIMFT_LOGGING_INTERVAL`, `AIMFT_LEARNING_RATE`, `AIMFT_LEARNING_RATE_WARMUP_STEPS`, `AIMFT_PER_ACCELERATOR_BATCH_SIZE`, `AIMFT_GRADIENT_ACCUMULATION`, `AIMFT_LORA_ALPHA`, `AIMFT_LORA_DROPOUT`, `AIMFT_LORA_RANK`, `AIMFT_MAX_CONTEXT_LENGTH`, `AIMFT_NUM_ACCELERATORS`, and `AIMFT_ENGINE_ARGS`.

CLI flags take precedence over environment variables.

For environment variables, string values can be passed directly. Non-string values should be valid JSON literals such as `true`, `false`, `1`, or `0.0001`.

## Typical Workflow

1. Download or stage the model and dataset locally.
2. Inspect available recipes.
3. Run `train` in dry-run mode to render the launch workspace.
4. Run `train --execute` inside the target execution environment when you are ready to launch.

## Download Assets

Use `download` when the model comes from Hugging Face and the dataset comes from S3 or MinIO.

Model download inputs:

| CLI flag | Environment variable | Meaning |
| --- | --- | --- |
| `--hf-model-id` | `AIMFT_HF_MODEL_ID` | Hugging Face model id |
| `--model-output-path` | `AIMFT_MODEL_OUTPUT_PATH` | Local model output directory |
| `--model-cache-dir` | `AIMFT_MODEL_CACHE_DIR` | Optional Hugging Face cache directory |
| `--hf-token` | `AIMFT_HF_TOKEN` or `HF_TOKEN` | Hugging Face token |

Dataset download inputs:

| CLI flag | Environment variable | Meaning |
| --- | --- | --- |
| `--dataset-s3-uri` | `AIMFT_DATASET_S3_URI` | Dataset source such as `s3://bucket/path` |
| `--dataset-output-path` | `AIMFT_DATASET_OUTPUT_PATH` | Local dataset output directory |
| `--s3-endpoint-url` | `AIMFT_S3_ENDPOINT_URL`, `AWS_ENDPOINT_URL`, or `BUCKET_STORAGE_HOST` | S3 or MinIO endpoint |
| `--s3-access-key-id` | `AIMFT_S3_ACCESS_KEY_ID`, `AWS_ACCESS_KEY_ID`, or `BUCKET_STORAGE_ACCESS_KEY` | Access key |
| `--s3-secret-access-key` | `AIMFT_S3_SECRET_ACCESS_KEY`, `AWS_SECRET_ACCESS_KEY`, or `BUCKET_STORAGE_SECRET_KEY` | Secret key |
| `--s3-session-token` | `AIMFT_S3_SESSION_TOKEN` or `AWS_SESSION_TOKEN` | Optional session token |
| `--s3-region` | `AIMFT_S3_REGION`, `AWS_REGION`, or `AWS_DEFAULT_REGION` | Optional region |

Example:

```bash
export HF_TOKEN='hf_xxx'
export BUCKET_STORAGE_HOST='http://minio:9000'
export BUCKET_STORAGE_ACCESS_KEY='minio-access'
export BUCKET_STORAGE_SECRET_KEY='minio-secret'

uv run aim_fine_tune download \
	--hf-model-id google/gemma-3-27b-it \
	--model-output-path inputs/model \
	--dataset-s3-uri s3://default-bucket/path/to/dataset \
	--dataset-output-path inputs/dataset
```

After the assets are staged locally, pass those paths to `train`.

## List Recipes

Use `list-recipes` to see what recipes are available.

```bash
uv run aim_fine_tune list-recipes
uv run aim_fine_tune list-recipes --compatible-only
uv run aim_fine_tune list-recipes --aim-id google/gemma-3-27b-it
uv run aim_fine_tune list-recipes --output-format json
uv run aim_fine_tune list-recipes --recipes-root /path/to/recipes
```

Use `--recipe-id` with `train` to pin one specific recipe.

If you want to render a specific recipe on a machine that does not have the target accelerator setup, you can bypass compatibility checks for dry-run only:

```bash
uv run aim_fine_tune train \
	--dry-run \
	--recipe-id aimfttk/sft/google-gemma-3-27b-it/lora/1gpu/default \
	--allow-incompatible-recipe \
	--model-path inputs/model \
	--train-data-path inputs/data.jsonl \
	--output-path outputs \
	--aim-model-id google/gemma-3-27b-it
```

This bypass only applies to dry-run. The recipe id, model id, and accelerator count still need to line up.

By default the CLI looks for recipes in `/workspace/recipes` first and falls back to the packaged `recipes/` directory.

If the CLI is installed into another environment and the default recipe directories are not available there, copy the repository `recipes/` directory and point `--recipes-root` or `AIMFT_RECIPES_ROOT` at it.

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

Backend config files remain under the existing config directory:

- `configs/accelerate_config.yaml`
- `configs/finetuning_config.yaml`
- `configs/entrypoint.sh`

Training outputs are written under the output directory, including checkpoint content and final model artifacts.

## Run With Environment Variables

```bash
export AIMFT_MODEL_PATH='inputs/model'
export AIMFT_TRAIN_DATA_PATH='inputs/data.jsonl'
export AIMFT_OUTPUT_PATH='outputs'
export AIM_MODEL_ID='amd/your-model-id'
export AIMFT_RECIPE_ID='aimfttk/sft/google-gemma-3-27b-it/lora/1gpu/default'
export AIMFT_ENGINE_ARGS='{"distributed_type":"auto-deepspeed-stage1","chat_template":"keep-original","merge_adapter":true}'
export AIMFT_USE_ADAPTER='true'
export AIMFT_EPOCHS='1'
export AIMFT_SEED='42'
export AIMFT_EVALUATION_INTERVAL='1000'
export AIMFT_LOGGING_INTERVAL='25'
export AIMFT_PER_ACCELERATOR_BATCH_SIZE='1'
export AIMFT_GRADIENT_ACCUMULATION='4'
export AIMFT_NUM_ACCELERATORS='2'

uv run aim_fine_tune train
```

## Execute A Job

Use `--execute` when the CLI is already running inside the target execution container and the expected mounts already exist.

```bash
uv run aim_fine_tune train \
	--execute \
	--backend aimfttk \
	--model-path inputs/model \
	--train-data-path inputs/data.jsonl \
	--output-path outputs \
	--aim-model-id amd/your-model-id
```

Current execute mode assumptions:

- the CLI is running inside the execution container
- `/local_resources/basemodel`, `/local_resources/input`, and `/workdir` are already mounted
- generated config files can be written into `/configs`

For cluster-side testing of the aimfttk backend against the silogen base image, see `examples/silogen-cli-debug-job.yaml`.

## Supported Engine Args

Backend-specific settings are passed through `AIMFT_ENGINE_ARGS` as a JSON object. These args follow the same structure as the `engine_args` field in the recipe metadata and are merged with the recipe defaults.
```
