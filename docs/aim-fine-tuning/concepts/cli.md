<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Command-line interface

The `aim_fine_tune` command-line interface (CLI) prepares and runs fine-tuning
jobs from local model and dataset inputs. It provides a common workflow over
the engine-specific fine-tuning engines.

## Commands

The CLI has three commands:

- `list-recipes` lists the packaged recipes and can filter to recipes compatible
  with the available accelerators.
- `train` selects a recipe, validates the request, and creates the engine
  launch workspace.

Use the command help to see the available inputs:

```bash
uv run aim_fine_tune --help
uv run aim_fine_tune train --help
```

## Training workflow

The `train` command requires a model directory, a training dataset file, an
output directory, and an AIM model identifier. It uses those inputs together
with the available accelerators to select a compatible [recipe](recipes.md).

Dry-run is the default mode. It renders the launch workspace, including the
normalized training request, launch specification, and engine configuration,
without starting training. Pass `--execute` only when the CLI is running in the
target execution environment and its required paths are mounted. Invoke the
installed `aim_fine_tune` console script rather than `uv run`, so engine
`python3` stays first on `PATH`.

Execute mode captures the training process output, relays it to stdout, and
writes `runtime/train-report.json` from the last numbered `checkpoints/checkpoint-*/trainer_state.json`
when present, otherwise from `train_metrics` events in `runtime/train.stdout.log`
(losses, accelerators detected at startup, duration, and throughput). Pass
`--json-logs` to emit JSON Lines (`train_metrics` from the engine and `log`
from the runtime) while retaining the complete engine output in
`runtime/train.stdout.log` and `runtime/train.stderr.log`.

```bash
aim_fine_tune train \
  --model-path inputs/model \
  --train-data-path inputs/data.jsonl \
  --output-path outputs \
  --aim-model-id vendor/model \
  --execute
```

## Configuration sources

Inputs can be supplied as command-line flags or environment variables. For
example, `--model-path` corresponds to `AIMFT_MODEL_PATH`, and `--recipe-id`
corresponds to `AIMFT_RECIPE_ID`. Command-line flags take precedence over
environment variables, while recipe defaults provide the remaining training
settings.

Use `--recipe-id` to select a specific recipe. Use `--extra-recipes` or
`AIMFT_EXTRA_RECIPES` to add a directory of your own recipes to the built-in ones, or `--recipe-path` or
`AIMFT_RECIPE_PATH` to use one recipe file directly. Engine-specific configuration belongs either in a recipe or in the `AIMFT_ENGINE_ARGS` JSON object. See [engines](engines.md) for details on the engine boundary.
