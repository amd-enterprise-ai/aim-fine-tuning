<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Recipes

A recipe is a YAML description of a supported fine-tuning configuration. It
binds a model, method, engine, accelerator requirement, training defaults, and
engine-specific settings into a tested starting point for a job.

Recipes are packaged under `/opt/aim_fine_tune/builtin_recipes` in [AIMFT images](https://hub.docker.com/r/amdenterpriseai/aimft-aimfttk/tags). To add your own,
mount a directory of recipes with its own `aimft-recipe-compatibility.yaml` and set `AIMFT_EXTRA_RECIPES` to it. An
extra recipe replaces a built-in recipe with the same recipe ID.

## Recipe structure

Each recipe contains three top-level mappings. `metadata` and `defaults` are
common recipe sections; `engine_args` is reserved for engine-specific
configuration.

- `metadata` identifies the recipe and its compatibility requirements.
- `defaults` provides runtime hyperparameter defaults.
- `engine_args` supplies engine-specific configuration.

The common recipe structure is documented in the [recipe schema reference](../reference/recipe-schema.md).
engine reference pages document the accepted fields inside `engine_args`.

The metadata includes a stable `recipeId`, engine, model ID, AIM ID, method,
adapter mode, accelerator count, and compatible accelerator IDs.

```yaml
metadata:
  recipeId: aimfttk/sft/google-gemma-3-27b-it/lora/1gpu/default
  engine: aimfttk
  modelId: google/gemma-3-27b-it
  aimId: google/gemma-3-27b-it
  method: SFT
  isAdapter: true
  acceleratorCount: 1
  compatibleAccelerators: ["74a1", "74a9"]

defaults:
  epochs: 1
  learning_rate: 0.000005
  per_accelerator_batch_size: 1

engine_args:
  distributed_type: auto-deepspeed-stage1
```

## Selection

By default, the CLI detects the available accelerators and chooses a recipe that
matches all of these dimensions:

- Requested engine.
- AIM model ID.
- Accelerator device IDs and requested accelerator count.
- Adapter or full-parameter mode, when it is specified.

When multiple recipes match, the one whose file path sorts first is selected. The CLI reports an error rather than silently selecting an incompatible
recipe.

## Inspect and pin recipes

List the available recipes, optionally filtering to those compatible with the
current environment:

```bash
uv run aim_fine_tune list-recipes
uv run aim_fine_tune list-recipes --compatible-only
uv run aim_fine_tune list-recipes --aim-id google/gemma-3-27b-it
```

Set `--recipe-id` or `AIMFT_RECIPE_ID` to pin one recipe. The pinned recipe must
still match the requested model, accelerator count, and hardware. To use a
recipe file without the catalog or hardware compatibility checks, set
`--recipe-path` or `AIMFT_RECIPE_PATH`; the model and accelerator count must still
match.
