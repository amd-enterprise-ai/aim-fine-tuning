<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Recipes

Each AIMFT Image includes a set of verified fine-tuning recipes. A recipe binds a model, method, backend, accelerator count, training defaults, and backend-specific settings into a starting point for a job. A recipe is selected automatically given certain inputs from the user.
The recipes that are included have had their compatibility verified for the base engine image. The set is part of the AIMFT Image, so different image tags can carry different recipes.

The running AIMFT Image selects a recipe from the environment:

- `AIM_MODEL_ID` selects the model.
- `AIMFT_USE_ADAPTER` selects adapter or full-parameter training.
- `AIMFT_BACKEND` and `AIMFT_METHOD` select the backend and training method. The `aimfttk` AIMFT Image defaults these to `aimfttk` and `SFT`.
- The accelerators visible to the container decide which packaged recipes are eligible.
- `AIMFT_RECIPE_ID` can be used to force the selection of a particular recipe, overriding the automatic process.


## Listing Recipes
Use `list-recipes` to inspect the recipes packaged in the AIMFT Image:

```bash
aim_fine_tune list-recipes [OPTIONS]
```

By default, the command outputs JSON. Use `--output-format pretty` for a human-readable listing.

| Option | Description |
| --- | --- |
| `--output-format [json\|pretty]` | Select the output format. Defaults to `json`. |
| `--compatible-only` | Show only recipes compatible with the detected accelerators. |
| `--aim-id TEXT` | Show only recipes matching the specified AIM model ID. |
| `--extra-recipes DIRECTORY` | Include recipes from an additional directory. The directory must contain its own `aimft-recipe-compatibility.yaml`. |

```bash
aim_fine_tune list-recipes --compatible-only
```

JSON output contains a `recipes` array. Each recipe includes its recipe ID, task, model IDs, training defaults, engine, accelerator requirements, compatible accelerator products, and an `isCompatible` field.

`--compatible-only` checks the detected accelerator hardware against the recipe's accelerator count and compatibility list. It does not select a recipe for training; recipe selection also considers the requested backend, method, model, and adapter mode.

Extra recipes are added to the built-in catalogue. If an extra recipe has the same `recipeId` as a built-in recipe, it replaces the built-in recipe.

## Adding recipes
AIMFT Images install their verified recipes under `/opt/aim_fine_tune/builtin_recipes`, together with the accelerator compatibility listing `aimft-recipe-compatibility.yaml`. Users can mount custom recipes. See the guide [here](guides/mount-recipes.md).

## Recipe structure

Each recipe contains three top-level mappings. `metadata` and `defaults` are common sections. `engine_args` is reserved for backend-specific configuration.

- `metadata` identifies the recipe and the request it can serve.
- `defaults` provides hyperparameter defaults.
- `engine_args` supplies backend-specific configuration.

The common fields are documented in the [recipe schema reference](reference/recipe-schema.md). Backend reference pages document the accepted fields inside `engine_args`.

`metadata` includes a stable `recipeId`, `backend`, model ID, AIM ID, method, adapter mode, and accelerator count. Accelerator compatibility is recorded in `aimft-recipe-compatibility.yaml`, not inside the recipe.

```yaml
metadata:
  recipeId: aimfttk/sft/google-gemma-3-27b-it/lora/1gpu/default
  backend: aimfttk
  modelId: google/gemma-3-27b-it
  aimId: google/gemma-3-27b-it
  method: SFT
  isAdapter: true
  acceleratorCount: 1

defaults:
  epochs: 1
  learning_rate: 0.000005
  per_accelerator_batch_size: 1

engine_args:
  distributed_type: auto-single-process
```
