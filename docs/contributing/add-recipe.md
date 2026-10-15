<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Add a recipe

A recipe adds a tested model and hardware configuration to an existing engine.
It does not require a new engine adapter or image unless the engine behavior
itself is new.

## 1. Choose the engine and model

Confirm that the engine is registered and that its adapter supports the
requested training method. Use the engine name exactly as it appears in the
registry, for example `aimfttk`.

Choose a unique recipe ID that describes the engine, method, model, adapter
mode, accelerator count, and variant. The recipe ID is the value users pass to
`--recipe-id` or `AIMFT_RECIPE_ID`.

Packaged recipes currently live under:

```text
aim_fine_tune/recipes/<engine>/
```

## 2. Define the common fields

Create a YAML file with the common sections documented in the [recipe schema
reference](../aim-fine-tuning/reference/recipe-schema.md):

- `metadata.recipeId`
- `metadata.engine`
- `metadata.modelId` and `metadata.aimId`
- `metadata.method`
- `metadata.isAdapter`
- `metadata.acceleratorCount`
- `metadata.compatibleAccelerators`
- `metadata.sizing`, when deployment sizing information is required
- `defaults` for common training defaults

The metadata must describe the actual model, method, adapter mode, and hardware
requirements. Do not use a recipe to bypass engine validation or to encode
engine-specific fields in `metadata` or `defaults`.

## 3. Add engine-specific arguments

Put engine-specific configuration under `engine_args`. The adapter owns the
schema and validation for this mapping. For `aimfttk`, examples include
`distributed_type`, `merge_adapter`, and `finetuning_config`.

```yaml
engine_args:
  distributed_type: auto-deepspeed-stage1
  finetuning_config:
    training_args:
      bf16: true
```

Keep the recipe's engine arguments as the base configuration. Request-level
`AIMFT_ENGINE_ARGS` values are merged over them, including nested mappings.

## 4. Validate selection

List the recipe and verify its serialized metadata:

```bash
uv run --project aim_fine_tune aim_fine_tune list-recipes --extra-recipes recipes
uv run --project aim_fine_tune aim_fine_tune list-recipes --extra-recipes recipes --aim-id <aim-model-id>
```

The repository recipes have no compatibility file until evidence for them is
staged, so they are listed as incompatible. Render a dry run with the intended
model, dataset, output path, and accelerator count by passing the recipe file
directly:

```bash
uv run --project aim_fine_tune aim_fine_tune train \
  --engine <engine> \
  --recipe-path recipes/<engine>/<method>/<vendor>/<model>.yaml \
  --model-path inputs/model \
  --train-data-path inputs/data.jsonl \
  --output-path outputs \
  --aim-model-id <aim-model-id>
```

Check the generated `train-request.json`, `launch-spec.json`, and engine
configuration before executing a job.

Update the relevant engine reference page when the recipe introduces a new
supported engine option. Keep the [recipe schema reference](../aim-fine-tuning/reference/recipe-schema.md)
focused on fields shared by all engines.
