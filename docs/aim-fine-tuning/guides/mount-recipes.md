<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Mount your own recipes

An AIMFT Image selects a packaged recipe unless you mount one of your own. A single recipe file is used directly. A directory of recipes extends the catalogue that ships with the AIMFT Image. Recipe fields are described in [Recipes](../recipes.md).

## Use one recipe file directly

Mount a single recipe file and set `AIMFT_RECIPE_PATH` to its path inside the container. The AIMFT Image uses that file and skips the recipe catalogue and the accelerator compatibility checks. The file must still match the requested model, method, backend, accelerator count, and adapter mode.

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host \
  --mount type=bind,src="$PWD/inputs/model",dst=/model,readonly \
  --mount type=bind,src="$PWD/inputs/data",dst=/data,readonly \
  --mount type=bind,src="$PWD/outputs",dst=/output \
  --mount type=bind,src="$PWD/my-recipe.yaml",dst=/recipe/my-recipe.yaml,readonly \
  -e AIMFT_MODEL_PATH=/model \
  -e AIMFT_TRAIN_DATA_PATH=/data/train.jsonl \
  -e AIMFT_OUTPUT_PATH=/output \
  -e AIM_MODEL_ID=vendor/model \
  -e AIMFT_RECIPE_PATH=/recipe/my-recipe.yaml \
  "$AIMFT_IMAGE"
```

## Extend the catalogue

Mount a directory of recipe files and set `AIMFT_EXTRA_RECIPES` to that directory inside the container. The directory needs its own `aimft-recipe-compatibility.yaml` at the root. Extra recipes are selected with the built-in ones. An extra recipe whose `recipeId` matches a built-in recipe replaces that built-in recipe. A recipe that is missing from the compatibility file is never selected, because it is compatible with no accelerators.

```text
extra-recipes/
├── aimft-recipe-compatibility.yaml
└── my-org/
    └── my-model.yaml
```

`compatibleAccelerators` entries are accelerator product names, such as `MI300X`. A device ID is accepted when it maps to a known product. `recipeDigest` is optional. When it is set, it must equal the digest of that recipe file.

```yaml
recipes:
  aimfttk/sft/my-org-my-model/lora/1gpu/default:
    compatibleAccelerators:
      - MI300X
```

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host \
  --mount type=bind,src="$PWD/inputs/model",dst=/model,readonly \
  --mount type=bind,src="$PWD/inputs/data",dst=/data,readonly \
  --mount type=bind,src="$PWD/outputs",dst=/output \
  --mount type=bind,src="$PWD/extra-recipes",dst=/extra-recipes,readonly \
  -e AIMFT_MODEL_PATH=/model \
  -e AIMFT_TRAIN_DATA_PATH=/data/train.jsonl \
  -e AIMFT_OUTPUT_PATH=/output \
  -e AIM_MODEL_ID=vendor/model \
  -e AIMFT_EXTRA_RECIPES=/extra-recipes \
  "$AIMFT_IMAGE"
```
