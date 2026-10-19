<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Recipe schema

A recipe is a YAML document with common fields read from the AIMFT Image environment and a backend-specific `engine_args` mapping consumed by the selected backend.

## Common structure

The common top-level sections are:

- `metadata`: recipe identity and the request it can serve.
- `defaults`: training defaults passed through the normalized training request. The fields and values may be training-method-specific. `metadata.method` determines how they are interpreted.
- `engine_args`: backend-specific configuration. Its fields are defined by the selected backend reference.

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

`metadata.backend` is the backend name, for example `aimfttk`. It must match `AIMFT_BACKEND`. The `aimfttk` AIMFT Image defaults that variable to `aimfttk`.

Accelerator compatibility is not a metadata field. It is listed by recipe ID in `aimft-recipe-compatibility.yaml` next to the recipes. See [Recipes](../recipes.md) for the recipe fields, and [Mount your own recipes](../guides/mount-recipes.md) for mounting a file or a directory.

## Backend-specific fields

The AIMFT Image validates and forwards `engine_args` to the adapter selected by `metadata.backend`. Keep values that already have a `defaults` field out of `engine_args`.
