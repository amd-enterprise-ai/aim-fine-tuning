<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Recipe schema

A recipe is a YAML document with common fields used by the CLI and an
engine-specific `engine_args` mapping consumed by the selected engine.

## Common structure

The common top-level sections are:

- `metadata`: recipe identity and compatibility requirements.
- `defaults`: training defaults passed through the normalized training request.
  The fields and values may be training-method-specific; the recipe's explicit
  `metadata.method` determines how they are interpreted.
- `engine_args`: engine-specific configuration. Its fields are defined by the
  selected engine reference.

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

## Engine-specific fields

The CLI validates and forwards `engine_args` to the adapter selected by
`metadata.engine`. Avoid using engine-specific fields that are equivalent to the default parameters available in the `defaults` section.

See the [AIM Fine-tuning Toolkit engine reference](engines/aimfttk.md) for
the currently supported `aimfttk` fields.
