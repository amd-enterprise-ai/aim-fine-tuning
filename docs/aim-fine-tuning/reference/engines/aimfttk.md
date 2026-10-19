<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Fine-tuning Toolkit (`aimfttk`)

`aimfttk` is the AIM Fine-tuning Toolkit backend. It prepares the engine configuration and launch script for SFT training requests.

The default backend is `aimfttk`. Set another backend with `AIMFT_BACKEND`. That value must match `metadata.backend` on the selected recipe.

The backend keeps common launch behavior separate from method-specific config generation. SFT has an AIMFTTK method integration, and recipes are selected using both the backend and `metadata.method`.

## AIMFT Image

The published `aimfttk` AIMFT Image is [`amdenterpriseai/aimft-aimfttk`](https://hub.docker.com/r/amdenterpriseai/aimft-aimfttk) on Docker Hub. `aimft-aimfttk` is the image name. `aimfttk` is the backend name used by recipes and engine configuration.

## Engine options

Engine arguments are JSON objects. Recipe-provided `engine_args` form the base configuration. Request-level engine arguments from `AIMFT_ENGINE_ARGS` are merged on top, with nested objects merged recursively. Unknown top-level keys are rejected. Pass these settings in `AIMFT_ENGINE_ARGS` or the recipe.

The common recipe fields are documented in the [recipe schema reference](../recipe-schema.md).

Supported options are:

- `distributed_type`: `auto-deepspeed-stage1`, `auto-deepspeed-stage3`, `auto-ddp`, `auto-single-process`, or a path to an existing Accelerate configuration file. When the field is omitted, one accelerator uses `auto-single-process` and more than one accelerator uses `auto-deepspeed-stage1`.
- `merge_adapter`: whether to merge the adapter into the model after training. The default is false.
- `finetuning_config`: additional or overriding AIMFTTK fine-tuning settings, such as `data_conf`, `training_args`, `peft_conf`, and `run_conf`.

Generated configuration is written under the output directory:

- `configs/accelerate_config.yaml`
- `configs/finetuning_config.yaml`
- `configs/entrypoint.sh`
