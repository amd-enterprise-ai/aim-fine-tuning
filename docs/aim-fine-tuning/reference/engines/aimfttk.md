<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Fine-tuning Toolkit (`aimfttk`)

`aimfttk` is the engine for the AIM Fine-tuning Toolkit engine. It prepares
the engine configuration and launch script for SFT training requests.

Use the engine explicitly when rendering or executing a job:

```bash
uv run aim_fine_tune train --engine aimfttk
```

The engine name in a request must match the engine declared by the selected
recipe.

The backend keeps common launch behavior separate from method-specific config
generation. SFT has an AIMFTTK method integration, and recipes
are selected using both the backend and `metadata.method`.

## Container image

The published AIM Fine-tuning Toolkit image is
[`amdenterpriseai/aimft-aimfttk`](https://hub.docker.com/r/amdenterpriseai/aimft-aimfttk)
on Docker Hub. `aimft-aimfttk` is the image name; `aimfttk` is the engine name
used by the CLI, recipes, and engine configuration.

## Engine options

Engine arguments are JSON objects. Recipe-provided `engine_args` form the base
configuration; request-level engine arguments are merged on top, with nested
objects merged recursively. Keep engine-specific settings in `AIMFT_ENGINE_ARGS`
or the corresponding recipe rather than adding them as common CLI options.

The common recipe fields are documented in the [recipe schema reference](../recipe-schema.md).

Supported options are:

- `distributed_type`: an automatic distribution mode or a path to an existing
  Accelerate configuration file.
- `merge_adapter`: whether to merge the adapter into the model after training.
- `finetuning_config`: additional or overriding AIMFTTK fine-tuning settings.
