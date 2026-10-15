<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Supported models

The packaged recipes currently support the following model, training method,
and engine combinations.

A recipe is a packaged preset of training parameters for a specific model,
training method, engine, and hardware configuration. Its compatibility rules
determine when it can be selected, and its default parameters fill in values
that are not provided in the training request.

## Supervised fine-tuning (SFT)

### `aimfttk` engine

The `aimfttk` engine provides the following SFT recipes:

| Model | Model ID | Training method | engine | Training mode | Recipe ID |
| --- | --- | --- | --- | --- | --- |
| Gemma 3 27B IT | `google/gemma-3-27b-it` | SFT | `aimfttk` | LoRA, 1 GPU | `aimfttk/sft/google-gemma-3-27b-it/lora/1gpu/default` |
| Llama 3.2 1B Instruct | `meta-llama/Llama-3.2-1B-Instruct` | SFT | `aimfttk` | Full parameter, 1 GPU | `aimfttk/sft/meta-llama-llama-3.2-1b-instruct/full/1gpu/default` |
| Llama 3.2 3B Instruct | `meta-llama/Llama-3.2-3B-Instruct` | SFT | `aimfttk` | Full parameter, 1 GPU | `aimfttk/sft/meta-llama-llama-3.2-3b-instruct/full/1gpu/default` |
| Mistral Small 24B Instruct 2501 | `mistralai/Mistral-Small-24B-Instruct-2501` | SFT | `aimfttk` | LoRA, 1 GPU | `aimfttk/sft/mistralai-mistral-small-24b-instruct-2501/lora/1gpu/default` |
| Qwen3 32B | `Qwen/Qwen3-32B` | SFT | `aimfttk` | LoRA, 1 GPU | `aimfttk/sft/qwen-qwen3-32b/lora/1gpu/default` |

## Select a supported model

Pass the model ID when running a job. The CLI uses it, together with the
available accelerator and requested training mode, to select a compatible
recipe:

```bash
aim_fine_tune train \
  --model-path /model \
  --train-data-path /data/train.jsonl \
  --output-path /output \
  --aim-model-id google/gemma-3-27b-it
```

To inspect the recipes included in the image, or to see only recipes compatible
with the current accelerator setup, run:

```bash
aim_fine_tune list-recipes # List all recipes
aim_fine_tune list-recipes --compatible-only
aim_fine_tune list-recipes --aim-id google/gemma-3-27b-it
```

Recipe compatibility and defaults are defined in the packaged recipe files.
See [Recipes](concepts/recipes.md) for selection behavior and the [recipe
schema reference](reference/recipe-schema.md) for the metadata fields.
