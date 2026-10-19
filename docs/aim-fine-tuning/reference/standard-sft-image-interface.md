<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->


# Table of AIMFT SFT Environment Variables

| **Key** | **Type** | **Default value / Required** | **Behaviour** |
|--|--|--|--|
| **Required** ||| *These variables must be set.* |
| `AIMFT_TRAIN_DATA_PATH` | *string* | **Required** | Path inside the container to the training data file, in the SFT data format. |
| `AIMFT_MODEL_PATH` | *string* | **Required** | Path inside the container to the directory that contains the model, in the Hugging Face format. |
| `AIM_MODEL_ID` | *string* | **Required** | AIM model ID, the canonical identifier of the model weight set. |
| `AIMFT_OUTPUT_PATH` | *string* | **Required** | Path inside the container to the directory where outputs are produced. |
|||||
| **Optional** ||| *These additional variables may be set. If unset, they take a default value.* |
| `AIMFT_BACKEND` | *string* | *"aimfttk"* | Backend that launches the job. It must match `metadata.backend` on the selected recipe. |
| `AIMFT_METHOD` | *string* | *"SFT"* | Training method. AIMFT Images default to `SFT`. |
| `AIMFT_USE_ADAPTER` | *boolean* | *false* | When true, selects an adapter recipe (typically LoRA). When false, selects a full-parameter recipe. |
| `AIMFT_RECIPE_ID` | *string\|unset* | *unset* | Pin catalogue selection to the recipe with this ID. The recipe must still match the request and the detected accelerators. |
| `AIMFT_RECIPE_PATH` | *string\|unset* | *unset* | Use this recipe file directly. Skips the catalogue and accelerator compatibility checks. The file must still match the model, method, backend, accelerator count, and adapter mode. |
| `AIMFT_EXTRA_RECIPES` | *string\|unset* | *unset* | Directory of extra recipes with its own `aimft-recipe-compatibility.yaml`. An extra recipe with the same ID replaces the built-in recipe. |
| `AIMFT_ENGINE_ARGS` | *object\|unset* | *unset* | JSON object merged over the recipe `engine_args`. |
| `AIMFT_VALIDATION_DATA_PATH` | *string\|null* | *null* | Path inside the container to the validation data in the SFT data format. Unset means validation is not run. |
| `AIMFT_JSON_LOGS` | *boolean* | *false* | When true, stdout is JSON Lines (`train_metrics` and runtime `log` events). `runtime/train.stdout.log` and `runtime/train.stderr.log` still capture the full engine output. |
| `AIMFT_EPOCHS` | *integer* | *1* | Number of times to iterate over the training data. |
| `AIMFT_SEED` | *integer* | *42* | Random seed. |
| `AIMFT_EVALUATION_INTERVAL` | *integer* | *1000* | Optimization steps between evaluations. |
| `AIMFT_LOGGING_INTERVAL` | *integer* | *25* | Optimization steps between log outputs. |
| `AIMFT_SAVE_INTERVAL` | *integer* | *500* | Steps between intermediate checkpoint outputs. `-1` saves no intermediate checkpoints. |
| `AIMFT_LEARNING_RATE` | *float* | *Recipe-dependent* | Learning rate. |
| `AIMFT_LEARNING_RATE_WARMUP_STEPS` | *integer* | *Recipe-dependent* | Initial training steps over which the learning rate is ramped up. |
| `AIMFT_PER_ACCELERATOR_BATCH_SIZE` | *integer* | *Recipe-dependent* | Batch size per accelerator. Total batch size is `AIMFT_PER_ACCELERATOR_BATCH_SIZE` times `AIMFT_GRADIENT_ACCUMULATION` times the data-parallel width. The data-parallel width depends on the distributed setup in the recipe. |
| `AIMFT_GRADIENT_ACCUMULATION` | *integer* | *Recipe-dependent* | Batches over which gradients are accumulated before an optimizer step. |
| `AIMFT_LORA_ALPHA` | *integer* | *Recipe-dependent* | Alpha parameter of the LoRA adapter. |
| `AIMFT_LORA_DROPOUT` | *float* | *Recipe-dependent; 0.0 when unset and LoRA is enabled* | Dropout probability of the LoRA adapter. |
| `AIMFT_LORA_RANK` | *integer* | *Recipe-dependent; 64 when unset and LoRA is enabled* | Rank of the LoRA adapter. |
| `AIMFT_MAX_CONTEXT_LENGTH` | *integer* | *Recipe-dependent* | Maximum context length in tokens. Longer examples are truncated. |
