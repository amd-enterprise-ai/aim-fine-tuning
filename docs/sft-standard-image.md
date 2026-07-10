<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Fine-tuning Standardized SFT Image Interface

**Date:** 2026-06-26

**Status:** Alpha

**Contact:** Aku Rouhe / aku.rouhe@amd.com

---

## Common Image Runtime

Each AIM Fine-tuning image uses a standard entrypoint that reads the user inputs from environment variables, and inspects the compute resources available to it, reads the set of Recipes available in the container, and chooses the best recipe according to the Recipe Preference Rules. Then, the runtime calls the fine-tuning engine to execute the chosen recipe with the user provided inputs and parameters.

### Recipes
Recipes are specific to a model ID, a task, and a fine-tuning engine. Recipes are compatible with a set of hardware. Recipes can produce an adapter or a full-parameter model.

User provided parameters are matched against recipes. For fine-tuning to proceed, a recipe matching the parameters and the environment must be found. Recipes can be selected by manual override, but the selected recipe should also match on all dimensions.

The recipe provides a set of default parameters. All the user needs to provide is the training data and the model weights files. If users overrides any hyperparameters, guarantees may not hold and the workload may for instance run out of memory on the accelerator.

### Recipe Listing Entrypoint

When the image is run with the commandline argument `list-recipes`, the container runtime does not run fine-tuning, but instead simply lists the recipes available in the image. The default output format is JSON. A human-readable pretty-printed format is available via `--output-format pretty`.

The machine readable JSON format output has the following schema:

```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["recipes"],
    "properties": {
        "recipes": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["recipeId", "status", "task", "aimId", "modelId", "defaults", "metadata"],
                "properties": {
                    "recipeId": {
                        "type": "string"
                    },
                    "status": {
                        "enum": ["preview", "validated"]
                    },
                    "task": {
                        "type": "string",
                        "enum": ["SFT"]
                    },
                    "aimId": {
                        "type": "string"
                    },
                    "modelId": {
                        "type": "string"
                    },
                    "defaults": {
                        "type": "object",
                        "description": "Recipe default hyperparameter values. Only fields with a recipe-defined default are present.",
                        "properties": {
                            "epochs": { "type": "integer" },
                            "seed": { "type": "integer" },
                            "evaluationInterval": { "type": "integer" },
                            "loggingInterval": { "type": "integer" },
                            "saveInterval": { "type": "integer" },
                            "learningRate": { "type": "number" },
                            "learningRateWarmupSteps": { "type": "integer" },
                            "perAcceleratorBatchSize": { "type": "integer" },
                            "gradientAccumulation": { "type": "integer" },
                            "loraAlpha": { "type": "integer" },
                            "loraDropout": { "type": "number", "minimum": 0, "maximum": 1 },
                            "loraRank": { "type": "integer" },
                            "maxContextLength": { "type": "integer" }
                        },
                        "additionalProperties": false
                    },
                    "metadata": {
                        "type": "object",
                        "required": ["compatibleAccelerators", "acceleratorCount", "isAdapter", "engine"],
                        "properties": {
                            "compatibleAccelerators": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                }
                            },
                            "acceleratorCount": {
                                "type": "integer"
                            },
                            "isAdapter": {
                                "type": "boolean"
                            },
                            "engine": {
                                "type": "string"
                            }
                        },
                        "additionalProperties": true
                    }
                },
                "additionalProperties": true
            }
        }
    },
    "additionalProperties": true
}
```

## SFT Data Format
The SFT Data Format is a JSONL encoded text file consisting lines that look like the following example:
```
{"messages": [{"role": "<system/user/assistant>", "content": "<first message content here>"},{"role": "<system/user/assistant>", "content": "<second message content here>"},...]}
```
Additional fields are allowed, for instance metadata relating to the data point.

The data lines should conform to the following JSON Schema:
```json
{
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["messages"],
    "properties": {
        "messages": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["role", "content"],
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["system", "user", "assistant"]
                    },
                    "content": {
                        "type": "string"
                    }
                },
                "additionalProperties": false
            }
        }
    },
    "additionalProperties": true
}
```

## Model Format

The AIM Fine-tuning images expect the input model to be provided in a Hugging Face transformers library compatible format. Additionally, the model's canonical `AIM_MODEL_ID` must be provided.

## Output Format

The AIM Fine-tuning images produce a set of runtime outputs, including the resolved parameters.

The AIM Fine-tuning images produce intermediate outputs in the finetuning engine's internal format, which is not standardized. If not doing adapter training, final full-parameter model is provided in a Hugging Face transformers library compatible format. If doing adapter training, the final adapter is provided in a Hugging Face PEFT format.

The AIM Fine-tuning images may generate configuration files to be passed on to the underlying fine-tuning engine. The configuration files are in the engine's native format.

The AIM Fine-tuning images may generate intermediate converted versions of the model or the data in a staging directory in the output_path.

The logs of the fine-tuning execution do not follow a standard format, each engine will have its own format. The logs are produced in the STDOUT stream, with some messages potentially in the STDERR stream.

### Output paths relative to AIMFT_OUTPUT_PATH

The runtime outputs are created in `${AIMFT_OUTPUT_PATH}/runtime`:
- `train-request.json` contains the training request metadata and the resolved parameters.
- `launch-spec.json` contains the training execution info, such as locations.
- Additionally, a `run.sh` may be included, containing a command that will execute the training run.

The staging outputs are created in `${AIMFT_OUTPUT_PATH}/staging`.

The intermediate outputs are created in `${AIMFT_OUTPUT_PATH}/checkpoints`.

The final full model is created in `${AIMFT_OUTPUT_PATH}/checkpoint-final`. The final adapter is created in `${AIMFT_OUTPUT_PATH}/checkpoint-final-adapter`.

The configuration files are created in `${AIMFT_OUTPUT_PATH}/configs`.

## Environment Variables

The environment variables use a common prefix of `AIMFT_` to avoid overloading variable names and to indicate which variables correspond to AIM Fine-tuning parameters. Certain other environment variables such as `HF_TOKEN` might be used by AIM Fine-tuning, but derive from other sources and thus do not use the prefix.

### Environment Variable Value Encoding

The values of the environment variables are encoded in JSON. The values can any of the basic types representable in JSON: *string*, *integer*, *float*, *object* (i.e. dictionary mapping), *array* (i.e. list), *boolean*, *null*. Additionally, values may be unset.

### Table of SFT Image Environment Variables

| **Key** | **Type** | **Default value / Required** | **Behaviour** |
|--|--|--|--|
| **Required** ||| *These variables must be set.* |
| `AIMFT_TRAIN_DATA_PATH` | *string* | **Required** | Path inside the container to the training data file, in the SFT Data Format. |
| `AIMFT_MODEL_PATH` | *string* | **Required** | Path inside the container to the directory that contains the model, in the Hugging Face format. |
| `AIM_MODEL_ID` | *string* | **Required** | AIM ModelID, the canonical identifier of the model weight set. |
| `AIMFT_OUTPUT_PATH` | *string* | **Required** | Path inside the container to the directory where outputs are produced. |
|||||
| **Optional** ||| *These additional variables may be set, but are not required. If not set, they take a default value.* |
| `AIMFT_RECIPE_ID` | *string\|unset* | *unset* | If set, manually override recipe selection and attempt to use the recipe corresponding to this ID. |
| `AIMFT_METHOD` | *string* | *"SFT"* | The training algorithm or method to use. In SFT images this is set to "SFT" by default. |
| `AIMFT_USE_ADAPTER` | *string* | *False* | If True, chooses an adapter recipe (typically LoRA). If False, chooses a full model recipe. |
| `AIMFT_VALIDATION_DATA_PATH` | *string\|null* | *null* | Path inside the container to the validation data in the SFT Data Format. A null value indicates no separate validation data file is used and validation is not done. |
| `AIMFT_EPOCHS` | *integer* | *1* | Sets the number of times to iterate over the training data. |
| `AIMFT_SEED` | *integer* | *42* | Sets the random seed. |
| `AIMFT_EVALUATION_INTERVAL` | *integer* | *1000* | Sets the number of optimization steps between evaluations. |
| `AIMFT_LOGGING_INTERVAL` | *integer* | *25* | Sets the number of optimization steps between log outputs. |
| `AIMFT_SAVE_INTERVAL` | *integer* | *500* | Sets the number of steps between intermediate checkpoint outputs. If set to -1, no checkpoints are saved. |
| `AIMFT_LEARNING_RATE` | *float* | *Recipe-dependent* | Sets the learning rate. |
| `AIMFT_LEARNING_RATE_WARMUP_STEPS` | *integer* | *Recipe-dependent* | Sets the number of initial training steps over which learning rate is ramped up. |
| `AIMFT_PER_ACCELERATOR_BATCH_SIZE` | *integer* | *Recipe-dependent* | Sets the batch size per accelerator. Total batch size is `AIMFT_PER_ACCELERATOR_BATCH_SIZE` \* `AIMFT_GRADIENT_ACCUMULATION` \* `The level of data parallelism`. The level of data parallelism is defined by the type of distributed setup, and depends on the recipe. |
| `AIMFT_GRADIENT_ACCUMULATION` | *integer* | *Recipe-dependent* | Sets the number of batches over which gradients are accumulated before taking an optimizer step. |
| `AIMFT_LORA_ALPHA` | *integer* | *Recipe-dependent* | Sets the alpha parameter of the LoRA Adapter. |
| `AIMFT_LORA_DROPOUT` | *float* | *Recipe-dependent; 0.0 if unset and LoRA is enabled* | Sets the dropout probability of the LoRA Adapter. |
| `AIMFT_LORA_RANK` | *integer* | *Recipe-dependent* | Sets the rank parameter of the LoRA Adapter. |
| `AIMFT_MAX_CONTEXT_LENGTH` | *integer* | *Recipe-dependent* | Sets the maximum context length in tokens. Examples longer than this are filtered or truncated. |
