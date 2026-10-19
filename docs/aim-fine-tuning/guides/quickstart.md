<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Quickstart

Run a fine-tuning job locally using an AIMFT Image. The AIMFT Image includes the training recipes, the training engine, and the accelerator software required to run the job. Configure the job with environment variables.

This example uses an AIMFT Image that has the `aimfttk` training engine installed to run a supervised fine-tuning (SFT) job on the Gemma 3 27B IT model with LoRA training.

## Prerequisites

You need:

- A Linux host with ROCm and Docker installed.
- Access to an AMD GPU through `/dev/kfd` and `/dev/dri`.
- A model and JSONL training dataset stored on the host.
- An AIMFT Image, such as [`amdenterpriseai/aimft-aimfttk`](https://hub.docker.com/r/amdenterpriseai/aimft-aimfttk) from Docker Hub.

## Prepare local inputs

Use a directory layout like this example:

```text
training-job/
├── inputs/
│   ├── model/
│   └── data/
│       └── train.jsonl
└── outputs/
```

The model directory and dataset are mounted read-only. The output directory is writable so checkpoints and runtime files remain on the host.

## Format the dataset

The SFT dataset must be a JSONL file with one conversation per line. Each conversation contains a `messages` array with `role` and `content` fields:

```json
{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"What is fine-tuning?"},{"role":"assistant","content":"Fine-tuning adapts a pretrained model to a specific task or domain."}]}
{"messages":[{"role":"user","content":"Name one benefit of fine-tuning."},{"role":"assistant","content":"It can improve performance on a target task using a focused dataset."}]}
```

Conversations should use the roles expected by the selected model's chat template, typically `system`, `user`, and `assistant`.

You could use [SiloAI/AMD-eAIrs-SFT-Demo-Data](https://huggingface.co/datasets/SiloAI/AMD-eAIrs-SFT-Demo-Data) sft-formatted demo dataset for running initial quickstart tests.

## Start the job

From the `training-job` directory run the container

```bash
export AIMFT_IMAGE=amdenterpriseai/aimft-aimfttk:0.2.3-0.2.0-rocmdefault

mkdir -p outputs

docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --ipc=host \
  --mount type=bind,src="$PWD/inputs/model",dst=/model,readonly \
  --mount type=bind,src="$PWD/inputs/data",dst=/data,readonly \
  --mount type=bind,src="$PWD/outputs",dst=/output \
  -e AIMFT_MODEL_PATH=/model \
  -e AIMFT_TRAIN_DATA_PATH=/data/<your_dataset>.jsonl \
  -e AIMFT_OUTPUT_PATH=/output \
  -e AIM_MODEL_ID=google/gemma-3-27b-it \
  -e AIMFT_USE_ADAPTER=true \
  "$AIMFT_IMAGE"
```

Training logs are printed in the terminal running Docker. To keep the container available for a later log command, omit `--rm`, assign a name with `--name aimft-sft`, and run:

```bash
docker logs -f aimft-sft
```

## Find the results

When training finishes, inspect the host `outputs/` directory. It contains runtime metadata, the rendered engine configuration, checkpoints, and the final adapter or model. With adapter training enabled, the final adapter is written to `outputs/checkpoint-final-adapter`.

The [AIMFT Image interface](../reference/standard-sft-image-interface.md) documents the environment variables and output paths.

For Kubernetes execution, see [Run standalone fine-tuning on Kubernetes](standalone-finetuning.md).
