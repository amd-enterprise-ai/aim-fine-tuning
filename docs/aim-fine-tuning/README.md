<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Fine-tuning

AIM Fine-tuning makes fine-tuning LLMs easy and reliable by providing a standard unified container interface and validated training recipes. This lets users focus on their data and what they want to accomplish, instead of spending days installing libraries and learning engine implementation details.

For a step-by-step example of running a training job, see the [Quickstart](quickstart.md).

Fine-tuning adapts an already trained AI model to suit a new purpose or learn a new capability.
It is an inherently experimental, iterative process involving three key elements:
- The model, which should have the capabilities necessary to learn what we want it to learn.
- The data, which should contain clear examples of the behaviour we want to teach the model.
- The learning process parameters, or hyperparameters, that control how the model learns from the data.
Fine-tuning involves finding suitable choices for each of these three elements.

## Supported training methods, engines, and models

### Training methods

- [Supervised fine-tuning (SFT)](concepts/training-methods.md)

### Training engines

- [`aimfttk`](reference/engines/aimfttk.md): AIM Fine-tuning Toolkit engine,
  available in the [`amdenterpriseai/aimft-aimfttk` Docker image](https://hub.docker.com/r/amdenterpriseai/aimft-aimfttk)

### Models

AIM Fine-tuning provides predefined recipes that specify training parameters for selected models and training engines. See the [Supported models](supported-models.md) page for the supported model, training method, engine, and recipe combinations.
