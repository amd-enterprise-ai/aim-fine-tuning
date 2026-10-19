<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Fine-tuning

AIM Fine-tuning makes fine-tuning LLMs easy and reliable with AIM Fine-tuning Standard Images, AIMFT Images for short, and validated training recipes. An AIMFT Image packages the `aim_fine_tune` runtime, a training engine, and the recipes verified for that image. This lets users focus on their data and what they want to accomplish, instead of spending days installing libraries and learning engine implementation details.

For a step-by-step example of running a training job, see the [Quickstart](guides/quickstart.md).

Fine-tuning adapts an already trained AI model to suit a new purpose or learn a new capability.
It is an inherently experimental, iterative process involving three key elements:
- The model, which should have the capabilities necessary to learn what we want it to learn.
- The data, which should contain clear examples of the behaviour we want to teach the model.
- The learning process parameters, or hyperparameters, that control how the model learns from the data.
Fine-tuning involves finding suitable choices for each of these three elements.

## Recipes

The key to reliable training runs is the set of verified recipes that an AIMFT Image includes. [Recipes](#recipes) are model-specific training starting points.
