<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Fine-tuning

AIM Fine-tuning offers tools and starting points to make fine-tuning easy to use, fast, and guaranteed to work in the context of [AIMs](https://enterprise-ai.docs.amd.com/en/latest/aims/overview.html).

Fine-tuning is the act of adapting or adjusting an already trained, capable AI model to suit a new purpose or learn a new ability.
It is an inherently experiment-based, iterative process involving three key elements:
- The model, which should have the capacity and base capabilities necessary to learn what we intend it to learn.
- The data, which should contain clear examples of the kind of behaviour that we want to teach the model.
- The learning process parameters, the variables (hyperparameters) that control how the model learns from the data.
Fine-tuning involves finding suitable choices for each of these three elements.

### Supervised Fine-tuning (SFT)
Supervised Fine-tuning (SFT) is a robust, straight-forward fine-tuning method, typically the first go-to approach for any task.

### Full parameter training and Adapter training
Fine-tuning can be done with full parameter updates, which is computationally heavier, but allows the full capacity of the model and the training to be used. A computationally lighter alternative is Low-Rank Adapter (LoRA) training, which can with the right training hyperparameters lead to very good results. LoRA training produces a neural adapter, and an added benefit is that multiple adapters can be served by a single basemodel deployment.
