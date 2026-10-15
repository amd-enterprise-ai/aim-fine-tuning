<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Training methods

A training method defines how a model is adapted using a training dataset and
the associated optimization parameters. Different methods use different kinds
of training examples and optimize the model in different ways.

## Supervised fine-tuning (SFT)

Supervised fine-tuning teaches a pretrained model by showing it examples of
the desired input and response behavior. The model learns to produce responses
that better match those examples.

SFT can be run in two common ways:

- **Full-parameter fine-tuning** updates the model's parameters and produces a
	new trained model.
- **Adapter fine-tuning**, such as LoRA, trains a smaller set of additional
	parameters and produces an adapter that is applied to the original model.

The available training method, model, and training mode are selected through a
compatible recipe. In a recipe, the `metadata.method` field identifies the
training method:

```yaml
metadata:
  method: SFT
```

See [Supported models](../supported-models.md) for the currently available
combinations.
