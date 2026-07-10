# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
This script is used to remove non-lora layers from a model. The PEFT library saves the base
embedding layers as well when save() is called. Currenlty, this is not supported in vllm. If
you have not trained with an expanded vocabulary and your base embeddings have not updated -
you can just remove the base layer weights. The solution and the documentaiton is inspired by
https://github.com/vllm-project/vllm/issues/3404#issuecomment-2028878893
"""

import os
from logging import getLogger

import safetensors.torch
from transformers.utils import ADAPTER_SAFE_WEIGHTS_NAME

logger = getLogger(__name__)


def remove_non_lora_layers(model_checkpoint_folder: str):
    """
    Remove non-lora layers from a model and save it to a new path.

    Note: This script assumes that other parts e.g. embeddings, lm_head have not been updated. Thu-
    s, the layers removed are indeed the same as the base model checkpoint. An example where is as-
    sumption is broken when the vocabulary is extended and thus, the embedding layer is updated. In
    such cases, the embeddings should not be removed. In future, we would like to check if any of
    checkpoint's constituent layers have been updated and remove only the layers that have not been
    updated.
    """
    model_path = f"{model_checkpoint_folder}/{ADAPTER_SAFE_WEIGHTS_NAME}"
    old_model_path = f"{model_checkpoint_folder}/old_with_extra_layers_{ADAPTER_SAFE_WEIGHTS_NAME}"

    if os.path.exists(old_model_path):
        logger.info("Model already has the extra layers removed. Exiting.")
        return

    tensors = safetensors.torch.load_file(model_path)

    nonlora_keys = []
    for k in list(tensors.keys()):
        if ".lora_" not in k:
            nonlora_keys.append(k)

    for k in nonlora_keys:
        del tensors[k]

    logger.info(f"Deleted adapter keys {nonlora_keys}")
    os.rename(model_path, old_model_path)
    safetensors.torch.save_file(tensors, model_path)
