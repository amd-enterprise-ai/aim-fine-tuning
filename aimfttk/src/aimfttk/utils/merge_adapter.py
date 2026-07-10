# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Merge adapters and save a single model"""

from logging import getLogger

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = getLogger(__name__)


def merge_and_save(model_name_or_path, tokenizer_name_or_path, adapter_name_or_path, outpath, device_map="auto"):
    """Loads basemodel and adapter, merges, then saves the merged model and the tokenizer at the output path"""

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        torch_dtype="auto",
    )
    logger.info(f"Loaded base model at {model_name_or_path}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    logger.info(f"Loaded tokenizer at {tokenizer_name_or_path}")

    model = PeftModel.from_pretrained(model, adapter_name_or_path, device_map=device_map)
    logger.info(f"Loaded adapter at {adapter_name_or_path}")

    model = model.merge_and_unload(safe_merge=True)  # safe_merge protects from numerical issues during merge
    logger.info("Merged adapter")

    model.save_pretrained(outpath)
    logger.info(f"Saved model at {outpath}")

    tokenizer.save_pretrained(outpath)
    logger.info(f"Saved tokenizer at {outpath}")
