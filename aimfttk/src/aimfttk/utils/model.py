# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

""" "Model utilities"""

from transformers import PretrainedConfig

from aimfttk.config.experiment import ExperimentConfig
from aimfttk.config.hf_integration import NO_PEFT
from aimfttk.data import subsetup_tokenizer
from aimfttk.data.chat_templates import get_chat_template


def is_quantized(model):
    """Checks if the model is quantized

    Check based on
    https://github.com/huggingface/peft/blob/a9425d1409379ccedd89acc2dc834ed73961b96b/src/peft/utils/other.py#L93-L95
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    is_aqlm_quantized = getattr(model, "quantization_method", None) == "aqlm"
    return loaded_in_kbit or is_gptq_quantized or is_aqlm_quantized


def is_kbit(model):
    """Checks if the model is quantized with bitsandbytes kbit training methods specifically

    Check based on
    https://github.com/huggingface/peft/blob/a9425d1409379ccedd89acc2dc834ed73961b96b/src/peft/utils/other.py#L93-L95
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    return loaded_in_kbit


def should_remove_non_lora_layers(exp_conf: ExperimentConfig):
    """Checks if non-LoRA layers should be removed from the model"""
    # Check if the extra layers in the adapter model be removed or not.
    # We handle extra layers as follows:
    # - Scenario 1: We want to remove the extra layers when HuggingFac-
    # e's save function's default rules add them.
    # - Scenario 2: If the extra layers are added by "us", then we sho-
    # uld not remove them. For example, we grow embeddings
    # (peft_extra_modules_to_save is not empty then), or some modules
    # are specified in the config.yaml
    # (exp_conf.peft_conf.peft_kwargs.get("modules_to_save", []) is no-
    # t empty).
    # Most adapters will be part of Scenario 1, so the extra
    # layers will be removed. But, every now and then, we want to ensu-
    # re we don't lose the layers we mainly added. For this case, we c-
    # reate a separate flyte task to massage the adapter model into th-
    # e correct format for vLLM.

    # We simply run the chat template and tokenizer setup because they are quite
    # fast and cheap to do.
    chat_template = get_chat_template(exp_conf.run_conf.tokenizer_args.chat_template_name)
    tokenizer, adds_new_tokens = subsetup_tokenizer(
        tokenizer_name_or_path=exp_conf.run_conf.tokenizer,
        chat_template=chat_template,
        missing_pad_token_strategy=exp_conf.run_conf.tokenizer_args.missing_pad_token_strategy,
        tokenizer_args=exp_conf.run_conf.tokenizer_args,
        overwrite_chat_template=True,
    )
    grows_embeddings = False
    if adds_new_tokens:
        model_config = PretrainedConfig.from_pretrained(exp_conf.run_conf.model)
        if model_config.vocab_size < len(tokenizer):
            grows_embeddings = True
    return not (
        exp_conf.peft_conf.peft_type == NO_PEFT
        or grows_embeddings
        or (hasattr(exp_conf.peft_conf, "peft_kwargs") and exp_conf.peft_conf.peft_kwargs.get("modules_to_save", []))
    )
