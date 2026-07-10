# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""DPO Implementation
An Example of the DPO Config Format (a doctest so that it remains up-to-date):

>>> yamlstr='''
... data_conf:
...   training_data:
...     type: "CONCATENATION"
...     datasets:
...       - path: "/datasets/preference_data_v1/ultrafeedback_binarized_train.jsonl"
...   validation_data:
...     type: "CONCATENATION"
...     datasets:
...       - path: "/datasets/preference_data_v1/ultrafeedback_binarized_test.jsonl"
... batchsize_conf:
...   total_train_batch_size: 64
...   max_per_device_train_batch_size: 2
... training_args:
...   output_dir: "/experiments/Zephyr-replication/zephyr-7b-beta-dpo-lora_merged-sft_2.2"
...   bf16: true
...   gradient_checkpointing: true
...   gradient_checkpointing_kwargs: {"use_reentrant": false}
...   learning_rate: 5.0e-7
...   optim: rmsprop
...   logging_steps: 10
...   logging_strategy: "steps"
...   lr_scheduler_type: "cosine"
...   max_steps: -1
...   num_train_epochs: 3
...   overwrite_output_dir: true
...   group_by_length: True  # This should use less padding
...   push_to_hub: False
...   report_to: ["none"]
...   save_strategy: "steps"
...   save_steps: 20
...   eval_strategy: "epoch"
...   seed: 42
...   warmup_ratio: 0.1
...   beta: 0.1
...   max_length: 1024
...   max_prompt_length: 512
... peft_conf:
...   peft_type: "LORA"
...   task_type: "CAUSAL_LM"
...   peft_kwargs:
...     r: 64
...     lora_alpha: 16
...     lora_dropout: 0.1
...     target_modules:
...       - q_proj
...       - k_proj
...       - v_proj
...       - o_proj
... run_conf:
...   model: "/experiments/Zephyr-replication/zephyr-7b-beta-sft-lora_2/checkpoint-final-merged"
...   tokenizer_args:
...     chat_template_name: "chat-ml"
...     padding_side: "right"
...     use_fast_tokenizer: true
...     fix_mistral_regex: false
...   model_args:
...     attn_implementation: "flash_attention_2"
...     use_cache: False
...   resume_from_checkpoint: False'''
>>> DPOExperimentConfig.from_yaml(yamlstr).training_args.dataloader_drop_last
False

"""

import logging

import trl

import aimfttk
import aimfttk.utils.running_process_opts as running_process_opts
from aimfttk.config.experiment import DPOExperimentConfig
from aimfttk.data import (
    apply_chat_template_to_preference_data,
    get_chat_template,
    handle_auto_split,
    setup_datainput,
    sort_longest_first,
    subsetup_tokenizer,
)
from aimfttk.model import get_model, subsetup_handle_peft
from aimfttk.utils.checkpoints import handle_checkpoint_resume

logger = logging.getLogger(__name__)


def subsetup_preference_training_data(exp_conf: DPOExperimentConfig, tokenizer):
    """Setup step: Training data"""
    train_data = setup_datainput(exp_conf.data_conf.training_data)
    train_data = apply_chat_template_to_preference_data(train_data, tokenizer)
    # Note: DPOTrainer handles filtering of long examples
    return train_data


def subsetup_preference_validation_data(exp_conf: DPOExperimentConfig, tokenizer, train_data):
    """Setup step: Validation data"""
    if exp_conf.data_conf.validation_data.type == aimfttk.config.data.DataInputType.AUTO_SPLIT:
        logger.warning("Using a split of the training data for validation.")
        train_data, valid_data = handle_auto_split(exp_conf.data_conf.validation_data, train_data)
    else:
        valid_data = setup_datainput(exp_conf.data_conf.validation_data)
        if valid_data is None:
            return train_data, None
        valid_data = apply_chat_template_to_preference_data(valid_data, tokenizer)
    # Note: DPOTrainer handles filtering of long examples
    valid_data = sort_longest_first(valid_data)
    return train_data, valid_data


def setup_dpo_trainer(exp_conf: DPOExperimentConfig):
    """Runs the setup steps and returns the trainer, which is the end result of the experiment setup"""
    # Chat template and tokenizer:
    chat_template = get_chat_template(exp_conf.run_conf.tokenizer_args.chat_template_name)
    tokenizer, new_tokens_added = subsetup_tokenizer(
        tokenizer_name_or_path=exp_conf.run_conf.tokenizer,
        chat_template=chat_template,
        missing_pad_token_strategy=exp_conf.run_conf.tokenizer_args.missing_pad_token_strategy,
        tokenizer_args=exp_conf.run_conf.tokenizer_args,
        overwrite_chat_template=False,
    )
    if new_tokens_added:
        raise ValueError("The DPO training is not suitable for adding new tokens, but tokenizer setup added some!")

    # Data:
    train_data = subsetup_preference_training_data(exp_conf, tokenizer)
    train_data, valid_data = subsetup_preference_validation_data(exp_conf, tokenizer, train_data)

    # Model:
    model = get_model(
        model_name_or_path=exp_conf.run_conf.model,
        model_load_kwargs=exp_conf.run_conf.model_args.get_model_load_kwargs(),
        quantization_config=exp_conf.quant_conf.get_hf_config(),
    )
    model = subsetup_handle_peft(
        exp_conf.peft_conf, model, peft_extra_modules_to_save=[], training_args=exp_conf.training_args
    )

    # Create the trainer, which is the actual output of the function.
    trainer = trl.DPOTrainer(
        model=model,
        args=exp_conf.training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        processing_class=tokenizer,
    )
    return trainer


def run_dpo(exp_conf):
    """DPO main function"""
    exp_conf.resolve_training_args()
    # DPO training args contains this
    exp_conf.training_args.dataset_num_proc = running_process_opts.num_preprocess_workers
    trainer = setup_dpo_trainer(exp_conf)
    resume = handle_checkpoint_resume(exp_conf.run_conf.resume_from_checkpoint, exp_conf.training_args.output_dir)
    trainer.train(resume_from_checkpoint=resume)
    final_checkpoint_path = exp_conf.training_args.output_dir + "/" + exp_conf.run_conf.final_checkpoint_name
    trainer.save_model(final_checkpoint_path)
