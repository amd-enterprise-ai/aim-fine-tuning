# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Barebones SFT finetuning script

The following config YAML is an example based on the Huggingface alignment
handbook, and specifically the SFT Lora single GPU setup, (a doctest so that it remains up-to-date):

>>> yamlstr='''
... data_conf:
...   training_data:
...     type: PRECOMPUTE_WEIGHTED_MIX
...     datasets:
...       - path: "/datasets/Dolly-Silosplit-train.jsonl"
...         sampling_weight: 1.0
...       - path: "/datasets/OASST1.jsonl"
...         sampling_weight: 0.5
...   validation_data:
...     type: CONCATENATION
...     datasets:
...       - path: "/datasets/Dolly-Silosplit-validation.jsonl"
... batchsize_conf:
...   total_train_batch_size: 512
...   max_per_device_train_batch_size: 4
... training_args:
...   output_dir: "/experiments/EXP-123"
...   bf16: true
...   gradient_checkpointing: true
...   gradient_checkpointing_kwargs: {"use_reentrant": false}
...   learning_rate: 2.0e-05
...   logging_steps: 5
...   logging_strategy: "steps"
...   lr_scheduler_type: "cosine"
...   max_steps: 50
...   num_train_epochs: 23
...   overwrite_output_dir: true
...   push_to_hub: False
...   report_to: ["none"]
...   save_strategy: "steps"
...   save_steps: 5
...   eval_strategy: "epoch"
... sft_args:
...   max_seq_length: 2048
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
...   model: "mistralai/Mistral-7B-v0.1"
...   model_args:
...     attn_implementation: "flash_attention_2"
...     use_cache: False
...   tokenizer_args:
...     chat_template_name: "chat-ml"
...     padding_side: "right"
...     use_fast_tokenizer: true
...     fix_mistral_regex: false
...   resume_from_checkpoint: False
... tracking:
...   mlflow_server_uri: "file:///home/<USERNAME>/mlruns or http://10.172.10.196:80"
...   experiment_name: "default"
...   hf_mlflow_log_artifacts: "False"'''
>>> SFTExperimentConfig.from_yaml(yamlstr).training_args.dataloader_drop_last
False

"""

import pathlib

import transformers
from transformers.utils import logging

import aimfttk
from aimfttk.config import SFTExperimentConfig
from aimfttk.data import (
    filter_long_examples,
    get_chat_template,
    handle_auto_split,
    in_context_tokenize,
    setup_datainput,
    sort_longest_first,
    subsetup_tokenizer,
    tokenize_with_chat_template,
)
from aimfttk.data.collator import DataCollatorForCompletionOnlyLM
from aimfttk.model import get_model, subsetup_handle_peft
from aimfttk.utils.checkpoints import handle_checkpoint_resume
from aimfttk.utils.structured_logging import StructuredTrainingLogCallback

logger = logging.get_logger("sft")


def subsetup_sft_training_data(exp_conf: SFTExperimentConfig, tokenizer):
    """Setup step: Training data"""
    train_data = setup_datainput(exp_conf.data_conf.training_data)
    train_data = tokenize_with_chat_template(
        train_data,
        tokenizer,
        truncation=exp_conf.sft_args.length_handling == "truncate",
        max_len=exp_conf.sft_args.max_seq_length,
    )
    if exp_conf.sft_args.length_handling == "filter":
        # Train data might be an iterable dataset, in which case we do not know how many samples get filtered out.
        # Let's still send out a warning when we can, the extra information is useful.
        train_len_before_filter = None
        if getattr(train_data, "num_rows", None) is not None:
            train_len_before_filter = train_data.num_rows
        train_data = filter_long_examples(train_data, exp_conf.sft_args.max_seq_length)
        if train_len_before_filter is not None and train_len_before_filter > train_data.num_rows:
            logger.warning(
                f"Filtered out {train_len_before_filter - train_data.num_rows} training examples "
                f"due to exceeding maximum sequence length {exp_conf.sft_args.max_seq_length}. "
                "Note that this warning is not emitted with iterable (streaming) datasets, but the filtering still takes "
                "place."
            )
    return train_data


def subsetup_sft_validation_data(exp_conf: SFTExperimentConfig, tokenizer, train_data):
    """Setup step: Validation data

    This may take a split of the training data,
    so it always returns both the train_data and the valid_data
    """
    if exp_conf.data_conf.validation_data.type == aimfttk.config.data.DataInputType.AUTO_SPLIT:
        logger.warning("Using a split of the training data for validation.")
        train_data, valid_data = handle_auto_split(exp_conf.data_conf.validation_data, train_data)
    else:
        valid_data = setup_datainput(exp_conf.data_conf.validation_data)
        if valid_data is None:
            return train_data, None
        valid_data = tokenize_with_chat_template(
            valid_data,
            tokenizer,
            truncation=exp_conf.sft_args.length_handling == "truncate",
            max_len=exp_conf.sft_args.max_seq_length,
        )
        if exp_conf.sft_args.length_handling == "filter":
            valid_len_before_filter = valid_data.num_rows
            valid_data = filter_long_examples(valid_data, exp_conf.sft_args.max_seq_length)
            if valid_len_before_filter > valid_data.num_rows:
                logger.warning(
                    f"Filtered out {valid_len_before_filter - valid_data.num_rows} validation examples "
                    f"due to exceeding maximum sequence length {exp_conf.sft_args.max_seq_length}."
                )
    valid_data = sort_longest_first(valid_data)
    return train_data, valid_data


def subsetup_sft_collator(exp_conf: SFTExperimentConfig, chat_template, tokenizer):
    """Setup step: Data collator"""
    if (
        exp_conf.run_conf.tokenizer_args.missing_pad_token_strategy
        == aimfttk.config.run.MissingPadTokenStrategy.UNK_CONVERT_TO_EOS
        and exp_conf.data_conf.train_on_completions_only
    ):
        # NOTE: this should be relatively easy to implement, since the collator funcationality can be chained. One
        # possible  implementation would be to somehow make a collator subclass that can take a set of collation
        # functions and run them in a chain.
        raise NotImplementedError("Currently cannot use both UNK_CONVERT_TO_EOS and train_on_completions_only")
    if (
        exp_conf.run_conf.tokenizer_args.missing_pad_token_strategy
        == aimfttk.config.run.MissingPadTokenStrategy.UNK_CONVERT_TO_EOS
    ):
        data_collator = aimfttk.data.collator.UnkConvertToEOSCollatorForLM(tokenizer=tokenizer, mlm=False)
    elif exp_conf.sft_args.train_on_completions_only:
        data_collator = DataCollatorForCompletionOnlyLM(
            assistant_start=in_context_tokenize(tokenizer, chat_template.assistant_start, prefix="\n"),
            assistant_end=in_context_tokenize(tokenizer, chat_template.assistant_end, prefix="\n"),
            tokenizer=tokenizer,
        )
    else:
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return data_collator


def subsetup_handle_new_tokens(exp_conf: SFTExperimentConfig, model, tokenizer, new_tokens_added):
    """Setup step: handle new tokens, save a new basemodel if using PEFT"""
    peft_extra_modules_to_save = []
    # TODO: Make the new basemodel creation a separate stage, not implicit in setup.
    if new_tokens_added:
        peft_extra_modules_to_save = aimfttk.model.grow_embeddings(model, new_num_tokens=len(tokenizer))
        if exp_conf.peft_conf.peft_type != aimfttk.config.hf_integration.NO_PEFT:
            save_path = pathlib.Path(exp_conf.training_args.output_dir) / exp_conf.sft_args.save_name_if_new_basemodel
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.warning(
                f"Saved a new basemodel to {str(save_path)} due to added tokens. Use this with the adapter checkpoints."
            )
    return peft_extra_modules_to_save


def setup_sft_trainer(exp_conf: SFTExperimentConfig):
    """Runs the setup steps and returns the trainer, which is the end result of the experiment setup"""
    # Chat template and tokenizer:
    chat_template = get_chat_template(exp_conf.run_conf.tokenizer_args.chat_template_name)
    tokenizer, new_tokens_added = subsetup_tokenizer(
        tokenizer_name_or_path=exp_conf.run_conf.tokenizer,
        chat_template=chat_template,
        missing_pad_token_strategy=exp_conf.run_conf.tokenizer_args.missing_pad_token_strategy,
        tokenizer_args=exp_conf.run_conf.tokenizer_args,
        overwrite_chat_template=True,
    )

    # Data:
    train_data = subsetup_sft_training_data(exp_conf, tokenizer)
    train_data, valid_data = subsetup_sft_validation_data(exp_conf, tokenizer, train_data)
    data_collator = subsetup_sft_collator(exp_conf, chat_template, tokenizer)

    # Model:
    model = get_model(
        model_name_or_path=exp_conf.run_conf.model,
        model_load_kwargs=exp_conf.run_conf.model_args.get_model_load_kwargs(),
        quantization_config=exp_conf.quant_conf.get_hf_config(),
    )
    peft_extra_modules_to_save = subsetup_handle_new_tokens(exp_conf, model, tokenizer, new_tokens_added)
    model = subsetup_handle_peft(
        exp_conf.peft_conf, model, peft_extra_modules_to_save, training_args=exp_conf.training_args
    )

    # Create the trainer, which is the actual output of the function.
    trainer = transformers.Trainer(
        model=model,
        args=exp_conf.training_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[StructuredTrainingLogCallback()],
    )

    return trainer


def run_sft(exp_conf):
    """SFT main function"""
    exp_conf.resolve_training_args()
    trainer = setup_sft_trainer(exp_conf)
    resume = handle_checkpoint_resume(exp_conf.run_conf.resume_from_checkpoint, exp_conf.training_args.output_dir)
    trainer.train(resume_from_checkpoint=resume)
    final_checkpoint_path = exp_conf.training_args.output_dir + "/" + exp_conf.run_conf.final_checkpoint_name
    trainer.save_model(final_checkpoint_path)
