# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import logging
from functools import wraps
from typing import Tuple

import transformers

from aimfttk.config.run import MissingPadTokenStrategy, TokenizerArguments
from aimfttk.data.chat_templates import ChatTemplate

logger = logging.getLogger(__file__)


def handle_missing_pad_token(tokenizer, strategy: MissingPadTokenStrategy):
    """Handles missing pad_token_id with the chosen strategy"""
    if tokenizer.pad_token_id is None:
        logger.info(f"Setting pad_token with strategy {strategy}")
        if strategy == MissingPadTokenStrategy.EOS_REPURPOSE:
            # This actually leads to ignoring all EOS tokens in loss computation, which is bad,
            # because we want to predict EOS in some cases.
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
        elif strategy == MissingPadTokenStrategy.BOS_REPURPOSE:
            tokenizer.pad_token_id = tokenizer.bos_token_id
            tokenizer.pad_token = tokenizer.bos_token
        elif strategy == MissingPadTokenStrategy.UNK_REPURPOSE:
            # This is the default
            tokenizer.pad_token_id = tokenizer.unk_token_id
            tokenizer.pad_token = tokenizer.unk_token
        elif strategy == MissingPadTokenStrategy.UNK_CONVERT_TO_EOS:
            tokenizer.pad_token_id = tokenizer.unk_token_id
            tokenizer.pad_token = tokenizer.unk_token
        else:
            raise ValueError(f"Missing pad_token_id and unknown missing pad token strategy {strategy}")
    elif strategy == MissingPadTokenStrategy.UNK_CONVERT_TO_EOS:
        raise ValueError("Pad token exists on model, but specified strategy {strategy}")
    else:
        logger.info("Pad token id is already set, no need to handle missing pad token")


def wrap_for_save_in_inference_mode(tokenizer: transformers.PreTrainedTokenizerBase, inference_end_token: str):
    """Overwrites the tokenizer instance's save_pretrained with a wrapper that sets inference time settings

    The inference time setting are:
        add_bos_token=True
        add_eos_token=False
        eos_token = inference_end_token
    """
    if hasattr(tokenizer, "_silogen_mod_save_pretrained"):
        raise RuntimeError("Trying to wrap tokenizer saver a second time!")

    @wraps(tokenizer.save_pretrained)
    def inference_mode_wrapper(self, *args, **kwargs):
        logger.info("Saving pretrained tokenizer in inference mode.")
        # Define inference state:
        inference_settings = {
            "add_bos_token": True,
            "add_eos_token": False,
            "eos_token": inference_end_token,
        }
        # Save previous state:
        state = {}
        for attr in inference_settings:
            if hasattr(self, attr):
                state[attr] = getattr(self, attr)
        # Set inference mode options:
        for attr, value in inference_settings.items():
            # Only set the inference state in case the attribute is part of the tokenizer's normal settings:
            if attr in state:
                logger.info(f"Setting tokenizer.{attr} to {value} for inference")
                setattr(self, attr, value)
        # Run saving method:
        call_output = self._silogen_mod_save_pretrained(*args, **kwargs)
        # Recall previous state:
        for attr, recalled in state.items():
            setattr(self, attr, recalled)
        return call_output

    tokenizer._silogen_mod_save_pretrained = tokenizer.save_pretrained
    tokenizer.save_pretrained = inference_mode_wrapper.__get__(tokenizer)  # type: ignore


def subsetup_tokenizer(
    tokenizer_name_or_path: str,
    chat_template: ChatTemplate | None,
    missing_pad_token_strategy: MissingPadTokenStrategy,
    tokenizer_args: TokenizerArguments,
    overwrite_chat_template=True,
) -> Tuple[transformers.PreTrainedTokenizerBase, bool]:
    """Setup step: Tokenizer

    NOTE: returns a tuple of (tokenizer, bool)
    where the boolean indicates if new tokens were added
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        **tokenizer_args.get_tokenizer_load_kwargs(),
    )
    num_new_tokens = 0
    inference_end_token = tokenizer.eos_token
    if chat_template is not None:
        if overwrite_chat_template:
            tokenizer.chat_template = chat_template.jinjastr
        else:
            if tokenizer.chat_template != chat_template.jinjastr:
                MSG = "Not allowing chat template to be overwritten, but the chat template stored on the tokenizer "
                MSG += "does not match the one specified!"
                raise ValueError(MSG)
        num_new_tokens = tokenizer.add_special_tokens(chat_template.special_tokens, replace_extra_special_tokens=False)
        inference_end_token = chat_template.assistant_end
    if tokenizer_args.tokenizer_type is None:
        handle_missing_pad_token(tokenizer, missing_pad_token_strategy)
        # Add a saving wrapper that prepares the saved version for inference:
        #  add_bos_token=True
        #  add_eos_token=False
        #  eos_token = chat_template.assistant_end
        wrap_for_save_in_inference_mode(tokenizer, inference_end_token)
    return tokenizer, num_new_tokens > 0


def subsetup_tokenizer_for_inference(tokenizer_name_or_path: str):
    """Setup tokenizer for inference mode"""
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        add_eos_token=False,
    )
    return tokenizer
