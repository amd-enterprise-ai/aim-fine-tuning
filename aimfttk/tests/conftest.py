# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import os

import datasets
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import (
    Gemma2Config,
    Gemma2ForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    enable_full_determinism,
    set_seed,
)

# Don't let tests use GPUs, which does not work in all environments + we don't want spurious GPU usage.
os.environ["CUDA_VISIBLE_DEVICES"] = ""


@pytest.fixture
def datasets_cache_in_tmpdir(tmpdir):
    datasets.config.hf_datasets_cache = tmpdir


@pytest.fixture(scope="session")
def tiny_llama_model_and_tokenizer(tmp_path_factory):
    """Create a tiny deterministic Llama model and simple tokenizer for testing.

    This fixture creates a locally-initialized model with random weights (using a fixed seed)
    to avoid downloading models from HuggingFace Hub during tests.
    """
    # Use a temp directory that persists for the whole test session
    model_dir = tmp_path_factory.mktemp("tiny_llama_model")

    # 1. Create tiny model config (small dimensions for fast CPU tests)
    config = LlamaConfig(
        vocab_size=1024,  # Small vocab
        hidden_size=64,  # Tiny hidden size
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )

    # 2. Initialize model with deterministic random weights
    # Use transformers' determinism utilities for complete reproducibility
    set_seed(42)
    enable_full_determinism(42)
    torch.manual_seed(42)
    model = LlamaForCausalLM(config)

    # 3. Create a simple deterministic tokenizer
    # Use BPE tokenizer with minimal vocabulary
    tokenizer_obj = Tokenizer(models.BPE())
    tokenizer_obj.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Train on simple vocab
    trainer = trainers.BpeTrainer(vocab_size=1024, special_tokens=["<unk>", "<s>", "</s>", "<pad>"])
    # Train on alphabet + spaces
    corpus = [" ".join(chr(i) for i in range(ord("A"), ord("Z") + 1))] * 10
    tokenizer_obj.train_from_iterator(corpus, trainer=trainer)

    # Wrap in PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )

    # Add a basic chat template (required by aimfttk)
    tokenizer.chat_template = (
        "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}{{ eos_token }}"
    )

    # 4. Save both to disk (so from_pretrained works)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    return str(model_dir)


@pytest.fixture(scope="session")
def tiny_gemma2_model_and_tokenizer(tmp_path_factory):
    """Create a tiny deterministic Gemma2 model and simple tokenizer for testing.

    This fixture creates a locally-initialized model with random weights (using a fixed seed)
    to avoid downloading models from HuggingFace Hub during tests.
    """
    model_dir = tmp_path_factory.mktemp("tiny_gemma2_model")

    # 1. Create tiny model config
    config = Gemma2Config(
        vocab_size=1024,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )

    # 2. Initialize model with deterministic random weights (different seed than Llama)
    # Use transformers' determinism utilities for complete reproducibility
    set_seed(43)
    enable_full_determinism(43)
    torch.manual_seed(43)
    model = Gemma2ForCausalLM(config)

    # 3. Create a simple deterministic tokenizer (same approach as Llama)
    tokenizer_obj = Tokenizer(models.BPE())
    tokenizer_obj.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = trainers.BpeTrainer(vocab_size=1024, special_tokens=["<unk>", "<s>", "</s>", "<pad>"])
    corpus = [" ".join(chr(i) for i in range(ord("A"), ord("Z") + 1))] * 10
    tokenizer_obj.train_from_iterator(corpus, trainer=trainer)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )

    # Add a basic chat template (required by aimfttk)
    tokenizer.chat_template = (
        "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}{{ eos_token }}"
    )

    # 4. Save both to disk
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    return str(model_dir)
