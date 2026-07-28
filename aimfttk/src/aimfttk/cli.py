# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The aimfttk CLIs"""

import argparse
import logging
from collections import namedtuple

from pydantic import ValidationError
from transformers import AutoTokenizer, enable_full_determinism, set_seed

import aimfttk.utils.running_process_opts as running_process_opts
from aimfttk.config import DPOExperimentConfig, GenerationConfig, SFTExperimentConfig
from aimfttk.dpo import run_dpo
from aimfttk.generate import generate
from aimfttk.sft import run_sft
from aimfttk.utils.conventions import ensure_tensorboard_logging_dir
from aimfttk.utils.merge_adapter import merge_and_save
from aimfttk.utils.model import should_remove_non_lora_layers
from aimfttk.utils.replace_tokens import modify_tokenizer_files
from aimfttk.utils.tracking import setup_tracking
from aimfttk.utils.vllm_compatibility import remove_non_lora_layers

logger = logging.getLogger(__name__)

FinetuningMethod = namedtuple("FinetuningMethod", ["config_class", "runner"])
METHODS = {"sft": FinetuningMethod(SFTExperimentConfig, run_sft), "dpo": FinetuningMethod(DPOExperimentConfig, run_dpo)}


def load_config(cls, path):
    """Load the config from the given path"""
    with open(path) as fi:
        config = cls.from_yaml(fi)
    return config


def aimfttk_main_cli(argv=None):
    parser = argparse.ArgumentParser(description="Finetuning")
    parser.add_argument("--logging-level", default="INFO")
    parser.add_argument(
        "--num-preprocess-workers", type=int, default=4, help="Number of processes to use for preprocessing"
    )
    parser.add_argument(
        "--mlflow-server-uri",
        type=str,
        default=None,
        help="MLflow server URI. Can be local path.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="default",
        help="Experiment name that is used for MLflow tracking",
    )
    parser.add_argument(
        "--hf-mlflow-log-artifacts",
        type=str,
        default="False",
        help="Whether to store model artifacts in MLFlow",
    )
    parser.add_argument(
        "method",
        type=str,
        choices=METHODS.keys(),
        help="The kind of finetuning to run. SFT is Supervised FineTuning. While DPO stands for Direct Preference "
        "Optimization, note that it implements a set of related preference optimization algorithms such as IPO "
        "and KTO as well.",
    )
    parser.add_argument("config", type=str, help="Path to the experiment's YAML config file.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.logging_level)
    ensure_tensorboard_logging_dir()
    config_class, runner = METHODS[args.method]
    config = load_config(config_class, args.config)
    setup_tracking(config, args)
    if config.run_conf.determinism == "full":
        enable_full_determinism(config.training_args.seed, warn_only=False)
    else:
        set_seed(config.training_args.seed, deterministic=config.run_conf.determinism == "half")
    running_process_opts.setup_running_process_options(args)
    runner(config)


def merge_adapter_cli(argv=None):
    parser = argparse.ArgumentParser(
        description="Merges adapters into a base model and saves the output as a single model"
    )
    parser.add_argument("basemodel", type=str, help="Name (e.g. HuggingFace Hub id) or Path of Base Model")
    parser.add_argument("peftmodel", type=str, help="Name or Path of Peft adapter")
    parser.add_argument("outpath", type=str, help="Where to save the merged model")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Name or Path to tokenizer to save with the model. If not specified, will use the adapter model tokenizer.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for loading the model. Can be 'auto', 'cpu', 'cuda' or a JSON string.",
    )

    args = parser.parse_args(argv)
    if args.tokenizer is not None:
        tokenizer_name_or_path = args.tokenizer
    else:
        tokenizer_name_or_path = args.peftmodel
    merge_and_save(args.basemodel, tokenizer_name_or_path, args.peftmodel, args.outpath, args.device_map)


def replace_tokens_cli(argv=None):
    parser = argparse.ArgumentParser(
        description="Replaces tokens, specify pairs like:\n"
        "--from <|assistant|> --to <|im_start|> --from <|user|> --to <|im_end|>"
    )
    parser.add_argument("tokenizer", type=str, help="Name (e.g. HuggingFace Hub id) or Path of Tokenizer")
    parser.add_argument("outpath", type=str, help="Where to save the tokenizer")
    parser.add_argument(
        "--from", type=str, action="append", help="The token to remove, specify in pairs with --to", required=True
    )
    parser.add_argument(
        "--to", type=str, action="append", help="The token to replace with, specify in pairs with --from", required=True
    )
    args = parser.parse_args(argv)
    if len(getattr(args, "from")) != len(args.to):
        parser.error("Different number of --from and --to specified!")
    replacements = dict(zip(getattr(args, "from"), args.to))
    # 1. Copy the tokenizer:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.save_pretrained(args.outpath)
    # 2. Modify the new tokenizer's files in place:
    modify_tokenizer_files(args.outpath, replacements=replacements)


def generation_cli(argv=None):
    parser = argparse.ArgumentParser(description="Generate output with an LLM")
    parser.add_argument("config", type=str, help="Path to the experiment's YAML config file.")
    parser.add_argument("outpath", type=str, help="Path where to write the YAML")
    args = parser.parse_args(argv)
    config = load_config(GenerationConfig, args.config)
    generate(config, args.outpath)


def create_vllm_compatible_adapter_cli(argv=None):
    """Create a vLLM compatible adapter by removing non-LORA layers. This is useful when you have trained an adapter.
    Currently, we remove all layers that do not have the string '.lora_' in their name. This is because the PEFT lib-
    rary saves the base embedding layers as well when save() is called. This is not supported in vllm. Thus, to make
    the adapter compatible with vLLM, we remove the base embedding layers.

    Note 2: that this script assumes that the removed layers are indeed the same as the base model checkpoint. An exa-
    mple where this assumption is broken is when the vocabulary is extended and thus, the embedding layer is updated.
    In such cases, the embeddings should not be removed. In future, we would like to check if any of checkpoint's con-
    stituent layers have been updated and remove only the layers that have not been updated.
    UPDATE: Added an option to check a training config file to determine whether the layers should be removed. However,
    this still needs the training config file. In the future, maybe we can simply check if the embedding layers are
    the same.

    Note 2: This script does not make any updates to the LORA layers. vLLM continues to support the LORA layers as ad-
    ded by the HuggingFace library.

    """
    parser = argparse.ArgumentParser(
        description="Take in a HuggingFace adapter's binary folder path and remove the embeddings layer weights required for vLLM compatibility"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model folder (e.g. path to the folder containing the adapter) to be made compatible",
    )
    parser.add_argument(
        "--training-config",
        type=str,
        help="Path to training config to check and determine whether the layers should be removed or not.",
    )
    args = parser.parse_args(argv)
    if args.training_config is not None:
        for finetuning_method in METHODS.values():
            try:
                config = load_config(finetuning_method.config_class, args.training_config)
                break
            except ValidationError:
                pass  # keep looping over different methods to find right kind of config
        else:
            raise ValueError(f"Could not load training config {args.config}")
        if not should_remove_non_lora_layers(config):
            logger.info("Keeping non-LoRA layers, they are needed.")
            return
    logger.info("Removing non-LoRA layers")
    remove_non_lora_layers(args.model_path)
