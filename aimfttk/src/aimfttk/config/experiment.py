# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import math
import warnings
from textwrap import dedent
from typing import Literal

from pydantic import Field, model_validator
from transformers.utils import logging

from aimfttk.config.base import BaseConfig
from aimfttk.config.data import ChatTrainValidConfig, DataInputType
from aimfttk.config.hf_integration import (
    AIMFTTKDPOConfig,
    AIMFTTKTrainingArguments,
    BatchsizeConfig,
    GenericPeftConfig,
    NoPeftConfig,
    PretrainedPeftConfig,
    SFTArguments,
)
from aimfttk.config.quantization import BnBQuantizationConfig, NoQuantizationConfig, QuantizationType
from aimfttk.config.run import ChatTemplateName, RunConfig
from aimfttk.config.tracking import TrackingConfig
from aimfttk.data.data_types import data_type_by_method
from aimfttk.utils.distributed import is_fsdp
from aimfttk.utils.training import resolve_batchsize_and_accumulation

logger = logging.get_logger(__file__)


class FinetuningTrackingConfig(TrackingConfig):
    """Settings that define how run details are logged"""

    hf_mlflow_log_artifacts: str = Field(default="False", description="Whether to store model artifacts in MLFlow.")


class Overrides(BaseConfig):
    """Override options

    These implement dynamic scaling for the learning rate.
    """

    lr_multiplier: float = Field(
        default=1.0, description="Multiplier applied to the learning rate in the training_args"
    )
    lr_batch_size_scaling: Literal["none", "sqrt", "linear"] = Field(
        default="none",
        description=dedent(
            """Scales the learning rate in the training_args by a factor derived from the total training batch size. \
            'none': No scaling. \
            'sqrt': Multiplies learning rate by square root of batch size (a classic scaling rule). \
            'linear': Multiplies learning rate by the batch size (a more modern scaling rule).
            """
        ),
    )


class ExperimentConfig(BaseConfig):
    """A full experiment's config

    See the various sub-configs for their options.
    """

    method: Literal["sft", "dpo"]
    data_conf: ChatTrainValidConfig = Field(description="The data input config")
    training_args: AIMFTTKTrainingArguments = Field(description="Transformer TrainingArguments with some restrictions")
    overrides: Overrides = Field(default=Overrides(), description="Override options to simplify the config interface")
    batchsize_conf: BatchsizeConfig = Field(description="Batch size configuration")
    peft_conf: NoPeftConfig | PretrainedPeftConfig | GenericPeftConfig = Field(description="Adapter configuration")
    run_conf: RunConfig = Field(description="Model related configuration")
    tracking: FinetuningTrackingConfig | None = Field(default=None, description="MLFlow tracking configuration")
    quant_conf: NoQuantizationConfig | BnBQuantizationConfig = Field(
        default=NoQuantizationConfig(), description="Quantization configuration"
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_data_tokenizer_fields(cls, values):
        """Migrate legacy tokenizer-related fields from data_conf to run_conf.tokenizer_args."""
        if not isinstance(values, dict):
            return values

        data_conf = values.get("data_conf")
        if not isinstance(data_conf, dict):
            return values

        has_legacy_padding_side = "padding_side" in data_conf
        has_legacy_chat_template_name = "chat_template_name" in data_conf
        has_legacy_missing_pad_token_strategy = "missing_pad_token_strategy" in data_conf
        if (
            not has_legacy_padding_side
            and not has_legacy_chat_template_name
            and not has_legacy_missing_pad_token_strategy
        ):
            return values

        run_conf = dict(values.get("run_conf") or {})
        tokenizer_args = dict(run_conf.get("tokenizer_args") or {})

        if has_legacy_padding_side:
            padding_side = data_conf.pop("padding_side")
            # Prefer explicit tokenizer_args.padding_side over legacy data_conf.padding_side.
            tokenizer_args.setdefault("padding_side", padding_side)

        if has_legacy_chat_template_name:
            chat_template_name = data_conf.pop("chat_template_name")
            # Prefer explicit tokenizer_args.chat_template_name over legacy data_conf.chat_template_name.
            tokenizer_args.setdefault("chat_template_name", chat_template_name)

        if has_legacy_missing_pad_token_strategy:
            missing_pad_token_strategy = data_conf.pop("missing_pad_token_strategy")
            # Prefer explicit tokenizer_args.missing_pad_token_strategy over legacy data_conf.missing_pad_token_strategy.
            tokenizer_args.setdefault("missing_pad_token_strategy", missing_pad_token_strategy)

        run_conf["tokenizer_args"] = tokenizer_args
        values["run_conf"] = run_conf
        return values

    def resolve_training_args(self):
        """Resolves the training args with the batch_size configuration and the HuggingFace post_init"""
        if self.batchsize_conf.enable_auto_batch_size:
            MSG = "Auto-resolving batch size configuration. This will override any batch size or gradient accumulation "
            MSG += "settings in the training_args."
            logger.info(MSG)
            per_device_train_batch_size, gradient_accumulation = resolve_batchsize_and_accumulation(
                self.batchsize_conf.total_train_batch_size, self.batchsize_conf.max_per_device_train_batch_size
            )
            per_device_eval_batch_size = (
                per_device_train_batch_size
                if self.batchsize_conf.per_device_eval_batch_size is None
                else self.batchsize_conf.per_device_eval_batch_size
            )
            self.training_args.per_device_train_batch_size = per_device_train_batch_size
            self.training_args.per_device_eval_batch_size = per_device_eval_batch_size
            self.training_args.gradient_accumulation_steps = gradient_accumulation
            if self.training_args.deepspeed:
                self.training_args.deepspeed.update(
                    {
                        "train_micro_batch_size_per_gpu": per_device_train_batch_size,
                        "train_batch_size": self.batchsize_conf.total_train_batch_size,
                    }
                )
        self.training_args.resolve_arguments()

    def __apply_overrides(self):
        original_lr = self.training_args.learning_rate
        if self.overrides.lr_multiplier != 1.0:
            MSG = f"Multiplying learning rate by multiplier {self.overrides.lr_multiplier}"
            logger.info(MSG)
            self.training_args.learning_rate *= self.overrides.lr_multiplier
        if self.overrides.lr_batch_size_scaling == "sqrt":
            MSG = "Scaling learning rate with the square root of the batch size"
            logger.info(MSG)
            self.training_args.learning_rate *= math.sqrt(self.batchsize_conf.total_train_batch_size)
        elif self.overrides.lr_batch_size_scaling == "linear":
            MSG = "Scaling learning rate linearly by the batch size"
            logger.info(MSG)
            self.training_args.learning_rate *= self.batchsize_conf.total_train_batch_size
        if self.training_args.learning_rate != original_lr:
            MSG = f"Learning rate changed from {original_lr} to {self.training_args.learning_rate}"
            logger.info(MSG)

    def model_post_init(self, __context):
        self.__apply_overrides()
        if (
            self.quant_conf.quantization_type == QuantizationType.BITSANDBYTES
            and is_fsdp
            and self.quant_conf.bnb_4bit_quant_storage is None
        ):
            # This is required for FSDP + QLoRA:
            dtype = "bfloat16" if self.training_args.bf16 else "float16" if self.training_args.fp16 else "float32"
            logger.warning(
                "Because of FSDP + Quantized basemodel, setting quant_conf.bnb_4bit_quant_storage to "
                f"inferred compute_dtype {dtype}."
            )
            self.quant_conf.bnb_4bit_quant_storage = dtype

        data_type_name = data_type_by_method[self.method].__name__
        if self.data_conf.training_data.data_type != data_type_name:
            logger.warning(
                f"Setting data_conf.training_data.data_type to {data_type_name} to match training method {self.method}"
            )
            self.data_conf.training_data.data_type = data_type_name

        if (
            self.data_conf.validation_data.type != DataInputType.NONE
            and self.data_conf.validation_data.data_type != data_type_name
        ):
            logger.warning(
                f"Setting data_conf.validation_data.data_type to {data_type_name} to match training method {self.method}"
            )
            self.data_conf.validation_data.data_type = data_type_name


class SFTExperimentConfig(ExperimentConfig):
    """A full SFT experiment's config

    See the various sub-configs for their options.
    """

    method: Literal["sft"] = "sft"
    sft_args: SFTArguments = Field(description="SFT specific arguments")

    def model_post_init(self, __context):
        super().model_post_init(__context)

        if (
            self.run_conf.tokenizer_args.chat_template_name != ChatTemplateName.CHAT_ML
            and self.sft_args.train_on_completions_only
        ):
            warnings.warn(
                "You are training on completions only, but not using the Chat-ML chat template. "
                "This might not work correctly."
            )


class DPOExperimentConfig(ExperimentConfig):
    """A full DPO experiment's config

    See the various sub-configs for their options.
    """

    method: Literal["dpo"] = "dpo"
    # We have to override training_args because TRL's DPOTrainer takes a subclass of TrainingArguments as args:
    # We used to have the regular training_args and a separate dpo_args subconfig in this class, similar to
    # SFTExperimentConfig, which has training_args + sft_args.
    training_args: AIMFTTKDPOConfig = Field(description="TRL DPOTrainerArguments with some restrictions")
