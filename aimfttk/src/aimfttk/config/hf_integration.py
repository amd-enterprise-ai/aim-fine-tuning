# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""HuggingFace Configs available through the config interface"""

from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Dict, List, Literal, Optional, Union

import peft
import transformers
from pydantic import ConfigDict, Field, model_serializer, model_validator
from transformers.trainer_pt_utils import AcceleratorConfig
from trl import DPOConfig

from aimfttk.config.base import BaseConfig, DisallowedInput
from aimfttk.utils.conventions import local_checkpoints_dir, local_logs_dir

NO_PEFT = "NO_PEFT"
PRETRAINED_PEFT = "PRETRAINED_PEFT"


class BatchsizeConfig(BaseConfig):
    """Config for determining the total batch size

    Total batch size is the effective batch size for the complete training run. It is equal to
    number of processes * per-device batch size * accumulation.

    The maximum batch size per device is the maximum batch size that can be accommodated on a single device.
    This mostly limited by the memory capacity of the device.
    """

    enable_auto_batch_size: bool = Field(
        default=True,
        description="Whether to automatically determine the per-device batch size",
    )
    total_train_batch_size: int = Field(description="The total batch size for the training run", default=1)
    max_per_device_train_batch_size: int = Field(description="The maximum training batch size per device", default=1)
    per_device_eval_batch_size: Optional[int] = Field(
        default=None,
        description="The maximum eval batch size per device, if not given, will use same as training batch size",
    )  # If None, will use the same as the training batch size


# Fix for the pydantic validation of transformers' AcceleratorConfig
@dataclass
class ValidatableAcceleratorConfig(AcceleratorConfig):
    dispatch_batches: Optional[bool] = field(
        default=None,
        metadata={
            "help": "If set to `True`, the dataloader prepared by the Accelerator is only iterated through on the main process"
            " and then the batches are split and broadcast to each process. Will default to `True` for `DataLoader` whose"
            " underlying dataset is an `IterableDataslet`, `False` otherwise."
        },
    )


class AIMFTTKTrainingArguments(BaseConfig, transformers.TrainingArguments):
    """HuggingFace TrainingArguments as Config with additional AIMFTTK conventions

    The list of training arguments is best available online (the version might not be up-to-date here):
    https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments

    output_dir has a default and likely should not be set by the user.

    gradient_accumulation_steps, per_device_train_batch_size, pre_device_eval_batch_size are all set automatically,
    based on the separate BatchsizeConfig, and cannot be modified in this config.

    The TrainingArguments object does a lot of things besides specifying the training configuration options
        (e.g. it holds a reference to some distributed computation state)
    """

    # NOTE:
    #     It takes a little bit of hacking to make TrainingArguments work right with pydantic. There are comments about
    #     the decisions below
    # Transformers sets parameters after instantiation so we need to allow extra params.
    # However, we can check for extra parameters before running
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    output_dir: str = Field(
        default=local_checkpoints_dir.as_posix(),
        description="The output directory where checkpoints will be written. Should be left as default, which is ./checkpoints",
    )

    # Since we're using pydantic BaseModel-style (as opposed to pydantic dataclasses), we need to run the
    # TrainingArguments __post_init__ in model_post_init explicitly
    def resolve_arguments(self):
        """Run the HuggingFace post_init. This should be run before passing to Trainer"""
        transformers.TrainingArguments.__post_init__(self)

    # Pydantic complains for inherited dataclasses with default_factories, need to redefine
    lr_scheduler_kwargs: Optional[Union[dict, str]] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Extra parameters for the lr_scheduler such as {'num_cycles': 1} for the cosine with hard restarts."
            )
        },
    )
    include_for_metrics: List[str] = field(
        default_factory=list,
        metadata={
            "help": "List of strings to specify additional data to include in the `compute_metrics` function."
            "Options: 'inputs', 'loss'."
        },
    )

    @model_validator(mode="after")
    def _validate_no_extra_inputs(self):
        if self.model_extra:
            raise ValueError(
                f"Extra input values are not allowed: {'; '.join(str(k)+': '+str(v) for k, v in self.model_extra.items())}"
            )
        return self

    @model_serializer(mode="wrap")
    def serialize_model(self, handler):
        partial_result = handler(self)
        # Note: using get(, True) to avoid KeyError (since the key might not be present in the dict
        if not partial_result.get("fsdp_config", True):
            del partial_result["fsdp_config"]
        if not partial_result.get("deepspeed", True):
            del partial_result["deepspeed"]
        if not partial_result.get("neftune_noise_alpha", True):
            del partial_result["neftune_noise_alpha"]
        if not partial_result.get("lr_scheduler_kwargs", True):
            del partial_result["lr_scheduler_kwargs"]
        return partial_result


class AIMFTTKDPOConfig(BaseConfig, DPOConfig):
    """HuggingFace TRL DPOConfig as Config with additional AIMFTTK conventions

    The list of training arguments is best available online (the version might not be up-to-date here):
    https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments

    Additionally, the DPOConfig has arguments specific to DPO training, which can be found here:
    https://huggingface.co/docs/trl/main/en/dpo_trainer#trl.DPOConfig

    The object does a lot of things besides specifying the training configuration options (e.g. it
    has computed properties like true training batch size etc.)

    NOTE:
        It takes a little bit of hacking to make TrainingArguments work right with pydantic. There are comments about
        the decisions below
    """

    output_dir: str = Field(
        default=local_checkpoints_dir.as_posix(),
        description="The output directory where checkpoints will be written. Should be left as default, which is ./checkpoints",
    )

    # Since we're using pydantic BaseModel-style (as opposed to pydantic dataclasses), we need to run the
    # TrainingArguments __post_init__ explicitly in model_post_init
    def resolve_arguments(self):
        """Run the HuggingFace post_init. This should be run before passing to Trainer"""
        transformers.TrainingArguments.__post_init__(self)

    # These class variables get added during post_init. The user cannot provide these as inputs.
    distributed_state: DisallowedInput = Field(default=None, exclude=True)
    deepspeed_plugin: DisallowedInput = Field(default=None, exclude=True)
    mixed_precision: DisallowedInput = Field(default=None, exclude=True)  # Added in transformers v5
    fsdp_plugin_args: DisallowedInput = Field(default=None, exclude=True)  # Added in transformers v5

    # Pydantic complains for inherited dataclasses with default_factories, need to redefine
    lr_scheduler_kwargs: Optional[Union[dict, str]] = field(
        default_factory=dict,
        metadata={
            "help": (
                "Extra parameters for the lr_scheduler such as {'num_cycles': 1} for the cosine with hard restarts."
            )
        },
    )
    include_for_metrics: List[str] = field(
        default_factory=list,
        metadata={
            "help": "List of strings to specify additional data to include in the `compute_metrics` function."
            "Options: 'inputs', 'loss'."
        },
    )

    # The following are DPOConfig-specific args:
    max_length: int = Field(default=2048, description="Maximum total length of inputs")
    max_prompt_length: int = Field(default=1536, description="Maximum prompt length")

    @model_serializer(mode="wrap")
    def serialize_model(self, handler):
        partial_result = handler(self)
        # Note: using get(, True) to avoid KeyError (since the key might not be present in the dict
        if not partial_result.get("fsdp_config", True):
            del partial_result["fsdp_config"]
        if not partial_result.get("deepspeed", True):
            del partial_result["deepspeed"]
        if not partial_result.get("neftune_noise_alpha", True):
            del partial_result["neftune_noise_alpha"]
        if not partial_result.get("lr_scheduler_kwargs", True):
            del partial_result["lr_scheduler_kwargs"]
        return partial_result


class NoPeftConfig(BaseConfig):
    """A trivial config specifying that no peft is used"""

    peft_type: Literal[NO_PEFT]  # type: ignore


class PretrainedPeftConfig(BaseConfig):
    """PEFT adapter uses the config and initialisation from a pretrained adapter"""

    peft_type: Literal[PRETRAINED_PEFT]  # type: ignore
    name_or_path: str = Field(description="HF ID or path to the pretrained peft.")


class GenericPeftConfig(BaseConfig):
    """Config for any new initialized PEFT Adapter

    See https://huggingface.co/docs/peft/tutorial/peft_model_config for the possible kwargs
    and https://github.com/huggingface/peft/blob/v0.7.1/src/peft/utils/peft_types.py for the types.

    Example:

        >>> loaded_data = {'peft_type':'LORA', 'task_type': 'CAUSAL_LM',
        ...         'peft_kwargs': {'r': 32, 'target_modules': ['v_proj']}}
        >>> generic_conf = GenericPeftConfig(**loaded_data)
        >>> generic_conf.get_peft_config()
        LoraConfig(task_type=<TaskType.CAUSAL_LM: 'CAUSAL_LM'>, peft_type=<PeftType.LORA: 'LORA'>, ...)
    """

    # TODO: Discriminate automatically between different PeftConfigs.
    # Unfortunately PEFT has taken slightly anti-pydantic approach in resolving which PEFT adapter type to use. In PEFT,
    # the peft_type attribute gets set in __post_init__ in the various subclasses, so it is difficult to discriminate
    # based on that.
    peft_type: peft.PeftType
    task_type: peft.TaskType = peft.TaskType.CAUSAL_LM
    peft_kwargs: Dict[str, Any] = field(default_factory=dict)

    def get_peft_config(self):
        """Resolve this generic config to the actual specific PeftConfig that this describes"""
        return peft.PEFT_TYPE_TO_CONFIG_MAPPING[self.peft_type](task_type=self.task_type, **self.peft_kwargs)


class SFTArguments(BaseConfig):
    """Supervised fine-tuning arguments"""

    max_seq_length: int = Field(
        default=2048,
        description="Maximum length input sequence length. Longer sequences will be filtered or truncated.",
    )
    length_handling: Literal["filter", "truncate"] = Field(
        default="filter",
        description=dedent(
            """How to handle examples that are longer than max_seq_length. \
            'filter': Filter out these examples from the training set. \
            'truncate': Truncate these examples to max_seq_length. Note that this might lead to loss of information and worse performance, especially if the important information is at the end of the sequence."""
        ),
    )
    # This is only used if a new basemodel needs to be saved, e.g. if the embeddings are grown to account for new
    # tokens.
    # By convention, we refer to "checkpoint-basemodel" as the original basemodel, and "checkpoint-new-basemodel" as the new basemodel
    save_name_if_new_basemodel: str = Field(
        default="checkpoint-new-basemodel", description="If a new basemodel is saved, it will be saved with this name"
    )
    train_on_completions_only: bool = Field(default=False, description="Only compute loss on the assistant's turns.")
