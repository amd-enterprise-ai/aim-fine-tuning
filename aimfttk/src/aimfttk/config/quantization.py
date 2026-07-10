# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Quantization configurations"""

from enum import Enum
from typing import List, Literal

from transformers import BitsAndBytesConfig

from aimfttk.config.base import BaseConfig


class QuantizationType(str, Enum):
    NO_QUANTIZATION = "no-quantization"
    BITSANDBYTES = "bits-and-bytes"


class BaseQuantizationConfig(BaseConfig):
    """Base type for Quantization configs"""

    quantization_type: QuantizationType


class NoQuantizationConfig(BaseQuantizationConfig):
    """A marker not to use quantization"""

    quantization_type: Literal[QuantizationType.NO_QUANTIZATION] = QuantizationType.NO_QUANTIZATION

    def get_hf_config(self):
        return None


class BnBQuantizationConfig(BaseQuantizationConfig):
    """Bits and Bytes configuration

    The options are from the BitsAndBytes config,
    see: https://huggingface.co/docs/transformers/en/main_classes/quantization#transformers.BitsAndBytesConfig
    """

    quantization_type: Literal[QuantizationType.BITSANDBYTES] = QuantizationType.BITSANDBYTES
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    llm_int8_threshold: float = 6.0
    llm_int8_skip_modules: List[str] | None = None
    llm_int8_enable_fp32_cpu_offload: bool = False
    llm_int8_has_fp16_weight: bool = False
    bnb_4bit_compute_dtype: str | None = None
    bnb_4bit_quant_type: Literal["fp4"] | Literal["nf4"] = "fp4"
    bnb_4bit_use_double_quant: bool = False
    bnb_4bit_quant_storage: str | None = None

    def get_hf_config(self):
        # Get the set of attributes which relate to this HF stuff specifically (not to QuantizationConfigs in general):
        hf_keys = set(self.model_fields.keys()) - set(super().model_fields.keys())
        return BitsAndBytesConfig(**{key: getattr(self, key) for key in hf_keys})
