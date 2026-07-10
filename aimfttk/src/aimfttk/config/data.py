# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from enum import Enum
from typing import List, Literal

from pydantic import Field

from aimfttk.config.base import BaseConfig


class DataInputType(str, Enum):
    """
    In PRECOMPUTE_WEIGHTED_MIX, the resulting interleaved dataset is fully precomputed. This means that we wait until
    every single sample in every dataset gets picked. The upside is that the resulting dataset can store and cache
    things like tokenization etc. (no need to continuously run expensive data processsing on-the-fly) and we can use
    things such as group_by_length (which requires random-access and helps to reduce padding). The downside is that the
    resulting dataset can be very large, so any full dataset mapping and filtering operations etc. can take a while.

    In CONCATENATION, the resulting dataset is fully precomputed, but this is fast because each dataset is just
    concatenated, unlike in the interleaving processes.

    AUTO_SPLIT is used for taking a split of the training data for validation.

    The NONE DataInputType is meant for skipping validation.
    """

    CONCATENATION = "CONCATENATION"
    PRECOMPUTE_WEIGHTED_MIX = "PRECOMPUTE_WEIGHTED_MIX"
    AUTO_SPLIT = "AUTO_SPLIT"
    NONE = "NONE"


class DatasetDefinition(BaseConfig):
    """Define how to load a dataset"""

    path: str = Field(description="Local path to a JSONL file in the aimfttk data format")


class WeightedDatasetDefinition(DatasetDefinition):
    """Define a dataset, with a weight for sampling"""

    sampling_weight: float = 1.0


class DataInput(BaseConfig):
    """Base config for data input"""

    type: DataInputType
    data_type: str = Field(
        default="ChatConversation",
        description="Generally, the data_type is automatically set based on the experiment config method.",
    )


class ConcatenationDataInput(DataInput):
    """A simple list of datasets

    These are simply concatenated, the same as sampling all with equal weight.

    The datasets themselves need to be in the aimfttk supported JSONL formats.
    For SFT this means lines:

        {"messages": [{"content": "string", "role": "string"}]}

    For DPO this means lines of:
        {
           "prompt_messages": [{"content": "string", "role": "string"}],
           "chosen_messages": [{"content": "string", "role": "string"}],
            "rejected_messages": [{"content": "string", "role": "string"}]
        }
    """

    type: Literal[DataInputType.CONCATENATION]
    datasets: List[DatasetDefinition] = Field(min_length=1)


class WeightedMixDataInput(DataInput):
    """A list of datasets where each is sampled by a certain weight

    These datasets are interleaved based on the sampling weights. The resulting dataset is fully precomputed, upto
    the point where every single sample in every dataset gets picked. This means that with small sampling weights,
    it can take a lot of draws to see every sample from a dataset and so the resulting dataset can be very large.

    The datasets themselves need to be in the aimfttk supported JSONL formats.
    For SFT this means lines:

        {"messages": [{"content": "string", "role": "string"}]}

    For DPO this means lines of:
        {
           "prompt_messages": [{"content": "string", "role": "string"}],
           "chosen_messages": [{"content": "string", "role": "string"}],
            "rejected_messages": [{"content": "string", "role": "string"}]
        }
    """

    type: Literal[DataInputType.PRECOMPUTE_WEIGHTED_MIX]
    datasets: List[WeightedDatasetDefinition] = Field(min_length=1)
    seed: int = Field(default=19851243, description="Seed for the random number generator for interleaving draws")


class AutoSplitDataInput(DataInput):
    """Automatic validation split from the training data"""

    type: Literal[DataInputType.AUTO_SPLIT]
    ratio: float = Field(default=0.2, description="Ratio of the training data to use for validation")
    seed: int = Field(default=1289525893, description="Seed for the random number generator for splitting")


class NoneDataInput(DataInput):
    """A special type for not using data e.g. in validation"""

    type: Literal[DataInputType.NONE]


class ChatTrainValidConfig(BaseConfig):
    """Training time data configuration

    Always defines some DataInput for training data and can include validation DataInput, though a trivial NoneDataInput
    is also allowed for the validation side.

    Includes data setup configuration.
    """

    training_data: ConcatenationDataInput | WeightedMixDataInput
    validation_data: ConcatenationDataInput | AutoSplitDataInput | NoneDataInput
