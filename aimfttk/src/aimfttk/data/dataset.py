# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The data fetchers (Dataset objects)"""

import datasets
from accelerate import PartialState

from aimfttk.config.base import BaseConfig
from aimfttk.config.data import AutoSplitDataInput, DatasetDefinition


def _auto_choose_num_shards(dataset):
    world_size = PartialState().num_processes
    if world_size is None or world_size == 0:
        world_size = 1
    suggested_numshards = 1 if world_size == 1 else len(dataset) // (world_size * 4)  # Arbitrary x4 multiplier
    if suggested_numshards < world_size:
        raise ValueError("Not enough shards!")
    return suggested_numshards


def _coerce(data, data_type: BaseConfig):
    """Coerce the data to the given pydantic data type, then dump it as dict

    This acts as input data validator, enforcing a uniform data format.
    """
    return data_type(**data).model_dump()


def make_dataset(dataconf: DatasetDefinition, data_type: BaseConfig, iterable=False):  # type: ignore
    """Makes a Dataset or IterableDataset based on the given DatasetDefinition

    Will delegate to the correct underlying dataset creation.

    It can be desirable to force the dataset to work in streaming mode (iterable=True)
    NOTE: if iterable=False, we should actually (try to) force the dataset to be map-style, because at least the
    PRECOMPUTE_WEIGHTED_MIX DataInputType depends on that behaviour.
    """
    dataset = datasets.Dataset.from_json(dataconf.path)
    dataset = dataset.map(_coerce, fn_kwargs={"data_type": data_type}, remove_columns=dataset.column_names)
    if iterable:
        dataset = dataset.to_iterable_dataset(num_shards=_auto_choose_num_shards(dataset))
    return dataset


def handle_auto_split(conf: AutoSplitDataInput, data) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Handles automatic validation split of the training dataset

    Returns a tuple with the resulting training data first, validation data second
    """
    if isinstance(data, datasets.IterableDataset):
        raise ValueError("Cannot split an IterableDataset")
    splits = data.train_test_split(test_size=conf.ratio, seed=conf.seed)
    return splits["train"], splits["test"]
