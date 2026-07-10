# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Dataset setup from data config"""

from datasets import Dataset, concatenate_datasets, interleave_datasets

from aimfttk.config.data import DataInput, DataInputType
from aimfttk.data.data_types import data_type_by_name
from aimfttk.data.dataset import make_dataset


def _handle_dataset_relative_weighting(weights):
    """Handles a list of weights

    In most cases squashes the weights to sum to 1.0, since interleave_datasets wants probabilities.
    If all weights are the default value of 1.0, then will return None, which in turn means exact round-robin.
    """
    # If all the weights are the default value (1.0), then set weights to None, which will exactly round-robin
    if all(weight == 1.0 for weight in weights):
        return None
    if any(weight < 0.0 for weight in weights):
        raise ValueError("Negative weights are not meaningful.")
    total_weight = sum(weights)
    if total_weight == 0.0:
        raise ValueError("Total dataset weight cannot be 0.0")
    return [weight / total_weight for weight in weights]


def setup_datainput(conf: DataInput) -> Dataset | None:
    """
    Sets up the data fetching

    Args:
        conf: Defines type of the dataset.

    Returns:
        Composed dataset.

    Raises:
        ValueError: When unknown data input type is received.
    """
    data_type = data_type_by_name[conf.data_type]
    if conf.type == DataInputType.PRECOMPUTE_WEIGHTED_MIX:
        datasets = []
        weights = []
        for dataconf in conf.datasets:
            # Skip zero weight datasets. interleave_datasets does not skip them, but will instead iterate indefinitely.
            if dataconf.sampling_weight == 0.0:
                continue
            dataset = make_dataset(dataconf, data_type=data_type, iterable=False)
            datasets.append(dataset)
            weights.append(dataconf.sampling_weight)
        weights = _handle_dataset_relative_weighting(weights)
        # Set stopping_strategy to 'all_exhausted' which will oversample.
        data_result = interleave_datasets(
            datasets, probabilities=weights, stopping_strategy="all_exhausted", seed=conf.seed
        )
    elif conf.type == DataInputType.CONCATENATION:
        datasets = []
        for dataconf in conf.datasets:
            dataset = make_dataset(dataconf, data_type=data_type, iterable=False)
            datasets.append(dataset)
        data_result = concatenate_datasets(datasets)
    elif conf.type == DataInputType.NONE:
        return None
    else:
        raise ValueError(f"Unknown configuration type {conf.type}")
    return data_result


def filter_long_examples(data, max_len):
    """Filters out examples that are too long, SFT specific

    Based on the 'input_ids' key, i.e. after tokenization, or 'length' if that exists.
    """

    # Iterable datasets may be lacking the information about column names
    if data.column_names is None:
        data = data._resolve_features()
    if "length" in data.column_names:

        def length_filter(length, max_len=max_len):
            return length <= max_len

        return data.filter(length_filter, input_columns="length")
    else:

        def length_filter(input_ids, max_len=max_len):
            return len(input_ids) <= max_len

        return data.filter(length_filter, input_columns="input_ids")


def sort_longest_first(data):
    """Sorts dataset so that longest examples come first.

    Useful for validation and test data, where this leads to similar length utterances being in the same batch, which in
    turn requires less padding, less wasted computation, and runs faster.

    Longest first is good, because it will give a conservative estimate of how long the evaluation will take and will
    fail faster if running Out-Of-Memory.

    Based on the 'input_ids' key, i.e. after tokenization, or if 'length' key exists, using that.
    """
    # Iterable datasets may be lacking the information about column names
    if data.column_names is None:
        data = data._resolve_features()
    if "length" in data.column_names:
        data = data.sort("length", reverse=True)
        return data
    else:
        # HuggingFace datasets don't have a way to sort on a computed property, so we need to add a temporary column and
        # sort on that.
        tmp_key = "__silogen_auto_added_length"

        def _add_length(example):
            example[tmp_key] = len(example["input_ids"])
            return example

        data = data.map(_add_length)
        data = data.sort(tmp_key, reverse=True)
        data = data.remove_columns(tmp_key)
        return data
