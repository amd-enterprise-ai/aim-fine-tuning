# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from unittest.mock import patch

import datasets
import pytest

from aimfttk.config.data import (
    AutoSplitDataInput,
    ConcatenationDataInput,
    DatasetDefinition,
    NoneDataInput,
    WeightedMixDataInput,
)
from aimfttk.data import handle_auto_split
from aimfttk.data.setup import _handle_dataset_relative_weighting, setup_datainput


def mock_make_dataset(dataconf: DatasetDefinition, data_type, iterable=False):
    del data_type  # Unused in the mock
    # add test dataset:
    testdata1 = datasets.Dataset.from_list(
        [
            {
                "dataset": "test-1",
                "id": "test-1-1",
                "messages": [
                    {"role": "system", "content": "lorem ipsum"},
                    {"role": "user", "content": "dolor sit amet"},
                    {"role": "assistant", "content": "consectetur adipiscing elit"},
                ],
            },
            {
                "dataset": "test-1",
                "id": "test-1-2",
                "messages": [
                    {"role": "system", "content": "sed do eiusmod tempor"},
                    {"role": "user", "content": "incididunt ut labore"},
                    {"role": "assistant", "content": "et dolore magna aliqua"},
                ],
            },
        ]
    )
    testdata2 = datasets.Dataset.from_list(
        [
            {
                "dataset": "test-2",
                "id": "test-2-1",
                "messages": [
                    {"role": "system", "content": "Ut enim ad"},
                    {"role": "user", "content": "minim veniam"},
                    {"role": "assistant", "content": "quis nostrud exercitation"},
                ],
            },
            {
                "dataset": "test data",
                "id": "test-2-2",
                "messages": [
                    {"role": "system", "content": "ullamco laboris nisi"},
                    {"role": "user", "content": "ut aliquip ex"},
                    {"role": "assistant", "content": "ea commodo consequat"},
                ],
            },
        ]
    )
    if iterable:
        testdata1 = testdata1.to_iterable_dataset()
        testdata2 = testdata2.to_iterable_dataset()
    if dataconf.path == "testdata1.jsonl":
        return testdata1
    elif dataconf.path == "testdata2.jsonl":
        return testdata2
    else:
        raise FileNotFoundError()


@patch("aimfttk.data.setup.make_dataset", new=mock_make_dataset)
def test_setup_precompute_weightedmix_datainput():
    conf = WeightedMixDataInput(
        **{
            "type": "PRECOMPUTE_WEIGHTED_MIX",
            "datasets": [
                {
                    "path": "testdata1.jsonl",
                },
                {
                    "path": "testdata2.jsonl",
                },
            ],
        }
    )
    datainput = setup_datainput(conf)
    # It gets us a map-stle dataset:
    assert isinstance(datainput, datasets.Dataset)
    # When weighted mix has no weights, it will be in exact alternating order:
    iterator = iter(datainput)
    assert next(iterator)["id"] == "test-1-1"
    assert next(iterator)["id"] == "test-2-1"
    assert next(iterator)["id"] == "test-1-2"
    assert next(iterator)["id"] == "test-2-2"
    # And the input stops all have been seen:
    with pytest.raises(StopIteration):
        next(iterator)


@patch("aimfttk.data.setup.make_dataset", new=mock_make_dataset)
def test_setup_concatenation_datainput():
    conf = ConcatenationDataInput(
        **{
            "type": "CONCATENATION",
            "datasets": [
                {
                    "path": "testdata1.jsonl",
                },
                {
                    "path": "testdata2.jsonl",
                },
            ],
        }
    )
    datainput = setup_datainput(conf)
    # It gets us what format the data is in, so Dataset in this case
    assert isinstance(datainput, datasets.Dataset)
    # Concatenation will be in exact order:
    iterator = iter(datainput)
    assert next(iterator)["id"] == "test-1-1"
    assert next(iterator)["id"] == "test-1-2"
    assert next(iterator)["id"] == "test-2-1"
    assert next(iterator)["id"] == "test-2-2"
    # And the input stops all have been seen:
    with pytest.raises(StopIteration):
        next(iterator)


def test_setup_none_datainput():
    conf = NoneDataInput(
        **{
            "type": "NONE",
        }
    )
    datainput = setup_datainput(conf)
    assert datainput is None


@patch("aimfttk.data.setup.make_dataset", new=mock_make_dataset)
def test_setup_unknown_dataset():
    conf = ConcatenationDataInput(
        **{
            "type": "CONCATENATION",
            "datasets": [
                {
                    "path": "non-existent-path.jsonl",
                },
            ],
        }
    )
    with pytest.raises(FileNotFoundError):
        setup_datainput(conf)


@patch("aimfttk.data.setup.make_dataset", new=mock_make_dataset)
def test_setup_weightedmix_exact_round_robin():
    conf = WeightedMixDataInput(
        **{
            "type": "PRECOMPUTE_WEIGHTED_MIX",
            "datasets": [
                {
                    "path": "testdata1.jsonl",
                    "sampling_weight": 1.0,
                },
                {
                    "path": "testdata2.jsonl",
                    "sampling_weight": 1.0,
                },
            ],
        }
    )
    datainput = setup_datainput(conf)
    # When weighted mix has 1.0 weights, it will be in exact alternating order:
    iterator = iter(datainput)
    assert next(iterator)["id"] == "test-1-1"
    assert next(iterator)["id"] == "test-2-1"
    assert next(iterator)["id"] == "test-1-2"
    assert next(iterator)["id"] == "test-2-2"
    # And the input stops all have been seen:
    with pytest.raises(StopIteration):
        next(iterator)


@patch("aimfttk.data.setup.make_dataset", new=mock_make_dataset)
def test_setup_weightedmix_relative_weights():
    conf = WeightedMixDataInput(
        **{
            "type": "PRECOMPUTE_WEIGHTED_MIX",
            "datasets": [
                {
                    "path": "testdata1.jsonl",
                    "sampling_weight": 10.0,
                },
                {
                    "path": "testdata2.jsonl",
                    "sampling_weight": 1.0,
                },
            ],
        }
    )
    datainput = setup_datainput(conf)
    # It is possible to iterate:
    iterator = iter(datainput)
    assert next(iterator)


@patch("aimfttk.data.setup.make_dataset", new=mock_make_dataset)
def test_setup_weightedmix_zero_weights():
    conf = WeightedMixDataInput(
        **{
            "type": "PRECOMPUTE_WEIGHTED_MIX",
            "datasets": [
                {
                    "path": "testdata1.jsonl",
                    "sampling_weight": 1.0,
                },
                {
                    "path": "testdata2.jsonl",
                    "sampling_weight": 0.0,
                },
            ],
        }
    )
    datainput = setup_datainput(conf)
    iterator = iter(datainput)
    assert next(iterator)["id"] == "test-1-1"
    assert next(iterator)["id"] == "test-1-2"
    # And the input stops all have been seen:
    with pytest.raises(StopIteration):
        next(iterator)


def test_relative_weighting_maths():
    weights = [4.5, 4.5, 1.0]
    expected = [0.45, 0.45, 0.1]
    assert _handle_dataset_relative_weighting(weights) == expected


def test_handle_auto_split():
    conf = AutoSplitDataInput(
        **{
            "type": "AUTO_SPLIT",
            "ratio": 0.5,
            "seed": 123,
        }
    )
    train_data = mock_make_dataset(DatasetDefinition(path="testdata1.jsonl"), "ChatConversation")
    train_data, valid_data = handle_auto_split(conf, train_data)
    assert len(train_data) == 1
    assert len(valid_data) == 1


def test_handle_auto_split_fails_for_iterable():
    conf = AutoSplitDataInput(
        **{
            "type": "AUTO_SPLIT",
            "ratio": 0.5,
            "seed": 123,
        }
    )
    train_data = mock_make_dataset(DatasetDefinition(path="testdata1.jsonl"), "ChatConversation", iterable=True)
    with pytest.raises(ValueError):
        handle_auto_split(conf, train_data)
