# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import json

import pytest
from pydantic import ValidationError

from aimfttk.config.data import DatasetDefinition
from aimfttk.data.data_types import ChatConversation, DirectPreference
from aimfttk.data.dataset import make_dataset


def common_setup(tmpdir):
    """Common setup makes a JSONL file and a corresponding DatasetDefinition and sets HF cache"""

    test_examples = [
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
    test_data_path = tmpdir / "test.jsonl"
    with open(test_data_path, "w") as fo:
        for example in test_examples:
            print(json.dumps(example), file=fo)
    dataconf = DatasetDefinition(path=str(test_data_path))
    return dataconf


def test_dataset_random_access_by_default(tmpdir, datasets_cache_in_tmpdir):
    # In this test we actually make a JSONL file and load it
    dataconf = common_setup(tmpdir)
    dataset = make_dataset(dataconf, data_type=ChatConversation)
    # Allows random access by default:
    assert dataset[1]["messages"][0]["content"] == "sed do eiusmod tempor"


def test_coerces_to_chatconversation(tmpdir, datasets_cache_in_tmpdir):
    # In this test we actually make a JSONL file and load it
    dataconf = common_setup(tmpdir)
    dataset = make_dataset(dataconf, data_type=ChatConversation)
    assert set(dataset.column_names) == set(ChatConversation.model_fields.keys())


def test_coercing_fails_for_wrong_type(tmpdir, datasets_cache_in_tmpdir):
    dataconf = common_setup(tmpdir)
    with pytest.raises(ValidationError):
        make_dataset(dataconf, data_type=DirectPreference)


def test_dataset_no_random_access_when_iterable(tmpdir, datasets_cache_in_tmpdir):
    # In this test we actually make a JSONL file and load it
    dataconf = common_setup(tmpdir)
    dataset = make_dataset(dataconf, data_type=ChatConversation, iterable=True)
    with pytest.raises(NotImplementedError):
        assert dataset[1]["messages"][0]["content"] == "sed do eiusmod tempor"


def test_dataset_iterabledataset_through(tmpdir, datasets_cache_in_tmpdir):
    # In this test we actually make a JSONL file and load it
    dataconf = common_setup(tmpdir)
    dataset = make_dataset(dataconf, data_type=ChatConversation)
    iterator = iter(dataset)
    assert next(iterator)["messages"][0]["content"] == "lorem ipsum"
    assert next(iterator)["messages"][0]["content"] == "sed do eiusmod tempor"
    with pytest.raises(StopIteration):
        next(iterator)


def test_dataset_non_existant_raises(tmpdir, datasets_cache_in_tmpdir):
    dataconf = DatasetDefinition(path="non-existant-path.jsonl")
    with pytest.raises(FileNotFoundError):
        make_dataset(dataconf, data_type=ChatConversation)
