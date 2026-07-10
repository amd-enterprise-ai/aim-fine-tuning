# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from unittest.mock import patch

import pytest

from aimfttk.utils.training import resolve_batchsize_and_accumulation


@pytest.mark.parametrize(
    "total_batch_size, max_batch_size_per_device, num_processes, correct_result",
    [
        (8, 4, 1, (4, 2)),
        (8, 4, 2, (4, 1)),
        (8, 4, 4, (2, 1)),
        (8, 4, 8, (1, 1)),
        (8, 8, 1, (8, 1)),
        (8, 8, 8, (1, 1)),
        (8, 10, 8, (1, 1)),
        (8, 10, 2, (4, 1)),
        (8, 10, 1, (8, 1)),
        (8, 3, 2, (2, 2)),
    ],
)
@patch("aimfttk.utils.training.PartialState")
def test_resolve_batchsize_and_accumulation_correct_inputs(
    partial_state_cls, total_batch_size, max_batch_size_per_device, num_processes, correct_result
):
    partial_state_cls.return_value.num_processes = num_processes
    result = resolve_batchsize_and_accumulation(total_batch_size, max_batch_size_per_device)
    assert result == correct_result


@pytest.mark.parametrize(
    "total_batch_size, max_batch_size_per_device, num_processes",
    [
        (
            8,
            4,
            10,
        ),
        (
            8,
            10,
            10,
        ),
        (
            1,
            10,
            2,
        ),
    ],
)
@patch("aimfttk.utils.training.PartialState")
def test_resolve_batchsize_and_accumulation_impossible(
    partial_state_cls, total_batch_size, max_batch_size_per_device, num_processes
):
    partial_state_cls.return_value.num_processes = num_processes
    with pytest.raises(ValueError):
        resolve_batchsize_and_accumulation(total_batch_size, max_batch_size_per_device)
