# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from accelerate import PartialState


def resolve_batchsize_and_accumulation(total_batch_size: int, max_batch_size_per_device: int):
    """Resolve the per-device batch size and number of gradient accumulation steps

    Total batch size is the effective batch size for the complete training run. It is equal to
    number of processes * per-device batch size * accumulation.

    The maximum batch size per device is the maximum batch size that can be accommodated on a single device.
    This mostly limited by the memory capacity of the device.
    """
    num_processes = PartialState().num_processes
    if total_batch_size % num_processes != 0:
        raise ValueError("Total batch size must be divisible by the number of processes.")
    per_device_batch_size = total_batch_size // num_processes
    accumulation = 1
    while per_device_batch_size > max_batch_size_per_device:
        accumulation += 1
        if total_batch_size % (num_processes * accumulation) != 0:
            continue
        per_device_batch_size = total_batch_size // (num_processes * accumulation)
    return per_device_batch_size, accumulation
