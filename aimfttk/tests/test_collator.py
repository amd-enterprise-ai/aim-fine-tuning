# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import pytest
import torch

from aimfttk.data.collator import DataCollatorForCompletionOnlyLM, UnkConvertToEOSCollatorForLM, in_context_tokenize


class MockUnkConverToEOSTokenizer:
    def __init__(self):
        self.unk_token_id = 0
        self.pad_token_id = 0
        self.eos_token_id = 1

    def pad(self, *args, **kwargs):
        # The unsqueeze opens a phantom batch dimension
        return {"input_ids": torch.tensor([2, 2, 3, self.eos_token_id, self.unk_token_id]).unsqueeze(0)}


def test_unk_convert_to_eos_collator():
    tokenizer = MockUnkConverToEOSTokenizer()
    collator = UnkConvertToEOSCollatorForLM(tokenizer)
    collated = collator.torch_call([{"input_ids": torch.tensor([2, 2, 3, tokenizer.eos_token_id])}])
    # This should replace unk with eos token id on the input and -100 on the labels (output)
    # In these checks the [0] indexes into the phatom batch dimension
    assert torch.equal(
        collated["input_ids"][0], torch.tensor([2, 2, 3, tokenizer.eos_token_id, tokenizer.eos_token_id])
    )
    assert torch.equal(collated["labels"][0], torch.tensor([2, 2, 3, tokenizer.eos_token_id, -100]))


class MockCompletionOnlyTokenizer:
    def __init__(self, assistant_start, assistant_end):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.assistant_start = assistant_start
        self.assistant_end = assistant_end

    def pad(self, *args, **kwargs):
        # The unsqueeze opens a phantom batch dimension
        return {
            "input_ids": torch.tensor(
                [10, 11, 12]
                + self.assistant_start
                + [10, 11, 12]
                + self.assistant_end
                + [10, 11, 12]
                + [self.eos_token_id]
            ).unsqueeze(0)
        }


assistant_detection_testcases = [
    ([2], [3]),
    ([2], [2]),
    ([2, 3], [4, 5]),
    ([2, 3], [2, 3]),
]


@pytest.mark.parametrize("assistant_start,assistant_end", assistant_detection_testcases)
def test_completion_only_collator(assistant_start, assistant_end):
    tokenizer = MockCompletionOnlyTokenizer(assistant_start=assistant_start, assistant_end=assistant_end)
    collator = DataCollatorForCompletionOnlyLM(
        assistant_start=assistant_start, assistant_end=assistant_end, tokenizer=tokenizer
    )
    # The input here does not matter:
    collated = collator.torch_call([{"input_ids": torch.tensor([2, 2, 3, tokenizer.eos_token_id])}])

    # Input side should always be:
    inputside = torch.tensor([10, 11, 12] + assistant_start + [10, 11, 12] + assistant_end + [10, 11, 12] + [1])
    assert torch.equal(collated["input_ids"][0], inputside)  # [0] indexes into phantom batch dimension
    # Output side should have -100 for the non-assistant portions:
    outputside = torch.tensor(
        [-100, -100, -100] + assistant_start + [10, 11, 12] + assistant_end + [-100, -100, -100] + [-100]
    )
    assert torch.equal(collated["labels"][0], outputside)  # [0] indexes into phantom batch dimension


class MockFastTokenizer:
    def __call__(self, *args, **kwargs):
        return {"input_ids": [0, 1, 2, 3], "offset_mapping": [(0, 1), (1, 2), (2, 3), (3, 4)]}


def test_in_context_tokenize():
    tokenizer = MockFastTokenizer()
    ids = in_context_tokenize(tokenizer, input_string="Not used", prefix="\n")  # The length of the prefix matters
    assert ids == [1, 2, 3]
