# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Data collators"""

import warnings
from typing import Any, Dict, List, Union

import transformers
from transformers import DataCollatorForLanguageModeling


class UnkConvertToEOSCollatorForLM(transformers.DataCollatorForLanguageModeling):
    """Assumes UNK is only used for padding, ignores it in labels and sets it to EOS in input_ids

    This is useful for LLMs which do not have a pad_token set. This pads with UNK token, and then sets all input-side
    UNKs to EOS (this assumes UNK never appears in the normal training data). On the output side, all UNKs get set to
    -100 (ignored by PyTorch in loss computation).

    Note: this is only for mlm=False, but keeping the argument here for compatibility with the parent class interface.
    """

    def __init__(self, tokenizer, mlm=False, *args, **kwargs):
        if tokenizer.unk_token_id != tokenizer.pad_token_id:
            raise ValueError("This collator is only used with unk_token_id == pad_token_id")
        if mlm:
            raise ValueError("This collator is only used with mlm=False")
        super().__init__(*args, mlm=mlm, tokenizer=tokenizer, **kwargs)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        labels = batch["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        batch["input_ids"][batch["input_ids"] == self.tokenizer.pad_token_id] = self.tokenizer.eos_token_id
        return batch


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    # Based on https://github.com/huggingface/trl/blob/18a33ffcd3a576f809b6543a710e989333428bd3/trl/trainer/utils.py#L57C1-L191C
    # which is Apache 2.0 licenced
    # We needed a new one which just detects the start and end of the assistant turns (as opposed to start of assistant
    # or human turns - what if you also have other roles...)
    """Data collator used for completion tasks.

    Sets the output-side data to an 'ignore_index' when on non-assistant turns. This ensures that the loss is only
    calculated on the completion made by the assistant. We may not want the assistant to learn to predict the system
    prompt and user instructions, though implicitly, this still teaches the assistant to use and model the information
    in them.

    Note: This will likely ignore the EOS at the end of the sequence, if it is not used as the assistant turn end.

    Args:
        assistant_start (`List[int]`): the template form that indicates the start of the assistant turn
        assistant_end (`List[int]`): the template form that indicates the end of the assistant turn
        ignore_index (`int`, *optional*, defaults to `-100`):
            The index to use to ignore the initial tokens. (-100 is the PyTorch default, used here by convention)
        mlm (`bool`, *optional*, defaults to `False`): Whether or not to use masked language modeling in the underlying
            `DataCollatorForLanguageModeling` class.
    """

    def __init__(
        self,
        assistant_start: List[int],
        assistant_end: List[int],
        *args,
        ignore_index: int = -100,
        mlm=False,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        self.assistant_start = assistant_start
        self.assistant_end = assistant_end
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            warnings.warn(
                "The pad_token_id and eos_token_id values of this tokenizer are identical. "
                "If you are planning for multi-turn training, "
                "it can result in the model continuously generating questions and answers without eos token. "
                "To avoid this, set the pad_token_id to a different value."
            )
        self.ignore_index = ignore_index

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        # TODO: We should think of some smarter (vectorised?) solution instead of a double python for-loop.
        for i in range(len(batch["labels"])):
            assistant_start_indices = []
            assistant_end_indices = []
            # There are three search states
            #  1. Searching for the start-of-start:
            #     - Here we are not in an assistant turn. We will turn mark each label to be ignored.
            #     - If we find the start token sequence, we start to not mark anything to be ignored and start to search
            #     for the end of the start token sequence
            #  2. Searching for the end-of-start:
            #     - We are in an assistant turn.
            #     - We must also allow this to be the same single token as the start-of-start. (use a new if condition
            #     for this)
            #     - Here we have just detected the start token sequence. We must go past enough indices to make it over
            #     the full start token sequence. We detect when we are at the last token of the start token sequence,
            #     and then start searching for the end of the end token sequence
            #  3. Searching for the end-of-end:
            #     - We are in an assistant turn.
            #     - This may not be the same single token as the end-of-start. (use an elif condition for this)
            #     - We detect when we are at the last token of the end token sequence. Then we go back to searching for
            #     the start of the start token sequence.
            searching_for = "start-of-start"
            for idx in range(len(batch["labels"][i])):
                if searching_for == "start-of-start":
                    # When searching for the start, we must look into the future so that all the assistant start mark
                    # tokens are accounted for. In Python, it is safe to select a stride past the end.
                    if self.assistant_start == (batch["labels"][i][idx : idx + len(self.assistant_start)]).tolist():
                        assistant_start_indices.append(idx)
                        searching_for = "end-of-start"
                    else:
                        # This should be ignored:
                        batch["labels"][i, idx] = self.ignore_index
                # 'if' is crucial here
                if searching_for == "end-of-start":
                    # To account for the cases where start and end sequences are the same, we must make it past the full
                    # start mark token sequence before we start to search for the end.
                    if (
                        idx > len(self.assistant_start)
                        and self.assistant_start
                        == (batch["labels"][i][idx + 1 - len(self.assistant_start) : idx + 1]).tolist()
                    ):
                        searching_for = "end-of-end"
                # 'elif' is needed here
                elif searching_for == "end-of-end":
                    # When searching for the end, we must look into the past so that all the assistant end mark
                    # tokens are accounted for. For this, we should make sure not to use negative indices.
                    if (
                        idx > len(self.assistant_end)
                        and self.assistant_end
                        == (batch["labels"][i][idx + 1 - len(self.assistant_end) : idx + 1]).tolist()
                    ):
                        assistant_end_indices.append(idx)
                        searching_for = "start-of-start"
            if not assistant_start_indices:
                warnings.warn(
                    f"Could not find assistant turn start `{self.tokenizer.decode(self.assistant_start)}` in the "
                    f'following example: {self.tokenizer.decode(batch["input_ids"][i])}. '
                    f"This example will be fully ignored in loss calculation. "
                )
            elif len(assistant_start_indices) != len(assistant_end_indices):
                warnings.warn(
                    f"Different number of assistant turn starts `{self.tokenizer.decode(self.assistant_start)}` "
                    f"and assistant turn ends `{self.tokenizer.decode(self.assistant_end)}`"
                    f'in the following example: {self.tokenizer.decode(batch["input_ids"][i])}. '
                    f"This typically means the conversation has been truncated on the right. "
                )
        return batch


def in_context_tokenize(tokenizer, input_string, prefix="\n"):
    """Tokenizes a string in-context

    i   This is used for completions-only training, where we want to detect the assistant start and end markers.
        See: https://huggingface.co/docs/trl/sft_trainer#using-tokenids-directly-for-responsetemplate

        A string may get tokenized differently when it is in a natural context as opposed to '<s>' or without any prefix.

        This needs a natural prefix string to be chosen. Fortunately '\n' has seemed to work well.
    """
    # Note: this method only works with PreTrainedFastTokenizers
    ids = tokenizer(prefix + input_string, add_special_tokens=False, return_offsets_mapping=True)
    desired_offset = len(prefix)
    # We find the first token after the prefix, and assume that the rest of the tokens make up the actual input_string.
    for i, (offset_start, offset_end) in enumerate(ids["offset_mapping"]):
        if offset_start == desired_offset:
            return ids["input_ids"][i:]
    # Error if we don't find a suitable detection:
    raise RuntimeError("Could not find a suitable token sequence")
