# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import datasets

from aimfttk.data.chat_templates import apply_chat_template_to_preference_data


def get_pref_dataset():
    return datasets.Dataset.from_list(
        [
            {
                "dataset": "pref-1",
                "id": "test-1-1",
                "prompt_messages": [
                    {"role": "system", "content": "A"},
                    {"role": "user", "content": "B"},
                ],
                "chosen_messages": [
                    {"role": "assistant", "content": "C"},
                ],
                "rejected_messages": [
                    {"role": "assistant", "content": "D"},
                ],
            },
            {
                "dataset": "pref-1",
                "id": "test-1-2",
                "prompt_messages": [
                    {"role": "system", "content": "1"},
                    {"role": "user", "content": "2"},
                ],
                "chosen_messages": [
                    {"role": "assistant", "content": "3"},
                ],
                "rejected_messages": [
                    {"role": "assistant", "content": "4"},
                ],
            },
        ]
    )


class MockTokenizerWithChatTemplate:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "".join(message["content"] + "|" for message in messages)

    def __call__(self, text):
        return {"input_ids": [ord(letter) for letter in text]}


def test_apply_chat_template_to_preference_data_keys(datasets_cache_in_tmpdir):
    dataset = get_pref_dataset()
    tokenizer = MockTokenizerWithChatTemplate()
    applied = apply_chat_template_to_preference_data(dataset, tokenizer)
    assert all(key in applied.column_names for key in ["prompt", "chosen", "rejected", "length"])


def test_apply_chat_template_to_preference_data_no_old_keys(datasets_cache_in_tmpdir):
    dataset = get_pref_dataset()
    tokenizer = MockTokenizerWithChatTemplate()
    applied = apply_chat_template_to_preference_data(dataset, tokenizer)
    assert not any(key in applied.column_names for key in ["prompt_messages", "chosen_messages", "rejected_messages"])


def test_apply_chat_template_to_preference_data_length(datasets_cache_in_tmpdir):
    dataset = get_pref_dataset()
    tokenizer = MockTokenizerWithChatTemplate()
    applied = apply_chat_template_to_preference_data(dataset, tokenizer)
    assert applied[0]["length"] == 8
    assert applied[1]["length"] == 8


def test_apply_chat_template_to_preference_data_content(datasets_cache_in_tmpdir):
    dataset = get_pref_dataset()
    tokenizer = MockTokenizerWithChatTemplate()
    applied = apply_chat_template_to_preference_data(dataset, tokenizer)
    assert applied[0]["prompt"] == "A|B|"
    assert applied[0]["chosen"] == "C|"
    assert applied[0]["rejected"] == "D|"
    assert applied[1]["prompt"] == "1|2|"
    assert applied[1]["chosen"] == "3|"
    assert applied[1]["rejected"] == "4|"
