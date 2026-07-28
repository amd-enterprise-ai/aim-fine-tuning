# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Chat template functionality"""

from collections import namedtuple

import datasets

import aimfttk.utils.running_process_opts as running_process_opts
from aimfttk.config.run import ChatTemplateName

ChatTemplate = namedtuple("ChatTemplate", ["jinjastr", "assistant_start", "assistant_end", "special_tokens"])


def get_chat_template(name: ChatTemplateName):
    """Gets a Jinja2 format chat template"""
    # Note, to make some of the jinja2 templates a little more readable, I used an unconventional indentation below, but
    # that required some black and flake8 ignores.
    if name == ChatTemplateName.MISTRAL_WITH_SYSTEM:
        # fmt: off
        # Note: every assistant message ends with an eos_token, so that assistant learns to always end messages that way
        # Note2: The system message <<SYS>> tags are inspired by the Llama template as documented e.g. here:
        #  https://huggingface.co/docs/transformers/main/en/chat_templating#advanced-how-do-chat-templates-work
        # Note3: The assistant response start without a space after [/INST], like in the Mistral Instruct template:
        #  https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/9ab9e76e2b09f9f29ea2d56aa5bd139e4445c59e/tokenizer_config.json#L32
        parts = [
            "{% for message in messages %}",
                "{% if message['role'] == 'user' %}",  # noqa: E131
                    "{{ '[INST] ' + message['content'] + ' [/INST]' }}",  # noqa: E131
                "{% elif message['role'] == 'system' %}",
                    "{{ '<<SYS>>\n' + message['content'] + '\n<</SYS>>\n' }}",
                "{% elif message['role'] == 'assistant' %}",
                    "{{ message['content'] + eos_token + ' ' }}",
                "{% endif %}",
            "{% endfor %}",
        ]
        jinjastr = "".join(parts)  # The point is to glue the template together without any whitespace
        # fmt: on
        # NOTE: The assistant start and end are not perfectly robust! They assume that every instruction is followed by
        # an assistant message, and that every assistant message is preceded by an instruction.
        return ChatTemplate(
            jinjastr=jinjastr,
            assistant_start="[/INST]",
            assistant_end="</s>",
            special_tokens={},
        )
    elif name == ChatTemplateName.CHAT_ML:
        # NOTE: This may have additional new lines compared to the original
        # fmt: off
        parts = [
            "{% for message in messages %}",
                "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}",  # noqa: E131
            "{% endfor %}",
            "{% if add_generation_prompt %}",
                "{{ '<|im_start|>assistant\n' }}",  # noqa: E131
            "{% endif %}",
        ]
        jinjastr = "".join(parts)
        # fmt: on
        return ChatTemplate(
            jinjastr=jinjastr,
            assistant_start="<|im_start|>assistant\n",
            assistant_end="<|im_end|>",
            # This may look weird, but HuggingFace Tokenizers want additional special tokens in this format:
            special_tokens={"additional_special_tokens": ("<|im_start|>", "<|im_end|>")},
        )
    elif name == ChatTemplateName.PORO:
        # The tags <|user|>, <|system|> & <|assistant|> were added to Poro's initial training data but later it
        # was decided to switch to ChatML tags. The tags were used in a small number of samples in pretraining.
        # The tags use the end of sentence token as the identifier of text given by an entity (user, system or assistant).
        parts = [
            "{% for message in messages %}",
            "{% if message['role'] == 'user' %}",
            "{{ '<|user|>\n' + message['content'] + eos_token }}",
            "{% elif message['role'] == 'system' %}",
            "{{ '<|system|>\n' + message['content'] + eos_token }}",
            "{% elif message['role'] == 'assistant' %}",
            "{{ '<|assistant|>\n'  + message['content'] + eos_token }}",
            "{% endif %}",
            "{% if loop.last and add_generation_prompt %}",
            "{{ '<|assistant|>\n' }}",
            "{% endif %}",
            "{% endfor %}",
        ]
        jinjastr = "".join(parts)
        # fmt: on
        return ChatTemplate(
            jinjastr=jinjastr,
            assistant_start="<|assistant|>\n",
            assistant_end="</s>",
            # This may look weird, but HuggingFace Tokenizers want additional special tokens in this format:
            special_tokens={"additional_special_tokens": ("<|user|>", "<|assistant|>")},
        )
    elif name == ChatTemplateName.SIMPLIFIED_LLAMA31:
        # fmt: off
        parts = [
            "{{- bos_token }}",
            "{%- for message in messages %}",
                "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}",  # noqa: E131
            "{%- endfor %}",
            "{%- if add_generation_prompt %}",
            "{{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}",
            "{%- endif %}",
        ]
        jinjastr = "".join(parts)
        # fmt: on
        return ChatTemplate(
            jinjastr=jinjastr,
            assistant_start="<|start_header_id|>assistant<|end_header_id|>\n\n",
            assistant_end="<|eot_id|>",
            special_tokens={"additional_special_tokens": ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")},
        )
    elif name == ChatTemplateName.KEEP_ORIGINAL:
        return None
    else:
        raise ValueError(f"Unknown chat-template: {name}")


def tokenize_with_chat_template(dataset, tokenizer, truncation=False, max_len=None):
    """Applies the chat template that is stored in the tokenizer to each example in a dataset"""

    def _apply_chat_template(example, tokenizer=tokenizer, truncation=truncation, max_len=max_len):
        """Actually does the templating, adds 'text' and 'length'

        Keeps tokenizer in scope as default argument
        """
        tokenized = tokenizer.apply_chat_template(
            example["messages"], tokenize=True, truncation=truncation, max_length=max_len
        )
        tokenized["length"] = len(tokenized["input_ids"])
        return tokenized

    map_kwargs = {}
    if isinstance(dataset, datasets.Dataset):
        map_kwargs["num_proc"] = running_process_opts.num_preprocess_workers
    return dataset.map(_apply_chat_template, remove_columns=dataset.column_names, **map_kwargs)


def apply_chat_template_to_preference_data(dataset, tokenizer):
    """Applies the chat template that is stored in the tokenizer to each example in a dataset

    The preference data is expected to have the keys 'prompt_messages', 'chosen_messages' and 'rejected_messages'.
    The length is added based on the total length (in tokens) based on (prompt + chosen + rejected).
    Note that although the total length is computed in input_ids, the output of this function is text, as that is the
    format that TRL takes.
    """

    def _apply_chat_template(example, tokenizer=tokenizer):
        """Actually does the templating, adds 'prompt', 'chosen', 'rejected'

        The length is added based on the total length (prompt + chosen + rejected).
        Keeps tokenizer in scope as default argument.
        """
        prompt = tokenizer.apply_chat_template(example["prompt_messages"], tokenize=False)
        # First we use the concatenated messages lists (prompt + chosen, prompt + rejected), so that any chat template
        # processing regarding e.g. start of sentence gets applied just once.
        chosen_with_prompt = tokenizer.apply_chat_template(
            example["prompt_messages"] + example["chosen_messages"], tokenize=False
        )
        rejected_with_prompt = tokenizer.apply_chat_template(
            example["prompt_messages"] + example["rejected_messages"], tokenize=False
        )
        chosen = chosen_with_prompt[len(prompt) :]
        rejected = rejected_with_prompt[len(prompt) :]
        length = len(tokenizer(prompt + chosen + rejected)["input_ids"])
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "length": length}

    map_kwargs = {}
    if isinstance(dataset, datasets.Dataset):
        map_kwargs["num_proc"] = running_process_opts.num_preprocess_workers
    return dataset.map(_apply_chat_template, remove_columns=dataset.column_names, **map_kwargs)
