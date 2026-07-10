# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import json

import pytest

from aimfttk.data.tokenizer import wrap_for_save_in_inference_mode


class MockTokenizer:
    def __init__(self):
        self.eos_token = "</s>"
        self.bos_token = "<s>"
        self.add_eos_token = True
        self.add_bos_token = True

    def save_pretrained(self, path):
        with open(path, "w") as fo:
            fo.write(
                json.dumps(
                    {
                        "eos_token": self.eos_token,
                        "bos_token": self.bos_token,
                        "add_eos_token": self.add_eos_token,
                        "add_bos_token": self.add_bos_token,
                    }
                )
            )
        return path


def test_wrap_for_save_in_inference_mode_keeps_default_params(tmpdir):
    tokenizer = MockTokenizer()
    eos_token = "<|im_end|>"
    wrap_for_save_in_inference_mode(tokenizer, eos_token)
    tokenizer.save_pretrained(tmpdir / "tokenizer.json")
    assert tokenizer.eos_token == "</s>"
    assert tokenizer.bos_token == "<s>"
    assert tokenizer.add_bos_token == True  # noqa: E712
    assert tokenizer.add_eos_token == True  # noqa: E712


def test_wrap_for_save_in_inference_mode_keeps_same_params(tmpdir):
    tokenizer = MockTokenizer()
    eos_token = "<|im_end|>"
    wrap_for_save_in_inference_mode(tokenizer, eos_token)
    tokenizer.eos_token = "EOS"
    tokenizer.bos_token = "BOS"
    tokenizer.add_eos_token = False
    tokenizer.add_bos_token = False
    tokenizer.save_pretrained(tmpdir / "tokenizer.json")
    assert tokenizer.eos_token == "EOS"
    assert tokenizer.bos_token == "BOS"
    assert tokenizer.add_bos_token == False  # noqa: E712
    assert tokenizer.add_eos_token == False  # noqa: E712


def test_wrap_for_save_in_inference_mode_saves_inference_params(tmpdir):
    tokenizer = MockTokenizer()
    eos_token = "<|im_end|>"
    wrap_for_save_in_inference_mode(tokenizer, eos_token)
    tokenizer.save_pretrained(tmpdir / "tokenizer.json")
    with open(tmpdir / "tokenizer.json") as fi:
        tokenizer_config = json.loads(fi.read())
    assert tokenizer_config == {
        "eos_token": "<|im_end|>",
        "bos_token": "<s>",
        "add_eos_token": False,
        "add_bos_token": True,
    }


def test_wrap_for_save_in_inference_mode_doesnt_change_bos(tmpdir):
    tokenizer = MockTokenizer()
    eos_token = "<|im_end|>"
    wrap_for_save_in_inference_mode(tokenizer, eos_token)
    tokenizer.eos_token = "EOS"
    tokenizer.bos_token = "BOS"
    tokenizer.add_eos_token = False
    tokenizer.add_bos_token = False
    tokenizer.save_pretrained(tmpdir / "tokenizer.json")
    with open(tmpdir / "tokenizer.json") as fi:
        tokenizer_config = json.loads(fi.read())
    assert tokenizer_config == {
        "eos_token": "<|im_end|>",
        "bos_token": "BOS",
        "add_eos_token": False,
        "add_bos_token": True,
    }


def test_wrap_for_save_in_inference_mode_cannot_wrap_twice(tmpdir):
    tokenizer = MockTokenizer()
    eos_token = "<|im_end|>"
    wrap_for_save_in_inference_mode(tokenizer, eos_token)
    with pytest.raises(RuntimeError):
        wrap_for_save_in_inference_mode(tokenizer, eos_token)


def test_wrap_for_save_in_inference_mode_return_the_same(tmpdir):
    tokenizer = MockTokenizer()
    eos_token = "<|im_end|>"
    wrap_for_save_in_inference_mode(tokenizer, eos_token)
    assert tokenizer.save_pretrained(tmpdir / "tokenizer.json") == tmpdir / "tokenizer.json"
