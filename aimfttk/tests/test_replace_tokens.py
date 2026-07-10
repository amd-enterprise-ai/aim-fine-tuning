# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from unittest.mock import patch

import pytest

from aimfttk.cli import replace_tokens_cli
from aimfttk.utils.replace_tokens import escape_for_sed, modify_tokenizer_files


@pytest.mark.parametrize(
    "argumentline",
    [
        ["tokenizer", "outpath", "--from", "blah"],
        ["tokenizer", "outpath", "--to", "blah"],
        ["tokenizer", "outpath", "--to", "blah"],
        [
            "tokenizer",
            "outpath",
        ],
        ["tokenizer", "outpath", "--from", "a", "--to", "b", "--from", "c"],
    ],
)
@patch("aimfttk.cli.AutoTokenizer")
@patch("aimfttk.cli.modify_tokenizer_files")
def test_replace_tokens_cli_unpaired_replacements(modify_mock, tokenizer_mock, argumentline):
    with pytest.raises(SystemExit):
        replace_tokens_cli(argumentline)


@patch("aimfttk.cli.AutoTokenizer")
@patch("aimfttk.cli.modify_tokenizer_files")
def test_replace_tokens_cli_check_replacements(modify_mock, tokenizer_mock):
    argumentline = [
        "tokenizer",
        "outpath",
        "--from",
        "a",
        "--to",
        "b",
        "--from",
        "c",
        "--from",
        "e",
        "--to",
        "d",
        "--to",
        "f",
    ]
    replace_tokens_cli(argv=argumentline)
    modify_mock.assert_called_with("outpath", replacements={"a": "b", "c": "d", "e": "f"})


@pytest.mark.parametrize("inp,out", [("Normal string", "Normal string"), ("String/With/Slash", r"String\/With\/Slash")])
def test_escape_for_sed(inp, out):
    assert escape_for_sed(inp) == out


@patch("aimfttk.utils.replace_tokens.subprocess.run")
def test_modify_tokens(subp_run_mock, tmpdir):
    # Make a file for the function to find (just easier to mock like this)
    with open(tmpdir / "tokenizer_config.json", "w") as fout:
        fout.write("")
    modify_tokenizer_files(tmpdir, replacements={"<|assistant|>": "<|im_start|>"})
    subp_run_mock.assert_called_with(
        ["sed", "-i", "-e", "s/<|assistant|>/<|im_start|>/g", str(tmpdir / "tokenizer_config.json")]
    )


@patch("aimfttk.utils.replace_tokens.subprocess.run")
def test_modify_tokens_raises_when_no_files(subp_run_mock, tmpdir):
    with pytest.raises(FileNotFoundError):
        modify_tokenizer_files(tmpdir, replacements={"<|assistant|>": "<|im_start|>"})
    assert not subp_run_mock.called
