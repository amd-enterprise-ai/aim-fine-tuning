# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Utility to modify a tokenizer to change the names of tokens

For example, we may want to change the <|assistant|> and <|user|> tokens to the more general <|im_start|> and
<|im_end|> tokens.
"""

import pathlib
import subprocess


def escape_for_sed(string):
    """Escapes the forward-slash for sed calls that use forward-slash as separator"""
    return string.replace("/", r"\/")


def modify_tokenizer_files(tokenizer_directory, replacements={}):
    """Modifies tokenizer files in-place"""

    tokenizer_directory = pathlib.Path(tokenizer_directory)
    files_to_change = (
        tokenizer_directory / "tokenizer_config.json",
        tokenizer_directory / "tokenizer.json",
        tokenizer_directory / "special_tokens_map.json",
        tokenizer_directory / "added_tokens.json",
    )
    cmd = ["sed", "-i"]
    for token_from, token_to in replacements.items():
        cmd.append("-e")
        cmd.append(f"s/{escape_for_sed(token_from)}/{escape_for_sed(token_to)}/g")
    if not any(path.is_file() for path in files_to_change):
        raise FileNotFoundError(f"No tokenizer files found in {str(tokenizer_directory)}")
    for path in files_to_change:
        if not path.is_file():
            continue
        complete_proc = subprocess.run(cmd + [str(path)])
        complete_proc.check_returncode()  # May raise error
