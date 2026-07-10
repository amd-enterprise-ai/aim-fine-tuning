# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from aimfttk.config.base import BaseConfig
from aimfttk.config.run import ModelArguments


class InferenceModelConfig(BaseConfig):
    """Configure model for inference time"""

    model: str  # Model HuggingFace ID, or path
    model_args: ModelArguments = ModelArguments()
    adapter: str | None = None  # Adapter HuggingFace ID, or path, or None for no adapter at all

    # tokenizer is a HuggingFace ID, or path, or None to use the one associated with the adapter (if specified) or model
    # in case adapter is not specified.
    tokenizer: None | str = None

    def model_post_init(self, __context):
        # Assign the model name/path for loading the tokenizer if not explicitly supplied
        if self.tokenizer is None:
            if self.adapter is not None:
                self.tokenizer = self.adapter
            else:
                self.tokenizer = self.model
