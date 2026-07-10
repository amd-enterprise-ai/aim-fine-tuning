# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import logging

from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


def handle_checkpoint_resume(option: bool | str, checkpoint_dir):
    """This augments the resume_from_checkpoint functionality of the HuggingFace Trainer, which usually accepts True, False or a checkpoint filepath. Instead we also handle a new 'auto' option here, which automatically fetches the last checkpoint.

    With resume_from_checkpoint='auto', the Trainer will resume from a checkpoint if one is found, and otherwise it will
    start from scratch.
    """
    if isinstance(option, bool):
        return option
    elif option == "auto":
        try:
            last_checkpoint = get_last_checkpoint(checkpoint_dir)
        except FileNotFoundError:
            last_checkpoint = None
        if last_checkpoint is None:
            logger.info("Did not find any existing checkpoint.")
            return False
        else:
            logger.info(f"Found existing checkpoint at {last_checkpoint}.")
            return last_checkpoint
    else:
        return option
