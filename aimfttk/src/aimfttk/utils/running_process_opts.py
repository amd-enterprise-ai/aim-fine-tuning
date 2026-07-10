# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""A special module of options that affect the runtime

Python modules act like singletons and can be accessed simply by importing them.
"""

# The number of workers that should be used for a task like dataset preprocessing
num_preprocess_workers = 4


def setup_running_process_options(args):
    """Sets options from the command line args (or any namespace) to this module"""
    global num_preprocess_workers
    num_preprocess_workers = args.num_preprocess_workers
