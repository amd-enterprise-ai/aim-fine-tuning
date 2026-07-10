# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Components of finetuning pipeline"""

import os
import random
import string

import mlflow

from aimfttk.config import DPOExperimentConfig, SFTExperimentConfig


def generate_random_name():
    """Generate a random name"""
    prev_state = random.getstate()
    random.seed()
    random_name = "".join(random.choices(string.ascii_letters + string.digits + "-", k=8))
    random.setstate(prev_state)
    return random_name


def setup_tracking(exp_conf: SFTExperimentConfig | DPOExperimentConfig, args):
    """Setup tracking experiment

    Arguments supplied through command line have precedence over those supplied through config yaml,
    which in turn has precedence over settings provided through environment variables.

    Modifies the config's training_args.report_to
    """
    if args.mlflow_server_uri is not None:
        mlflow.set_tracking_uri(args.mlflow_server_uri)
        mlflow.set_experiment(args.experiment_name)
        mlflow.set_tag("mlflow.runName", generate_random_name())
        os.environ["MLFLOW_FLATTEN_PARAMS"] = "TRUE"
        os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = args.hf_mlflow_log_artifacts
    elif exp_conf.tracking is not None:
        mlflow.set_tracking_uri(exp_conf.tracking.mlflow_server_uri)
        mlflow.set_experiment(exp_conf.tracking.experiment_name)
        mlflow.set_tag("mlflow.runName", generate_random_name())
        os.environ["MLFLOW_FLATTEN_PARAMS"] = "TRUE"
        os.environ["HF_MLFLOW_LOG_ARTIFACTS"] = exp_conf.tracking.hf_mlflow_log_artifacts
    elif "MLFLOW_TRACKING_URI" in os.environ:
        mlflow.set_tag("mlflow.runName", generate_random_name())
    elif "MLFLOW_TRACKING_URI" not in os.environ:
        os.environ["DISABLE_MLFLOW_INTEGRATION"] = "TRUE"

    if os.environ.get("DISABLE_MLFLOW_INTEGRATION", "FALSE") == "TRUE":
        if exp_conf.training_args.report_to == "mlflow":
            exp_conf.training_args.report_to = []
        try:
            exp_conf.training_args.report_to.remove("mlflow")
        except (ValueError, AttributeError):
            # mlflow was not in report_to, nothing to remove
            pass
        if exp_conf.training_args.report_to in ["all", ["all"]]:
            raise ValueError("Cannot report to all when mlflow is disabled")
    else:
        if not ("mlflow" in exp_conf.training_args.report_to or exp_conf.training_args.report_to in ["all", ["all"]]):
            if exp_conf.training_args.report_to is None:
                exp_conf.training_args.report_to = []
            elif isinstance(exp_conf.training_args.report_to, str):
                exp_conf.training_args.report_to = [exp_conf.training_args.report_to]
            exp_conf.training_args.report_to.append("mlflow")
