# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import os
from collections import namedtuple
from typing import List
from unittest.mock import patch

from aimfttk.config.base import BaseConfig
from aimfttk.config.experiment import FinetuningTrackingConfig
from aimfttk.utils.tracking import setup_tracking

Arguments = namedtuple("Arguments", ["mlflow_server_uri", "experiment_name", "hf_mlflow_log_artifacts"])


class MockTrainingArguments(BaseConfig):
    """Training Arguments"""

    report_to: str | List[str] = "none"


class ExperimentConfig(BaseConfig):
    """Mock Config"""

    training_args: MockTrainingArguments = MockTrainingArguments()
    tracking: FinetuningTrackingConfig | None = None


def mock_mlflow_set_experiment(experiment_name: str):
    os.environ["TEST_MLFLOW_EXPERIMENT_NAME"] = experiment_name


def mock_mlflow_set_tracking_uri(mlflow_server_uri: str):
    os.environ["TEST_MLFLOW_SERVER_URI"] = mlflow_server_uri


def mock_mlflow_set_tag(tag_name: str, tag_value: str):
    os.environ[tag_name.replace(".", "_")] = tag_value


@patch("mlflow.set_experiment", new=mock_mlflow_set_experiment)
@patch("mlflow.set_tracking_uri", new=mock_mlflow_set_tracking_uri)
@patch("mlflow.set_tag", new=mock_mlflow_set_tag)
def test_setup_tracking_args_over_conf():
    """Check that arguments settings have precedence over config settings"""
    conf = ExperimentConfig(
        tracking={"mlflow_server_uri": "file:///mlruns", "experiment_name": "conf", "hf_mlflow_log_artifacts": "False"}
    )
    args = Arguments("file:///mlruns", "args", "False")

    setup_tracking(conf, args)

    assert os.environ["TEST_MLFLOW_EXPERIMENT_NAME"] == "args"


@patch("mlflow.set_experiment", new=mock_mlflow_set_experiment)
@patch("mlflow.set_tracking_uri", new=mock_mlflow_set_tracking_uri)
@patch("mlflow.set_tag", new=mock_mlflow_set_tag)
def test_setup_tracking_conf_over_env():
    """Check that config settings have precedence over env variables"""
    conf = ExperimentConfig(
        tracking={"mlflow_server_uri": "file:///mlruns", "experiment_name": "conf", "hf_mlflow_log_artifacts": "False"}
    )
    args = Arguments(None, "args", "False")

    os.environ["MLFLOW_TRACKING_URI"] = "file:///mlruns"

    setup_tracking(conf, args)

    assert os.environ["TEST_MLFLOW_EXPERIMENT_NAME"] == "conf"


@patch("mlflow.set_experiment", new=mock_mlflow_set_experiment)
@patch("mlflow.set_tracking_uri", new=mock_mlflow_set_tracking_uri)
@patch("mlflow.set_tag", new=mock_mlflow_set_tag)
def test_setup_tracking_disabled():
    """Check that mlflow is disabled if none of the settings options supplied"""
    conf = ExperimentConfig(tracking=None)
    args = Arguments(None, "args", "False")

    os.environ.pop("MLFLOW_TRACKING_URI", None)

    setup_tracking(conf, args)

    assert os.environ["DISABLE_MLFLOW_INTEGRATION"] == "TRUE"
