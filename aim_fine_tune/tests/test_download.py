# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from unittest import TestCase, mock

import pytest
from click.testing import CliRunner

from aim_fine_tune.cli import app
from aim_fine_tune.extras.download import DownloadRequest

pytestmark = pytest.mark.download

FIXTURE_MODEL_ID = "google/gemma-3-27b-it"


class DownloadCommandTests(TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_debug_download_command_builds_download_request(self) -> None:
        with self.runner.isolated_filesystem():
            with mock.patch(
                "aim_fine_tune.extras.download.run_download_request",
                return_value={"model_path": "downloads/model", "dataset_path": "downloads/dataset"},
            ) as download_mock:
                result = self.runner.invoke(
                    app,
                    [
                        "debug-download",
                        "--hf-model-id",
                        FIXTURE_MODEL_ID,
                        "--model-output-path",
                        "downloads/model",
                        "--dataset-s3-uri",
                        "s3://datasets-bucket/train-data",
                        "--dataset-output-path",
                        "downloads/dataset",
                        "--s3-endpoint-url",
                        "http://minio:9000",
                        "--s3-access-key-id",
                        "minio-access",
                        "--s3-secret-access-key",
                        "minio-secret",
                    ],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Downloaded model to downloads/model", result.output)
        request = download_mock.call_args.args[0]
        self.assertIsInstance(request, DownloadRequest)
        self.assertEqual(request.hf_model_id, FIXTURE_MODEL_ID)
        self.assertEqual(request.dataset_s3_uri, "s3://datasets-bucket/train-data")

    def test_debug_download_command_downloads_hf_dataset_at_revision(self) -> None:
        with self.runner.isolated_filesystem():
            with mock.patch("huggingface_hub.snapshot_download") as snapshot_mock:
                result = self.runner.invoke(
                    app,
                    ["debug-download", "--dataset-output-path", "downloads/dataset"],
                    env={"AIMFT_HF_DATASET_ID": "amd/aimft-dev", "AIMFT_HF_DATASET_REVISION": "abc123"},
                )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Downloaded dataset to downloads/dataset", result.output)
        snapshot_mock.assert_called_once_with(
            repo_id="amd/aimft-dev",
            repo_type="dataset",
            revision="abc123",
            local_dir="downloads/dataset",
            token=None,
        )

    def test_debug_download_command_rejects_two_dataset_sources(self) -> None:
        result = self.runner.invoke(
            app,
            [
                "debug-download",
                "--hf-dataset-id",
                "amd/aimft-dev",
                "--dataset-s3-uri",
                "s3://datasets-bucket/train-data",
                "--dataset-output-path",
                "downloads/dataset",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Set only one of AIMFT_HF_DATASET_ID and AIMFT_DATASET_S3_URI.", result.output)
