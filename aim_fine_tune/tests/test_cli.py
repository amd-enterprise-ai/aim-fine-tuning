# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase, main, mock

import yaml
from aim_runtime.accelerator_detector import AcceleratorDetectionResult
from aim_runtime.gpu_detector import GPUInfo
from aim_runtime.object_model import AcceleratorType
from click.testing import CliRunner

from aim_fine_tune.cli import app, build_train_request
from aim_fine_tune.download import DownloadRequest
from aim_fine_tune.models import TrainCliValues
from aim_fine_tune.recipe_selection import RecipeDefaults, list_recipe_summaries, select_recipe_defaults

FIXTURE_RECIPE_ID = "aimfttk/sft/google-gemma-3-27b-it/lora/1gpu/default"
FIXTURE_MODEL_ID = "google/gemma-3-27b-it"


def _fake_detection(device_ids: list[str]) -> AcceleratorDetectionResult:
    return AcceleratorDetectionResult(
        accelerator_type=AcceleratorType.GPU,
        accelerator_count=len(device_ids),
        gpu_info=[
            GPUInfo(
                device_id=device_id, model=None, vram_total=0, vram_used=0, gfx_utilization=0.0, mem_utilization=0.0
            )
            for device_id in device_ids
        ],
    )


def _write_recipe_fixture(
    recipe_path: Path,
    *,
    recipe_id: str = FIXTURE_RECIPE_ID,
    status: str = "preview",
    backend: str = "aimfttk",
    model_id: str = FIXTURE_MODEL_ID,
    aim_model_id: str | None = None,
    aim_id: str = FIXTURE_MODEL_ID,
    is_adapter: bool = True,
    compatible_accelerators: list[str] | None = None,
    defaults: dict[str, object] | None = None,
    engine_args: dict[str, object] | None = None,
) -> Path:
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "recipeId": recipe_id,
        "status": status,
        "backend": backend,
        "modelId": model_id,
        "aimId": aim_id,
        "method": "SFT",
        "isAdapter": is_adapter,
        "acceleratorCount": 1,
        "compatibleAccelerators": compatible_accelerators or ["74a1"],
    }
    if aim_model_id is not None:
        metadata["aimModelId"] = aim_model_id
    recipe: dict[str, object] = {
        "metadata": metadata,
        "model": model_id,
        "defaults": defaults or {},
        "engine_args": engine_args or {},
        "aimManifest": {"modelId": model_id, "aimId": aim_id},
    }
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
    return recipe_path


def _minimal_training_env(workspace: Path) -> dict[str, str]:
    model_dir = workspace / "model"
    train_file = workspace / "train.jsonl"
    output_dir = workspace / "outputs"
    model_dir.mkdir()
    train_file.write_text('{"messages": [{"role": "user", "content": "hi"}]}\n', encoding="utf-8")
    return {
        "AIMFT_MODEL_PATH": str(model_dir),
        "AIMFT_TRAIN_DATA_PATH": str(train_file),
        "AIMFT_OUTPUT_PATH": str(output_dir),
        "AIM_MODEL_ID": FIXTURE_MODEL_ID,
    }


class TrainCommandTests(TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def test_list_recipes_defaults_to_json_with_public_fields(self) -> None:
        with self.runner.isolated_filesystem():
            recipes_root = Path.cwd() / "recipes"
            _write_recipe_fixture(
                recipes_root / "aimfttk" / "google" / "gemma-3-27b-it.yaml",
                defaults={"epochs": 1, "seed": 42, "lora_rank": 64},
            )

            result = self.runner.invoke(app, ["list-recipes", "--recipes-root", str(recipes_root)])

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output[result.output.find("{") :])
        self.assertEqual(len(payload["recipes"]), 1)

        recipe = payload["recipes"][0]
        self.assertEqual(recipe["recipeId"], FIXTURE_RECIPE_ID)
        self.assertEqual(recipe["status"], "preview")
        self.assertEqual(recipe["task"], "SFT")
        self.assertEqual(recipe["aimId"], FIXTURE_MODEL_ID)
        self.assertEqual(recipe["modelId"], FIXTURE_MODEL_ID)
        self.assertEqual(recipe["defaults"], {"epochs": 1, "seed": 42, "loraRank": 64})
        self.assertNotIn("recipe_id", recipe)
        self.assertNotIn("lora_rank", recipe["defaults"])
        self.assertEqual(
            set(recipe["metadata"]),
            {"compatibleAccelerators", "acceleratorCount", "isAdapter", "engine", "isCompatible"},
        )
        self.assertNotIn("compatible_accelerators", recipe["metadata"])
        self.assertNotIn("backend", recipe["metadata"])
        self.assertNotIn("sizing", recipe["metadata"])

    def test_pretty_recipe_listing_is_still_available(self) -> None:
        with self.runner.isolated_filesystem():
            recipes_root = Path.cwd() / "recipes"
            _write_recipe_fixture(
                recipes_root / "aimfttk" / "google" / "gemma-3-27b-it.yaml",
                defaults={"epochs": 1},
            )
            recipes = list_recipe_summaries(
                recipes_root=recipes_root,
                accelerator_detector=_fake_detection(["74a1"]),
                compatible_only=True,
            )

            with mock.patch("aim_fine_tune.cli.list_recipe_summaries", return_value=recipes):
                result = self.runner.invoke(app, ["list-recipes", "--compatible-only", "--output-format", "pretty"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Found 1 recipe(s).", result.output)
        self.assertIn(f"{FIXTURE_RECIPE_ID} [compatible]", result.output)
        self.assertIn("defaults:", result.output)

    def test_list_recipes_filters_by_aim_id(self) -> None:
        with self.runner.isolated_filesystem():
            recipes_root = Path.cwd() / "recipes"
            _write_recipe_fixture(
                recipes_root / "aimfttk" / "google" / "gemma-3-27b-it.yaml",
                model_id="google/gemma-3-27b-it-hf",
                aim_model_id="google/gemma-3-27b-it-aim-model",
                aim_id=FIXTURE_MODEL_ID,
                defaults={"epochs": 1},
            )
            _write_recipe_fixture(
                recipes_root / "aimfttk" / "qwen" / "qwen-3-32b.yaml",
                recipe_id="aimfttk/sft/qwen-3-32b/lora/1gpu/default",
                model_id=FIXTURE_MODEL_ID,
                aim_model_id=FIXTURE_MODEL_ID,
                aim_id="Qwen/Qwen3-32B",
                defaults={"epochs": 2},
            )

            result = self.runner.invoke(
                app,
                ["list-recipes", "--recipes-root", str(recipes_root), "--aim-id", FIXTURE_MODEL_ID],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output[result.output.find("{") :])
        self.assertEqual([recipe["recipeId"] for recipe in payload["recipes"]], [FIXTURE_RECIPE_ID])

    def test_recipe_selection_prefers_validated_match_over_preview(self) -> None:
        with self.runner.isolated_filesystem():
            recipes_root = Path.cwd() / "recipes"
            _write_recipe_fixture(
                recipes_root / "vendor" / "a-preview.yaml",
                recipe_id="aimfttk/sft/vendor-test-model/adapter/1gpu/preview",
                model_id="vendor/test-model",
                aim_id="vendor/test-model",
                status="preview",
                defaults={"epochs": 1},
            )
            validated_recipe_path = _write_recipe_fixture(
                recipes_root / "vendor" / "z-validated.yaml",
                recipe_id="aimfttk/sft/vendor-test-model/adapter/1gpu/validated",
                model_id="vendor/test-model",
                aim_id="vendor/test-model",
                status="validated",
                defaults={"epochs": 2},
            )

            recipe_defaults = select_recipe_defaults(
                backend="aimfttk",
                aim_model_id="vendor/test-model",
                num_accelerators=1,
                recipes_root=recipes_root,
                accelerator_detector=_fake_detection(["74a1"]),
            )

        self.assertIsNotNone(recipe_defaults)
        assert recipe_defaults is not None
        self.assertEqual(recipe_defaults.path, validated_recipe_path)
        self.assertEqual(recipe_defaults.epochs, 2)

    def test_build_train_request_applies_recipe_defaults_and_env_overrides(self) -> None:
        with self.runner.isolated_filesystem():
            workspace = Path.cwd()
            env = _minimal_training_env(workspace) | {
                "AIMFT_USE_ADAPTER": "true",
                "AIMFT_EPOCHS": "3",
            }
            recipe_defaults = RecipeDefaults(
                path=Path("fixtures") / "recipe.yaml",
                model_id=FIXTURE_MODEL_ID,
                num_accelerators=1,
                use_adapter=True,
                epochs=1,
                seed=99,
                learning_rate=0.000005,
                lora_dropout=0.05,
                lora_rank=64,
                backend_options={"distributed_type": "auto-deepspeed-stage1"},
            )

            with mock.patch("aim_fine_tune.cli.select_recipe_defaults", return_value=recipe_defaults):
                request = build_train_request(TrainCliValues(), env)

        self.assertTrue(request.metadata.is_adapter)
        self.assertEqual(request.defaults.epochs, 3)
        self.assertEqual(request.defaults.seed, 99)
        self.assertEqual(request.defaults.learning_rate, 0.000005)
        self.assertEqual(request.defaults.lora_dropout, 0.05)
        self.assertEqual(request.defaults.lora_rank, 64)
        self.assertEqual(request.engine_args["distributed_type"], "auto-deepspeed-stage1")

    def test_train_command_renders_aimfttk_workspace(self) -> None:
        with self.runner.isolated_filesystem():
            workspace = Path.cwd()
            env = _minimal_training_env(workspace) | {
                "AIMFT_USE_ADAPTER": "true",
                "AIMFT_LORA_ALPHA": "32",
                "AIMFT_LORA_DROPOUT": "0.1",
                "AIMFT_LORA_RANK": "16",
                "AIMFT_EPOCHS": "2",
                "AIMFT_SEED": "123",
            }

            with mock.patch("aim_fine_tune.cli.select_recipe_defaults", return_value=None):
                result = self.runner.invoke(app, ["train"], env=env)

            output_dir = workspace / "outputs"
            runtime_dir = output_dir / "runtime"
            request = json.loads((runtime_dir / "train-request.json").read_text(encoding="utf-8"))
            launch_spec = json.loads((runtime_dir / "launch-spec.json").read_text(encoding="utf-8"))
            run_script = (runtime_dir / "run.sh").read_text(encoding="utf-8")
            finetuning_config = (output_dir / "configs" / "finetuning_config.yaml").read_text(encoding="utf-8")

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(set(request), {"metadata", "data", "defaults", "engineArgs"})
        self.assertEqual(request["metadata"]["recipeId"], None)
        self.assertEqual(request["metadata"]["backend"], "aimfttk")
        self.assertEqual(request["metadata"]["method"], "SFT")
        self.assertEqual(request["metadata"]["modelId"], FIXTURE_MODEL_ID)
        self.assertEqual(request["metadata"]["aimId"], FIXTURE_MODEL_ID)
        self.assertTrue(request["metadata"]["isAdapter"])
        self.assertEqual(request["metadata"]["acceleratorCount"], 1)
        self.assertNotIn("model_path", request["metadata"])
        self.assertNotIn("output_path", request["metadata"])
        self.assertEqual(request["data"]["modelPath"], str(workspace / "model"))
        self.assertEqual(request["data"]["trainDataPath"], str(workspace / "train.jsonl"))
        self.assertIsNone(request["data"]["validationDataPath"])
        self.assertEqual(request["data"]["outputPath"], str(workspace / "outputs"))
        self.assertEqual(request["defaults"]["epochs"], 2)
        self.assertEqual(request["defaults"]["seed"], 123)
        self.assertEqual(request["defaults"]["loraAlpha"], 32)
        self.assertEqual(request["defaults"]["loraDropout"], 0.1)
        self.assertEqual(request["defaults"]["loraRank"], 16)
        self.assertNotIn("model_path", request["data"])
        self.assertNotIn("lora_alpha", request["defaults"])
        self.assertIsInstance(request["engineArgs"], dict)
        self.assertEqual(launch_spec["backend"], "aimfttk")
        self.assertIn("entrypoint.sh", run_script)
        self.assertIn('peft_type: "LORA"', finetuning_config)
        self.assertIn("lora_alpha: 32.0", finetuning_config)
        self.assertIn("lora_dropout: 0.1", finetuning_config)

    def test_build_train_request_defaults_lora_dropout_to_zero_for_adapters(self) -> None:
        with self.runner.isolated_filesystem():
            workspace = Path.cwd()
            env = _minimal_training_env(workspace) | {
                "AIMFT_USE_ADAPTER": "true",
            }

            with mock.patch("aim_fine_tune.cli.select_recipe_defaults", return_value=None):
                request = build_train_request(TrainCliValues(), env)

        self.assertEqual(request.defaults.lora_dropout, 0.0)

    def test_train_command_merges_engine_args(self) -> None:
        with self.runner.isolated_filesystem():
            workspace = Path.cwd()
            env = _minimal_training_env(workspace) | {
                "AIMFT_ENGINE_ARGS": json.dumps(
                    {
                        "finetuning_config": {
                            "training_args": {"seed": 7},
                            "run_conf": {"resume_from_checkpoint": "never"},
                        }
                    }
                ),
            }

            with mock.patch("aim_fine_tune.cli.select_recipe_defaults", return_value=None):
                result = self.runner.invoke(app, ["train"], env=env)

            runtime_dir = workspace / "outputs" / "runtime"
            finetuning_config = (workspace / "outputs" / "configs" / "finetuning_config.yaml").read_text(
                encoding="utf-8"
            )
            request = json.loads((runtime_dir / "train-request.json").read_text(encoding="utf-8"))

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("seed: 7", finetuning_config)
        self.assertIn('resume_from_checkpoint: "never"', finetuning_config)
        self.assertNotIn("finetuning_config", request["engineArgs"])
        finetuning_config_args = request["engineArgs"]["finetuningConfig"]
        self.assertEqual(finetuning_config_args["trainingArgs"]["seed"], 7)
        self.assertEqual(finetuning_config_args["runConf"]["resumeFromCheckpoint"], "never")

    def test_debug_download_command_builds_download_request(self) -> None:
        with self.runner.isolated_filesystem():
            with mock.patch(
                "aim_fine_tune.download.run_download_request",
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

    def test_execute_mode_runs_backend_command(self) -> None:
        with self.runner.isolated_filesystem():
            workspace = Path.cwd()
            env = _minimal_training_env(workspace)
            recipe_defaults = RecipeDefaults(
                path=Path("fixtures") / "recipe.yaml",
                model_id=FIXTURE_MODEL_ID,
                num_accelerators=1,
                use_adapter=False,
            )

            with (
                mock.patch("aim_fine_tune.cli.subprocess.run", return_value=mock.Mock(returncode=0)) as run_mock,
                mock.patch("aim_fine_tune.cli.select_recipe_defaults", return_value=recipe_defaults),
            ):
                result = self.runner.invoke(app, ["train", "--execute"], env=env)

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            run_mock.call_args.args[0],
            ["/bin/bash", str(workspace / "outputs" / "configs" / "entrypoint.sh")],
        )


if __name__ == "__main__":
    main()
