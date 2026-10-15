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

from aim_fine_tune.backends.base import LaunchSpec, RawStdoutChunk, StdoutClosed
from aim_fine_tune.cli import app, build_train_request
from aim_fine_tune.models import TrainCliValues
from aim_fine_tune.recipe_selection import (
    RECIPE_COMPATIBILITY_FILE_NAME,
    RecipeDefaults,
    list_recipe_summaries,
    load_recipe,
    select_recipe_defaults,
)
from aim_fine_tune.training import build_execution_environment

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
    backend: str = "aimfttk",
    model_id: str = FIXTURE_MODEL_ID,
    aim_model_id: str | None = None,
    aim_id: str = FIXTURE_MODEL_ID,
    method: str = "SFT",
    is_adapter: bool = True,
    defaults: dict[str, object] | None = None,
    engine_args: dict[str, object] | None = None,
) -> Path:
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "recipeId": recipe_id,
        "backend": backend,
        "modelId": model_id,
        "aimId": aim_id,
        "method": method,
        "isAdapter": is_adapter,
        "acceleratorCount": 1,
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


def _write_recipe_compatibility(recipes_root: Path) -> Path:
    recipe_ids = [
        load_recipe(path).metadata.recipe_id
        for path in sorted(recipes_root.rglob("*.yaml"))
        if path.name != RECIPE_COMPATIBILITY_FILE_NAME
    ]
    compatibility = {"recipes": {recipe_id: {"compatibleAccelerators": ["0x74a1"]} for recipe_id in recipe_ids}}
    (recipes_root / RECIPE_COMPATIBILITY_FILE_NAME).write_text(yaml.safe_dump(compatibility), encoding="utf-8")
    return recipes_root


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


def _write_trainer_state(output_path: Path, log_history: list[dict[str, float]]) -> None:
    checkpoint = output_path / "checkpoints" / "checkpoint-50"
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 50, "log_history": log_history}),
        encoding="utf-8",
    )


def _launch_spec(*, env: dict[str, str] | None = None) -> LaunchSpec:
    return LaunchSpec(
        backend="aimfttk",
        command=("/bin/bash", "/configs/entrypoint.sh"),
        env=env or {},
        files={},
    )


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

            _write_recipe_compatibility(recipes_root)

            result = self.runner.invoke(app, ["list-recipes", "--extra-recipes", str(recipes_root)])

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output[result.output.find("{") :])
        self.assertEqual(len(payload["recipes"]), 1)

        recipe = payload["recipes"][0]
        self.assertEqual(recipe["recipeId"], FIXTURE_RECIPE_ID)
        self.assertNotIn("status", recipe)
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
                extra_recipes=_write_recipe_compatibility(recipes_root),
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

            _write_recipe_compatibility(recipes_root)

            result = self.runner.invoke(
                app,
                ["list-recipes", "--extra-recipes", str(recipes_root), "--aim-id", FIXTURE_MODEL_ID],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output[result.output.find("{") :])
        self.assertEqual([recipe["recipeId"] for recipe in payload["recipes"]], [FIXTURE_RECIPE_ID])

    def test_recipe_selection_filters_by_training_method(self) -> None:
        with self.runner.isolated_filesystem():
            recipes_root = Path.cwd() / "recipes"
            sft_recipe_path = _write_recipe_fixture(
                recipes_root / "sft.yaml",
                defaults={"epochs": 1},
            )
            _write_recipe_fixture(
                recipes_root / "other.yaml",
                recipe_id="aimfttk/other/google-gemma-3-27b-it/lora/1gpu/default",
                method="OTHER",
                defaults={"epochs": 2},
            )

            recipe_defaults = select_recipe_defaults(
                backend="aimfttk",
                method="SFT",
                aim_model_id=FIXTURE_MODEL_ID,
                num_accelerators=1,
                extra_recipes=_write_recipe_compatibility(recipes_root),
                accelerator_detector=_fake_detection(["74a1"]),
            )

        self.assertIsNotNone(recipe_defaults)
        assert recipe_defaults is not None
        self.assertEqual(recipe_defaults.path, sft_recipe_path)
        self.assertEqual(recipe_defaults.epochs, 1)

    def test_extra_recipes_add_to_and_replace_builtin_recipes(self) -> None:
        with self.runner.isolated_filesystem():
            builtin_root = Path.cwd() / "builtin"
            _write_recipe_fixture(builtin_root / "gemma.yaml", defaults={"epochs": 1})
            builtin_qwen_path = _write_recipe_fixture(
                builtin_root / "qwen.yaml",
                recipe_id="aimfttk/sft/qwen-qwen3-32b/lora/1gpu/default",
                model_id="Qwen/Qwen3-32B",
                aim_id="Qwen/Qwen3-32B",
            )
            extra_root = Path.cwd() / "extra"
            extra_gemma_path = _write_recipe_fixture(extra_root / "gemma.yaml", defaults={"epochs": 3})

            with mock.patch(
                "aim_fine_tune.recipe_selection.BUILTIN_RECIPES_ROOT", _write_recipe_compatibility(builtin_root)
            ):
                summaries = list_recipe_summaries(
                    extra_recipes=_write_recipe_compatibility(extra_root),
                    accelerator_detector=_fake_detection(["74a1"]),
                )

        self.assertEqual(
            {summary.recipe_id: summary.path for summary in summaries},
            {FIXTURE_RECIPE_ID: extra_gemma_path, "aimfttk/sft/qwen-qwen3-32b/lora/1gpu/default": builtin_qwen_path},
        )
        self.assertEqual(
            next(summary for summary in summaries if summary.recipe_id == FIXTURE_RECIPE_ID).defaults["epochs"], 3
        )

    def test_build_train_request_applies_recipe_defaults_and_env_overrides(self) -> None:
        with self.runner.isolated_filesystem():
            workspace = Path.cwd()
            env = _minimal_training_env(workspace) | {
                "AIMFT_USE_ADAPTER": "true",
                "AIMFT_EPOCHS": "3",
            }
            recipe_defaults = RecipeDefaults(
                path=Path("fixtures") / "recipe.yaml",
                recipe_id=FIXTURE_RECIPE_ID,
                recipe_digest="sha256:fixture",
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
        self.assertIn('length_handling: "truncate"', finetuning_config)
        self.assertNotIn("use_cache", finetuning_config)
        self.assertIn('include_num_input_tokens_seen: "non_padding"', finetuning_config)
        self.assertIn("logging_first_step: true", finetuning_config)
        self.assertNotIn("eval_on_start", finetuning_config)

    def test_train_command_evaluates_on_start_when_validation_data_is_set(self) -> None:
        with self.runner.isolated_filesystem():
            workspace = Path.cwd()
            validation_file = workspace / "valid.jsonl"
            validation_file.write_text('{"messages": [{"role": "user", "content": "hello"}]}\n', encoding="utf-8")
            env = _minimal_training_env(workspace) | {
                "AIMFT_VALIDATION_DATA_PATH": str(validation_file),
            }

            with mock.patch("aim_fine_tune.cli.select_recipe_defaults", return_value=None):
                result = self.runner.invoke(app, ["train"], env=env)

            finetuning_config = (workspace / "outputs" / "configs" / "finetuning_config.yaml").read_text(
                encoding="utf-8"
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("eval_on_start: true", finetuning_config)
        self.assertIn('eval_strategy: "steps"', finetuning_config)

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

    def test_execute_mode_runs_backend_command(self) -> None:
        with self.runner.isolated_filesystem():
            workspace = Path.cwd()
            env = _minimal_training_env(workspace)
            engine_bin = workspace / "engine" / "bin"
            env["PATH"] = str(engine_bin)
            recipe_defaults = RecipeDefaults(
                path=Path("fixtures") / "recipe.yaml",
                recipe_id=FIXTURE_RECIPE_ID,
                recipe_digest="sha256:fixture",
                model_id=FIXTURE_MODEL_ID,
                num_accelerators=1,
                use_adapter=False,
            )
            captured_log = (
                "GPU[0]\t\t: Device ID: 0x74a1\n"
                '{"event":"train_metrics","train_loss":2.5,"step":1,"elapsed_seconds":2.5,"tokens_seen":1000}\n'
                '{"event":"train_metrics","train_loss":1.1,"step":50,"elapsed_seconds":10.0,"tokens_seen":8000}\n'
            )

            def fake_run(command, env, output_pipeline):
                _write_trainer_state(
                    workspace / "outputs",
                    [
                        {
                            "loss": 2.5,
                            "epoch": 0.1,
                            "num_input_tokens_seen": 1000,
                            "train_runtime": 2.5,
                            "train_tokens_per_second": 400.0,
                        },
                        {"eval_loss": 2.4, "epoch": 0.5},
                        {
                            "loss": 1.1,
                            "epoch": 1.0,
                            "num_input_tokens_seen": 8000,
                            "train_runtime": 10.0,
                            "train_tokens_per_second": 800.0,
                        },
                        {"eval_loss": 1.2, "epoch": 1.0},
                    ],
                )
                output_pipeline.publish(RawStdoutChunk(captured_log.encode()))
                output_pipeline.publish(StdoutClosed())
                output_pipeline.flush()
                return mock.Mock(returncode=0, duration_seconds=12.5)

            with (
                mock.patch("aim_fine_tune.training.run_captured_command", side_effect=fake_run) as run_mock,
                mock.patch("aim_fine_tune.cli.select_recipe_defaults", return_value=recipe_defaults),
                mock.patch("aim_fine_tune.training.detected_accelerator_ids", return_value=("0x74a1",)),
            ):
                result = self.runner.invoke(app, ["train", "--execute", "--json-logs"], env=env)

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(
                run_mock.call_args.args[0],
                ["/bin/bash", str(workspace / "outputs" / "configs" / "entrypoint.sh")],
            )
            execution_env = run_mock.call_args.args[1]
            self.assertEqual(execution_env["PATH"], str(engine_bin))
            self.assertEqual(execution_env["ACCELERATE_LOG_LEVEL"], "info")
            self.assertIsNotNone(run_mock.call_args.kwargs["output_pipeline"])
            report_path = workspace / "outputs" / "runtime" / "train-report.json"
            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["backend"], "aimfttk")
            self.assertEqual(report["accelerators"], ["0x74a1"])
            self.assertEqual(report["initialTrainLoss"], 2.5)
            self.assertEqual(report["finalTrainLoss"], 1.1)
            self.assertEqual(report["initialValidationLoss"], 2.4)
            self.assertEqual(report["finalValidationLoss"], 1.2)
            self.assertEqual(report["durationSeconds"], 12.5)
            self.assertEqual(report["averageTokensPerGpuPerSecond"], 800.0)
            self.assertEqual(report["incomplete"], [])
            self.assertIn(
                "train_metrics",
                (workspace / "outputs" / "runtime" / "train.stdout.log").read_text(encoding="utf-8"),
            )
            self.assertEqual(
                (workspace / "outputs" / "runtime" / "train.stderr.log").read_text(encoding="utf-8"),
                "",
            )
            status = json.loads((workspace / "outputs" / "runtime" / "train-status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "COMPLETED")
            self.assertTrue(status["timestamp"].endswith("Z"))
            events = [json.loads(line) for line in result.output.splitlines() if line.strip()]
            metrics = [event for event in events if event.get("event") == "train_metrics"]
            self.assertEqual([event["step"] for event in metrics], [1, 50])
            reports = [event["report"] for event in events if event.get("event") == "train_report"]
            self.assertEqual(reports, [report])
            messages = [event["message"] for event in events if event.get("event") == "log"]
            self.assertTrue(any(message.startswith("Rendered aimfttk launch workspace to") for message in messages))
            self.assertTrue(any(message.startswith("Wrote training report to") for message in messages))
            self.assertIn("Execution completed successfully.", messages)

    def test_execute_mode_writes_report_when_backend_fails(self) -> None:
        with self.runner.isolated_filesystem():
            workspace = Path.cwd()
            env = _minimal_training_env(workspace)
            recipe_defaults = RecipeDefaults(
                path=Path("fixtures") / "recipe.yaml",
                recipe_id=FIXTURE_RECIPE_ID,
                recipe_digest="sha256:fixture",
                model_id=FIXTURE_MODEL_ID,
                num_accelerators=1,
                use_adapter=False,
            )

            def fake_run(command, env, output_pipeline):
                _write_trainer_state(workspace / "outputs", [{"loss": 9.0, "epoch": 0.1}])
                return mock.Mock(returncode=1, duration_seconds=1.5)

            with (
                mock.patch("aim_fine_tune.training.run_captured_command", side_effect=fake_run) as run_mock,
                mock.patch("aim_fine_tune.cli.select_recipe_defaults", return_value=recipe_defaults),
                mock.patch("aim_fine_tune.training.detected_accelerator_ids", return_value=("0x74a1",)),
            ):
                result = self.runner.invoke(app, ["train", "--execute"], env=env)

            self.assertNotEqual(result.exit_code, 0, result.output)
            self.assertIsNotNone(run_mock.call_args.kwargs["output_pipeline"])
            report = json.loads((workspace / "outputs" / "runtime" / "train-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["initialTrainLoss"], 9.0)
            self.assertEqual(report["finalTrainLoss"], 9.0)
            self.assertEqual(report["durationSeconds"], 1.5)
            self.assertFalse(report["ranToCompletion"])
            report_lines = [line for line in result.output.splitlines() if line.startswith('{"event":"train_report"')]
            self.assertEqual([json.loads(line)["report"] for line in report_lines], [report])
            status = json.loads((workspace / "outputs" / "runtime" / "train-status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "FAILED")

    def test_execution_environment_merges_launch_spec_without_rewriting_path(self) -> None:
        base_env = {
            "PATH": "/usr/local/bin:/usr/bin",
            "PRESERVED": "value",
        }

        execution_env = build_execution_environment(
            _launch_spec(env={"ACCELERATE_LOG_LEVEL": "info"}),
            base_env=base_env,
        )

        self.assertEqual(execution_env["PATH"], "/usr/local/bin:/usr/bin")
        self.assertEqual(execution_env["PRESERVED"], "value")
        self.assertEqual(execution_env["ACCELERATE_LOG_LEVEL"], "info")


if __name__ == "__main__":
    main()
