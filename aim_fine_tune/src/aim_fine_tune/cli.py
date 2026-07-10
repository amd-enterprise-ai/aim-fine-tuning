# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import click
from pydantic import ValidationError

from aim_fine_tune.backends import get_backend_adapter, get_backend_names
from aim_fine_tune.backends.base import LaunchSpec, merge_engine_args, parse_engine_args
from aim_fine_tune.download import debug_download_command
from aim_fine_tune.errors import AimFineTuneError
from aim_fine_tune.models import (
    TrainCliValues,
    TrainRequest,
    TrainRequestData,
    TrainRequestMetadata,
    TrainRuntimeParameters,
)
from aim_fine_tune.recipe_selection import (
    RecipeDefaults,
    RecipeSummary,
    list_recipe_summaries,
    select_recipe_defaults,
)

DEFAULT_BACKEND = "aimfttk"
DEFAULT_METHOD = "SFT"
RUNTIME_DIR_NAME = "runtime"


@click.group()
@click.version_option()
def app():
    """AIM fine-tuning CLI."""


app.add_command(debug_download_command)


@app.command("list-recipes")
@click.option(
    "--output-format",
    type=click.Choice(["pretty", "json"], case_sensitive=False),
    default="json",
    show_default=True,
)
@click.option("--compatible-only", is_flag=True, default=False)
@click.option("--aim-id", default=None, help="Filter recipes by AIM ID.")
@click.option(
    "--recipes-root",
    type=click.Path(path_type=Path, exists=False, file_okay=False, dir_okay=True),
    default=None,
)
def list_recipes(
    output_format: str,
    compatible_only: bool,
    aim_id: str | None,
    recipes_root: Path | None,
) -> None:
    """List available fine-tuning recipes."""
    try:
        resolved_recipes_root = _coerce_optional_existing_directory(
            _merged_value(recipes_root, dict(os.environ), "AIMFT_RECIPES_ROOT"),
            "AIMFT_RECIPES_ROOT",
        )
        recipes = list_recipe_summaries(
            compatible_only=compatible_only,
            recipes_root=resolved_recipes_root,
            aim_id=_coerce_optional_string(aim_id),
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    normalized_format = output_format.lower()
    if normalized_format == "json":
        click.echo(json.dumps({"recipes": [recipe.to_jsonable() for recipe in recipes]}, indent=2))
        return

    click.echo(_render_recipe_listing(recipes, compatible_only=compatible_only))


@app.command()
@click.option("--backend", type=click.Choice(get_backend_names()), default=None)
@click.option("--dry-run/--execute", default=True)
@click.option("--method", type=click.Choice(["SFT"], case_sensitive=False), default=None)
@click.option(
    "--model-path",
    type=click.Path(path_type=Path, exists=False, file_okay=False, dir_okay=True),
    default=None,
)
@click.option(
    "--train-data-path",
    type=click.Path(path_type=Path, exists=False, file_okay=True, dir_okay=False),
    default=None,
)
@click.option(
    "--validation-data-path",
    type=click.Path(path_type=Path, exists=False, file_okay=True, dir_okay=False),
    default=None,
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path, exists=False, file_okay=False, dir_okay=True),
    default=None,
)
@click.option("--aim-model-id", default=None)
@click.option("--recipe-id", default=None)
@click.option("--allow-incompatible-recipe", is_flag=True, default=False)
@click.option(
    "--recipes-root",
    type=click.Path(path_type=Path, exists=False, file_okay=False, dir_okay=True),
    default=None,
)
@click.option(
    "--render-dir",
    type=click.Path(path_type=Path, exists=False, file_okay=False, dir_okay=True),
    default=None,
)
def train(
    backend: str | None,
    dry_run: bool,
    method: str | None,
    model_path: Path | None,
    train_data_path: Path | None,
    validation_data_path: Path | None,
    output_path: Path | None,
    aim_model_id: str | None,
    recipe_id: str | None,
    allow_incompatible_recipe: bool,
    recipes_root: Path | None,
    render_dir: Path | None,
):
    """Translate AIMFT inputs into a backend launch workspace."""
    cli_values = TrainCliValues(
        backend=backend,
        method=method,
        model_path=model_path,
        train_data_path=train_data_path,
        validation_data_path=validation_data_path,
        output_path=output_path,
        aim_model_id=aim_model_id,
        recipe_id=recipe_id,
        allow_incompatible_recipe=allow_incompatible_recipe,
        recipes_root=recipes_root,
    )

    try:
        request = build_train_request(cli_values, dict(os.environ), dry_run=dry_run)
        adapter = get_backend_adapter(request.metadata.backend)
        launch_spec = adapter.prepare(request)
        resolved_render_dir = resolve_render_dir(request.data.output_path, request.metadata.backend, render_dir)
        write_render_workspace(resolved_render_dir, request, launch_spec)
        click.echo(f"Rendered {request.metadata.backend} launch workspace to {resolved_render_dir}")

        if not dry_run:
            ensure_output_path(request.data.output_path)
            execute_launch_spec(launch_spec, resolved_render_dir)
            click.echo("Execution completed successfully.")
    except (AimFineTuneError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


def build_train_request(cli_values: TrainCliValues, env: dict[str, str], *, dry_run: bool = True) -> TrainRequest:
    backend = _coerce_backend(_merged_value(cli_values.backend, env, "AIMFT_BACKEND", DEFAULT_BACKEND))
    method = _coerce_method(_merged_value(cli_values.method, env, "AIMFT_METHOD", DEFAULT_METHOD))
    model_path = _coerce_required_directory(
        _merged_value(cli_values.model_path, env, "AIMFT_MODEL_PATH"),
        "AIMFT_MODEL_PATH",
    )
    train_data_path = _coerce_required_file(
        _merged_value(cli_values.train_data_path, env, "AIMFT_TRAIN_DATA_PATH"),
        "AIMFT_TRAIN_DATA_PATH",
    )
    validation_data_path = _coerce_optional_file(
        _merged_value(cli_values.validation_data_path, env, "AIMFT_VALIDATION_DATA_PATH")
    )
    output_path = _coerce_directory_path(
        _merged_value(cli_values.output_path, env, "AIMFT_OUTPUT_PATH"),
        "AIMFT_OUTPUT_PATH",
    )
    aim_model_id = _coerce_required_string(
        _merged_value(cli_values.aim_model_id, env, "AIM_MODEL_ID"),
        "AIM_MODEL_ID",
    )
    recipe_id = _coerce_optional_string(_merged_value(cli_values.recipe_id, env, "AIMFT_RECIPE_ID"))
    recipes_root = _coerce_optional_existing_directory(
        _merged_value(cli_values.recipes_root, env, "AIMFT_RECIPES_ROOT"),
        "AIMFT_RECIPES_ROOT",
    )
    allow_incompatible_recipe = _coerce_optional_bool(
        _merged_value(
            cli_values.allow_incompatible_recipe,
            env,
            "AIMFT_ALLOW_INCOMPATIBLE_RECIPE",
        ),
        "AIMFT_ALLOW_INCOMPATIBLE_RECIPE",
    )
    if allow_incompatible_recipe and not dry_run:
        raise AimFineTuneError("AIMFT_ALLOW_INCOMPATIBLE_RECIPE can only be used with --dry-run.")
    num_accelerators = _resolve_num_accelerators(
        cli_values.num_accelerators,
        env,
    )
    use_adapter = _coerce_bool(
        _merged_value(
            cli_values.use_adapter,
            env,
            "AIMFT_USE_ADAPTER",
            False,
        ),
        "AIMFT_USE_ADAPTER",
    )
    recipe_defaults = select_recipe_defaults(
        backend=backend,
        aim_model_id=aim_model_id,
        num_accelerators=num_accelerators,
        use_adapter=use_adapter,
        recipe_id=recipe_id,
        allow_incompatible_hardware=bool(allow_incompatible_recipe),
        recipes_root=recipes_root,
    )
    if recipe_defaults is None and not dry_run:
        raise AimFineTuneError(
            "No compatible recipe found for execution. Use 'list-recipes --compatible-only' to inspect available recipes."
        )
    runtime_raw = {
        "epochs": _merged_value(
            cli_values.epochs,
            env,
            "AIMFT_EPOCHS",
            _recipe_default(recipe_defaults, "epochs", 1),
        ),
        "seed": _merged_value(
            cli_values.seed,
            env,
            "AIMFT_SEED",
            _recipe_default(recipe_defaults, "seed", 42),
        ),
        "evaluation_interval": _merged_value(
            cli_values.evaluation_interval,
            env,
            "AIMFT_EVALUATION_INTERVAL",
            _recipe_default(recipe_defaults, "evaluation_interval", 1000),
        ),
        "logging_interval": _merged_value(
            cli_values.logging_interval,
            env,
            "AIMFT_LOGGING_INTERVAL",
            _recipe_default(recipe_defaults, "logging_interval", 25),
        ),
        "learning_rate": _merged_value(
            cli_values.learning_rate,
            env,
            "AIMFT_LEARNING_RATE",
            _recipe_default(recipe_defaults, "learning_rate"),
        ),
        "learning_rate_warmup_steps": _merged_value(
            cli_values.learning_rate_warmup_steps,
            env,
            "AIMFT_LEARNING_RATE_WARMUP_STEPS",
            _recipe_default(recipe_defaults, "learning_rate_warmup_steps"),
        ),
        "per_accelerator_batch_size": _merged_value(
            cli_values.per_accelerator_batch_size,
            env,
            "AIMFT_PER_ACCELERATOR_BATCH_SIZE",
            _recipe_default(recipe_defaults, "per_accelerator_batch_size"),
        ),
        "gradient_accumulation": _merged_value(
            cli_values.gradient_accumulation,
            env,
            "AIMFT_GRADIENT_ACCUMULATION",
            _recipe_default(recipe_defaults, "gradient_accumulation"),
        ),
        "lora_alpha": _merged_value(
            cli_values.lora_alpha,
            env,
            "AIMFT_LORA_ALPHA",
            _recipe_default(recipe_defaults, "lora_alpha"),
        ),
        "lora_dropout": _merged_value(
            cli_values.lora_dropout,
            env,
            "AIMFT_LORA_DROPOUT",
            _recipe_default(recipe_defaults, "lora_dropout"),
        ),
        "lora_rank": _merged_value(
            cli_values.lora_rank,
            env,
            "AIMFT_LORA_RANK",
            _recipe_default(recipe_defaults, "lora_rank"),
        ),
        "max_context_length": _merged_value(
            cli_values.max_context_length,
            env,
            "AIMFT_MAX_CONTEXT_LENGTH",
            _recipe_default(recipe_defaults, "max_context_length"),
        ),
        "save_interval": (
            _merged_value(
                cli_values.save_interval,
                env,
                "AIMFT_SAVE_INTERVAL",
                _recipe_default(recipe_defaults, "save_interval", 500),
            )
            or 500
        ),
    }

    try:
        runtime_values = TrainRuntimeParameters.model_validate(runtime_raw)
    except ValidationError as exc:
        raise AimFineTuneError(f"Invalid runtime training parameters: {exc}") from exc

    epochs = runtime_values.epochs
    seed = runtime_values.seed
    evaluation_interval = runtime_values.evaluation_interval
    logging_interval = runtime_values.logging_interval
    learning_rate = runtime_values.learning_rate
    learning_rate_warmup_steps = runtime_values.learning_rate_warmup_steps
    per_accelerator_batch_size = runtime_values.per_accelerator_batch_size
    gradient_accumulation = runtime_values.gradient_accumulation
    lora_alpha = runtime_values.lora_alpha
    lora_dropout = runtime_values.lora_dropout
    lora_rank = runtime_values.lora_rank
    max_context_length = runtime_values.max_context_length
    save_interval = runtime_values.save_interval

    if use_adapter and lora_dropout is None:
        lora_dropout = 0.0
    if use_adapter and lora_rank is None:
        lora_rank = 64

    backend_options = merge_engine_args(
        _recipe_backend_options(recipe_defaults),
        parse_engine_args(
            _merged_value(cli_values.engine_args, env, "AIMFT_ENGINE_ARGS"),
            source="AIMFT_ENGINE_ARGS",
        ),
    )

    return TrainRequest(
        metadata=TrainRequestMetadata(
            recipe_id=recipe_id,
            backend=backend,
            model_id=aim_model_id,
            aim_id=aim_model_id,
            method=method,
            is_adapter=use_adapter,
            accelerator_count=num_accelerators,
        ),
        data=TrainRequestData(
            model_path=model_path,
            train_data_path=train_data_path,
            validation_data_path=validation_data_path,
            output_path=output_path,
        ),
        defaults=TrainRuntimeParameters(
            epochs=epochs,
            seed=seed,
            evaluation_interval=evaluation_interval,
            logging_interval=logging_interval,
            learning_rate=learning_rate,
            learning_rate_warmup_steps=learning_rate_warmup_steps,
            per_accelerator_batch_size=per_accelerator_batch_size,
            gradient_accumulation=gradient_accumulation,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_rank=lora_rank,
            max_context_length=max_context_length,
            save_interval=save_interval,
        ),
        engine_args=backend_options,
    )


def resolve_render_dir(output_path: Path, backend: str, render_dir: Path | None) -> Path:
    if render_dir is not None:
        return render_dir.expanduser()
    return output_path


def write_render_workspace(render_dir: Path, request: TrainRequest, launch_spec: LaunchSpec) -> None:
    render_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, content in launch_spec.files.items():
        target_path = render_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")

    runtime_dir = render_dir / RUNTIME_DIR_NAME
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "train-request.json").write_text(
        json.dumps(request.to_jsonable(), indent=2) + "\n",
        encoding="utf-8",
    )
    (runtime_dir / "launch-spec.json").write_text(
        json.dumps(launch_spec.to_jsonable(), indent=2) + "\n",
        encoding="utf-8",
    )

    (runtime_dir / "run.sh").write_text(
        render_shell_script(launch_spec),
        encoding="utf-8",
    )


def build_execution_command(launch_spec: LaunchSpec) -> list[str]:
    return list(launch_spec.command)


def execute_launch_spec(launch_spec: LaunchSpec, render_dir: Path) -> None:
    stage_runtime_configs(launch_spec, render_dir)
    validate_execution_paths(launch_spec)
    execution_env = os.environ.copy()
    execution_env.update(launch_spec.env)
    command = build_execution_command(launch_spec)
    completed_process = subprocess.run(command, check=False, env=execution_env)
    if completed_process.returncode != 0:
        raise AimFineTuneError(f"Backend execution failed with exit code {completed_process.returncode}.")


def validate_execution_paths(launch_spec: LaunchSpec) -> None:
    runtime_configs_dir = Path(str(launch_spec.config_mount_destination))
    for relative_path in launch_spec.files:
        if not relative_path.startswith("configs/"):
            continue
        staged_path = runtime_configs_dir / Path(relative_path).name
        if not staged_path.exists():
            raise AimFineTuneError(f"Staged config file is missing: '{staged_path}'.")

    command = build_execution_command(launch_spec)
    if len(command) >= 2:
        command_target = Path(command[1])
        if command_target.is_absolute() and not command_target.exists():
            raise AimFineTuneError(f"Backend command path does not exist: '{command_target}'.")


def stage_runtime_configs(launch_spec: LaunchSpec, render_dir: Path) -> None:
    runtime_configs_dir = Path(str(launch_spec.config_mount_destination))
    runtime_configs_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in launch_spec.files:
        if not relative_path.startswith("configs/"):
            continue
        target_path = runtime_configs_dir / Path(relative_path).name
        source_path = render_dir / relative_path
        target_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")


def ensure_output_path(output_path: Path) -> None:
    if output_path.exists() and not output_path.is_dir():
        raise AimFineTuneError(f"AIMFT_OUTPUT_PATH must point to a directory, got '{output_path}'.")
    output_path.mkdir(parents=True, exist_ok=True)


def render_shell_script(launch_spec: LaunchSpec) -> str:
    export_lines = "\n".join(f"export {key}={_quote_shell(value)}" for key, value in sorted(launch_spec.env.items()))
    command = " ".join(_quote_shell(argument) for argument in launch_spec.command)
    return f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- \"$(dirname -- \"${{BASH_SOURCE[0]}}\")\" && pwd)
mkdir -p {launch_spec.config_mount_destination}
cp \"$SCRIPT_DIR/configs/\"* {launch_spec.config_mount_destination}/
{export_lines}
exec {command}
"""


def _render_recipe_listing(recipes: list[RecipeSummary], *, compatible_only: bool) -> str:
    if not recipes:
        return "No compatible recipes found." if compatible_only else "No recipes found."

    lines = [f"Found {len(recipes)} recipe(s)."]
    for recipe in recipes:
        compatibility = "compatible" if recipe.is_compatible else "not compatible"
        accelerator_ids = ", ".join(recipe.compatible_accelerators) or "none"
        training_type = "adapter" if recipe.is_adapter else "full-parameter"
        lines.extend(
            [
                "",
                f"{recipe.recipe_id} [{compatibility}]",
                f"  model: {recipe.model_id}",
                f"  aim_id: {recipe.aim_id}",
                f"  task: {recipe.task}",
                f"  engine: {recipe.engine}",
                f"  training_type: {training_type}",
                f"  accelerator_count: {recipe.accelerator_count}",
                f"  compatible_accelerators: {accelerator_ids}",
            ]
        )
        if recipe.defaults:
            lines.append("  defaults:")
            for key, value in recipe.defaults.items():
                lines.append(f"    {key}: {value}")
        if recipe.validation is not None:
            lines.append(f"  validation_status: {recipe.validation}")
    return "\n".join(lines)


def _quote_shell(argument: str) -> str:
    return "'" + argument.replace("'", "'\"'\"'") + "'"


def _merged_value(cli_value: Any, env: dict[str, str], env_key: str, default: Any = None) -> Any:
    if cli_value is not None:
        return cli_value
    if env_key in env:
        return _decode_env_value(env_key, env[env_key])
    return default


def _merged_values(cli_value: Any, env: dict[str, str], *env_keys: str, default: Any = None) -> Any:
    if cli_value is not None:
        return cli_value
    for env_key in env_keys:
        if env_key in env:
            return _decode_env_value(env_key, env[env_key])
    return default


def _decode_env_value(env_key: str, raw_value: str) -> Any:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        if env_key in {
            "AIMFT_MODEL_PATH",
            "AIMFT_TRAIN_DATA_PATH",
            "AIMFT_VALIDATION_DATA_PATH",
            "AIMFT_OUTPUT_PATH",
            "AIMFT_HF_MODEL_ID",
            "AIMFT_MODEL_OUTPUT_PATH",
            "AIMFT_MODEL_CACHE_DIR",
            "AIMFT_RECIPES_ROOT",
            "AIMFT_HF_TOKEN",
            "HF_TOKEN",
            "AIMFT_DATASET_S3_URI",
            "AIMFT_DATASET_OUTPUT_PATH",
            "AIMFT_S3_ENDPOINT_URL",
            "AWS_ENDPOINT_URL",
            "BUCKET_STORAGE_HOST",
            "AIMFT_S3_ACCESS_KEY_ID",
            "AWS_ACCESS_KEY_ID",
            "BUCKET_STORAGE_ACCESS_KEY",
            "AIMFT_S3_SECRET_ACCESS_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "BUCKET_STORAGE_SECRET_KEY",
            "AIMFT_S3_SESSION_TOKEN",
            "AWS_SESSION_TOKEN",
            "AIMFT_S3_REGION",
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
            "AIMFT_BACKEND",
            "AIMFT_METHOD",
            "AIM_MODEL_ID",
            "AIMFT_RECIPE_ID",
            "AIMFT_ENGINE_ARGS",
        }:
            return raw_value
        raise AimFineTuneError(f"Environment variable {env_key} must be valid JSON or a raw string value.")


def _coerce_backend(value: Any) -> str:
    text = _coerce_required_string(value, "AIMFT_BACKEND").lower()
    supported = get_backend_names()
    if text not in supported:
        formatted_supported = ", ".join(supported)
        raise AimFineTuneError(f"Unsupported backend '{text}'. Supported backends: {formatted_supported}.")
    return text


def _coerce_method(value: Any) -> str:
    text = _coerce_required_string(value, "AIMFT_METHOD").upper()
    if text != DEFAULT_METHOD:
        raise AimFineTuneError("Only AIMFT_METHOD=SFT is supported in the current CLI.")
    return text


def _coerce_required_string(value: Any, name: str) -> str:
    if value is None:
        raise AimFineTuneError(f"{name} is required.")
    if not isinstance(value, str):
        raise AimFineTuneError(f"{name} must be a string.")
    text = value.strip()
    if not text:
        raise AimFineTuneError(f"{name} must not be empty.")
    return text


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise AimFineTuneError("Expected a string value.")
    text = value.strip()
    return text or None


def _coerce_bool(value: Any, name: str) -> bool:
    optional_value = _coerce_optional_bool(value, name)
    if optional_value is None:
        raise AimFineTuneError(f"{name} is required.")
    return optional_value


def _coerce_optional_bool(value: Any, name: str) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise AimFineTuneError(f"{name} must be a boolean.")


def _coerce_optional_int(value: Any, name: str, *, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise AimFineTuneError(f"{name} must be an integer.")
    if minimum is not None and value < minimum:
        raise AimFineTuneError(f"{name} must be >= {minimum}.")
    return value


def _coerce_directory_path(value: Any, name: str) -> Path:
    if value is None:
        raise AimFineTuneError(f"{name} is required.")
    path = _coerce_path(value, name)
    if path.exists() and not path.is_dir():
        raise AimFineTuneError(f"{name} must point to a directory.")
    return path


def _coerce_optional_directory_path(value: Any, name: str) -> Path | None:
    if value is None:
        return None
    path = _coerce_path(value, name)
    if path.exists() and not path.is_dir():
        raise AimFineTuneError(f"{name} must point to a directory.")
    return path


def _coerce_optional_existing_directory(value: Any, name: str) -> Path | None:
    path = _coerce_optional_directory_path(value, name)
    if path is None:
        return None
    if not path.exists() or not path.is_dir():
        raise AimFineTuneError(f"{name} must point to an existing directory.")
    return path


def _coerce_required_directory(value: Any, name: str) -> Path:
    path = _coerce_directory_path(value, name)
    if not path.exists() or not path.is_dir():
        raise AimFineTuneError(f"{name} must point to an existing directory.")
    return path


def _coerce_required_file(value: Any, name: str) -> Path:
    path = _coerce_path(value, name)
    if not path.exists() or not path.is_file():
        raise AimFineTuneError(f"{name} must point to an existing file.")
    return path


def _coerce_optional_file(value: Any) -> Path | None:
    if value is None:
        return None
    path = _coerce_path(value, "AIMFT_VALIDATION_DATA_PATH")
    if not path.exists() or not path.is_file():
        raise AimFineTuneError("AIMFT_VALIDATION_DATA_PATH must point to an existing file when provided.")
    return path


def _coerce_path(value: Any, name: str) -> Path:
    if isinstance(value, Path):
        return value.expanduser()
    if not isinstance(value, str):
        raise AimFineTuneError(f"{name} must be a string path.")
    text = value.strip()
    if not text:
        raise AimFineTuneError(f"{name} must not be empty.")
    return Path(text).expanduser()


def _recipe_default(recipe_defaults: RecipeDefaults | None, attribute: str, fallback: Any = None) -> Any:
    if recipe_defaults is None:
        return fallback
    value = getattr(recipe_defaults, attribute)
    if value is None:
        return fallback
    return value


def _recipe_backend_option(recipe_defaults: RecipeDefaults | None, key: str) -> Any:
    if recipe_defaults is None:
        return None
    return recipe_defaults.backend_options.get(key)


def _recipe_backend_options(recipe_defaults: RecipeDefaults | None) -> dict[str, Any]:
    if recipe_defaults is None:
        return {}
    return dict(recipe_defaults.backend_options)


def _resolve_num_accelerators(cli_value: Any, env: dict[str, str]) -> int:
    explicit = _coerce_optional_int(cli_value, "AIMFT_NUM_ACCELERATORS", minimum=1)
    if explicit is not None:
        return explicit

    env_value = _merged_value(None, env, "AIMFT_NUM_ACCELERATORS")
    parsed = _coerce_optional_int(env_value, "AIMFT_NUM_ACCELERATORS", minimum=1)
    if parsed is not None:
        return parsed

    for env_key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        raw_value = env.get(env_key)
        if raw_value is None:
            continue
        visible_devices = [
            item.strip() for item in raw_value.split(",") if item.strip() and item.strip() not in {"-1", "none", "None"}
        ]
        if visible_devices:
            return len(visible_devices)
    return 1


if __name__ == "__main__":
    app()
