# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import click
from pydantic import TypeAdapter, ValidationError

from aim_fine_tune.backends import (
    get_backend_method_names,
    get_backend_names,
)
from aim_fine_tune.backends.base import (
    merge_engine_args,
    parse_engine_args,
)
from aim_fine_tune.errors import AimFineTuneError
from aim_fine_tune.extras.download import download_command
from aim_fine_tune.models.methods.base import BaseParameters
from aim_fine_tune.models.methods.common import CommonParameters
from aim_fine_tune.models.methods.sft import SftParameters
from aim_fine_tune.models.toolkit import ToolkitParameters
from aim_fine_tune.models.training import (
    TrainCliValues,
    TrainRequest,
    TrainRequestData,
    TrainRequestMetadata,
    TrainRuntimeParameters,
)
from aim_fine_tune.recipe_selection import (
    RecipeDefaults,
    list_recipe_summaries,
    select_recipe_defaults,
)
from aim_fine_tune.training import run_training

# REGISTER SUPPORTED TRAINING PARAMETERS
PARAMETER_MODELS = (ToolkitParameters, CommonParameters, SftParameters)

_BOOL_ADAPTER = TypeAdapter(bool)


def _method_parameter_options(function):
    fields = {
        name: field
        for parameters_type in PARAMETER_MODELS
        for name, field in parameters_type.model_fields.items()
        if BaseParameters._resolution_sources(field) is not None
    }
    for name, field in reversed(fields.items()):
        sources = BaseParameters._resolution_sources(field)
        cli_options = sources["cli_options"] if sources is not None else {}
        function = click.option(f"--{name.replace('_', '-')}", default=None, **cli_options)(function)
    return function


@click.group()
@click.version_option()
def app():
    """AIM fine-tuning CLI."""


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
    "--extra-recipes",
    type=click.Path(path_type=Path, exists=False, file_okay=False, dir_okay=True),
    default=None,
    help="Directory of extra recipes, with its own recipe compatibility file.",
)
def list_recipes(
    output_format: str,
    compatible_only: bool,
    aim_id: str | None,
    extra_recipes: Path | None,
) -> None:
    """List available fine-tuning recipes."""
    try:
        resolved_extra_recipes = ToolkitParameters.resolve_extra_recipes(extra_recipes, dict(os.environ))
        recipes = list_recipe_summaries(
            compatible_only=compatible_only,
            extra_recipes=resolved_extra_recipes,
            aim_id=aim_id.strip() or None if aim_id is not None else None,
        )
    except (AimFineTuneError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    normalized_format = output_format.lower()
    if normalized_format == "json":
        click.echo(json.dumps({"recipes": [recipe.to_jsonable() for recipe in recipes]}, indent=2))
        return

    if not recipes:
        click.echo("No compatible recipes found." if compatible_only else "No recipes found.")
    else:
        click.echo(
            "\n\n".join([f"Found {len(recipes)} recipe(s)."] + [recipe.to_pretty_string() for recipe in recipes])
        )


@app.command()
@click.option("--dry-run/--execute", default=True)
@click.option(
    "--json-logs/--tee-engine-logs",
    default=None,
    help="Emit JSON Lines (training metrics and runtime logs) instead of relaying all engine output.",
)
@click.option(
    "--render-dir",
    type=click.Path(path_type=Path, exists=False, file_okay=False, dir_okay=True),
    default=None,
)
@_method_parameter_options
def train(
    dry_run: bool,
    json_logs: bool | None,
    render_dir: Path | None,
    **parameter_values: Any,
):
    """Translate AIMFT inputs into a backend launch workspace."""
    try:
        resolved_json_logs = _resolve_json_logs(json_logs, os.environ)
    except ValidationError as exc:
        raise click.ClickException(f"Invalid AIMFT_JSON_LOGS: {exc}") from exc

    cli_values = TrainCliValues(**parameter_values)
    env = dict(os.environ)
    try:
        run_training(
            lambda: build_train_request(cli_values, env, dry_run=dry_run),
            dry_run=dry_run,
            json_logs=resolved_json_logs,
            render_dir=render_dir,
        )
    except (AimFineTuneError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


app.add_command(download_command, name="debug-download")

# Helper functions


def build_train_request(cli_values: TrainCliValues, env: dict[str, str], *, dry_run: bool = True) -> TrainRequest:
    toolkit_values = _resolve_toolkit_parameters(cli_values, env)
    backend, method = _validate_backend_method(toolkit_values.backend, toolkit_values.method)
    num_accelerators = _resolve_num_accelerators(toolkit_values.num_accelerators, env)
    recipe_defaults = _select_recipe_defaults(
        toolkit_values,
        backend=backend,
        method=method,
        num_accelerators=num_accelerators,
        dry_run=dry_run,
    )
    runtime_values = _resolve_runtime_parameters(
        cli_values,
        env,
        recipe_defaults,
        method=method,
        use_adapter=toolkit_values.use_adapter,
    )
    backend_options = merge_engine_args(
        {} if recipe_defaults is None else dict(recipe_defaults.backend_options),
        parse_engine_args(toolkit_values.engine_args, source="AIMFT_ENGINE_ARGS"),
    )

    return TrainRequest(
        metadata=TrainRequestMetadata(
            recipe_id=None if recipe_defaults is None else recipe_defaults.recipe_id,
            recipe_digest=None if recipe_defaults is None else recipe_defaults.recipe_digest,
            backend=backend,
            model_id=cast(str, toolkit_values.aim_model_id),
            aim_id=cast(str, toolkit_values.aim_model_id),
            method=method,
            is_adapter=toolkit_values.use_adapter,
            accelerator_count=num_accelerators,
        ),
        data=TrainRequestData(
            model_path=cast(Path, toolkit_values.model_path),
            train_data_path=cast(Path, toolkit_values.train_data_path),
            validation_data_path=toolkit_values.validation_data_path,
            output_path=cast(Path, toolkit_values.output_path),
        ),
        defaults=runtime_values,
        engine_args=backend_options,
    )


def _resolve_toolkit_parameters(cli_values: TrainCliValues, env: dict[str, str]) -> ToolkitParameters:
    try:
        return ToolkitParameters.resolve(cli_values, env)
    except ValidationError as exc:
        raise AimFineTuneError(f"Invalid toolkit parameters: {exc}") from exc


def _validate_backend_method(backend: str, method: str) -> tuple[str, str]:
    supported_backends = get_backend_names()
    if backend not in supported_backends:
        formatted_supported = ", ".join(supported_backends)
        raise AimFineTuneError(f"Unsupported backend '{backend}'. Supported backends: {formatted_supported}.")

    supported_methods = get_backend_method_names(backend)
    if method not in supported_methods:
        formatted_supported = ", ".join(supported_methods)
        raise AimFineTuneError(
            f"Backend '{backend}' does not support method '{method}'. Supported methods: {formatted_supported}."
        )
    return backend, method


def _select_recipe_defaults(
    toolkit_values: ToolkitParameters,
    *,
    backend: str,
    method: str,
    num_accelerators: int,
    dry_run: bool,
) -> RecipeDefaults | None:
    recipe_defaults = select_recipe_defaults(
        backend=backend,
        method=method,
        aim_model_id=cast(str, toolkit_values.aim_model_id),
        num_accelerators=num_accelerators,
        use_adapter=toolkit_values.use_adapter,
        recipe_id=toolkit_values.recipe_id,
        recipe_path=toolkit_values.recipe_path,
        extra_recipes=toolkit_values.extra_recipes,
    )
    if recipe_defaults is None and not dry_run:
        raise AimFineTuneError(
            "No compatible recipe found for execution. Use 'list-recipes --compatible-only' to inspect available recipes."
        )
    return recipe_defaults


def _resolve_runtime_parameters(
    cli_values: TrainCliValues,
    env: dict[str, str],
    recipe_defaults: RecipeDefaults | None,
    *,
    method: str,
    use_adapter: bool,
) -> SftParameters | TrainRuntimeParameters:
    try:
        runtime_values = TrainRuntimeParameters.resolve_for_method(method, cli_values, env, recipe_defaults)
    except (ValidationError, ValueError) as exc:
        raise AimFineTuneError(f"Invalid runtime training parameters: {exc}") from exc

    return runtime_values.apply_adapter_defaults(use_adapter)


def _resolve_json_logs(cli_value: bool | None, env: Mapping[str, str]) -> bool:
    value = cli_value if cli_value is not None else env.get("AIMFT_JSON_LOGS", False)
    return _BOOL_ADAPTER.validate_python(value)


def _resolve_num_accelerators(cli_value: int | None, env: dict[str, str]) -> int:
    if cli_value is not None:
        return cli_value

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
