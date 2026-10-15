# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import field_validator, model_validator

from .methods.base import BaseParameters, parameter_field


class ToolkitParameters(BaseParameters):
    backend: str = parameter_field("aimfttk", env="AIMFT_BACKEND")
    method: str = parameter_field("SFT", env="AIMFT_METHOD")
    model_path: Path | None = parameter_field(None, env="AIMFT_MODEL_PATH")
    train_data_path: Path | None = parameter_field(None, env="AIMFT_TRAIN_DATA_PATH")
    validation_data_path: Path | None = parameter_field(None, env="AIMFT_VALIDATION_DATA_PATH")
    output_path: Path | None = parameter_field(None, env="AIMFT_OUTPUT_PATH")
    aim_model_id: str | None = parameter_field(None, env="AIM_MODEL_ID")
    recipe_id: str | None = parameter_field(None, env="AIMFT_RECIPE_ID")
    recipe_path: Path | None = parameter_field(None, env="AIMFT_RECIPE_PATH")
    extra_recipes: Path | None = parameter_field(None, env="AIMFT_EXTRA_RECIPES")
    use_adapter: bool = parameter_field(False, env="AIMFT_USE_ADAPTER")
    num_accelerators: int | None = parameter_field(None, env="AIMFT_NUM_ACCELERATORS", ge=1)
    engine_args: Any = parameter_field(None, env="AIMFT_ENGINE_ARGS")

    @classmethod
    def resolve_extra_recipes(cls, cli_value: Path | None, env: dict[str, str]) -> Path | None:
        value = cli_value
        if value is None and "AIMFT_EXTRA_RECIPES" in env:
            value = Path(env["AIMFT_EXTRA_RECIPES"]).expanduser()
        if value is not None and not value.is_dir():
            raise ValueError("AIMFT_EXTRA_RECIPES must point to an existing directory.")
        return value

    @field_validator("backend")
    @classmethod
    def normalize_backend(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("method")
    @classmethod
    def normalize_method(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("aim_model_id", "recipe_id")
    @classmethod
    def normalize_identifier(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @field_validator(
        "model_path",
        "train_data_path",
        "validation_data_path",
        "output_path",
        "recipe_path",
        "extra_recipes",
    )
    @classmethod
    def expand_path(cls, value: Path | None) -> Path | None:
        return None if value is None else value.expanduser()

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if self.model_path is None:
            raise ValueError("AIMFT_MODEL_PATH is required.")
        if not self.model_path.is_dir():
            raise ValueError("AIMFT_MODEL_PATH must point to an existing directory.")
        if self.train_data_path is None:
            raise ValueError("AIMFT_TRAIN_DATA_PATH is required.")
        if not self.train_data_path.is_file():
            raise ValueError("AIMFT_TRAIN_DATA_PATH must point to an existing file.")
        if self.validation_data_path is not None and not self.validation_data_path.is_file():
            raise ValueError("AIMFT_VALIDATION_DATA_PATH must point to an existing file when provided.")
        if self.output_path is None:
            raise ValueError("AIMFT_OUTPUT_PATH is required.")
        if self.output_path.exists() and not self.output_path.is_dir():
            raise ValueError("AIMFT_OUTPUT_PATH must point to a directory.")
        if self.aim_model_id is None:
            raise ValueError("AIM_MODEL_ID is required.")
        if self.recipe_path is not None and not self.recipe_path.is_file():
            raise ValueError("AIMFT_RECIPE_PATH must point to an existing file.")
        if self.extra_recipes is not None and not self.extra_recipes.is_dir():
            raise ValueError("AIMFT_EXTRA_RECIPES must point to an existing directory.")
        return self
