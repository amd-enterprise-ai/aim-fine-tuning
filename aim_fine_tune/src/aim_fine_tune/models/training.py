# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aim_fine_tune.models.methods.base import BaseParameters
from aim_fine_tune.models.methods.common import CommonParameters


class TrainRuntimeParameters(CommonParameters):
    """Backward-compatible common runtime parameters container."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @classmethod
    def resolve_for_method(
        cls,
        method: str,
        cli_values: TrainCliValues,
        env: dict[str, str],
        recipe_defaults: BaseModel | dict[str, Any] | None,
    ) -> BaseParameters:
        parameter_types = BaseParameters.method_parameter_types()
        parameters_type = next(
            (parameter_type for parameter_type in parameter_types if parameter_type.method_name == method),
            None,
        )
        if parameters_type is None:
            raise ValueError(f"Unsupported training method '{method}'.")

        for parameter_type in parameter_types:
            if parameter_type is not parameters_type and parameter_type.has_explicit_values(cli_values, env):
                raise ValueError(parameter_type.incompatible_method_error(method, cli_values, env))
        return parameters_type.resolve(cli_values, env, recipe_defaults)


class TrainRequestMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    recipe_id: str | None = Field(default=None, alias="recipeId")
    recipe_digest: str | None = Field(default=None, alias="recipeDigest")
    backend: str
    model_id: str = Field(alias="modelId")
    aim_id: str = Field(alias="aimId")
    method: str
    is_adapter: bool = Field(alias="isAdapter")
    accelerator_count: int = Field(alias="acceleratorCount")

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "recipeId": self.recipe_id,
            "recipeDigest": self.recipe_digest,
            "backend": self.backend,
            "modelId": self.model_id,
            "aimId": self.aim_id,
            "method": self.method,
            "isAdapter": self.is_adapter,
            "acceleratorCount": self.accelerator_count,
        }


class TrainRequestData(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    model_path: Path = Field(alias="modelPath")
    train_data_path: Path = Field(alias="trainDataPath")
    validation_data_path: Path | None = Field(alias="validationDataPath")
    output_path: Path = Field(alias="outputPath")

    def to_jsonable(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)


class TrainRequest(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    metadata: TrainRequestMetadata
    data: TrainRequestData
    defaults: BaseParameters
    engine_args: dict[str, Any] = Field(default_factory=dict, alias="engineArgs")

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_jsonable(),
            "data": self.data.to_jsonable(),
            "defaults": self.defaults.model_dump(mode="json", by_alias=True, exclude_none=True),
            "engineArgs": _json_field_names(self.engine_args),
        }


class TrainCliValues(BaseModel):
    model_config = ConfigDict(extra="allow")


def _json_field_names(value: Any) -> Any:
    if isinstance(value, dict):
        return {_snake_to_camel(str(key)): _json_field_names(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [_json_field_names(item) for item in value]
    if isinstance(value, tuple):
        return [_json_field_names(item) for item in value]
    return value


def _snake_to_camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])
