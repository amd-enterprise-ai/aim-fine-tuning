# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import FieldInfo

RESOLUTION_METADATA_KEY = "parameter_resolution"


def parameter_field(
    default: Any = None,
    *,
    env: str,
    cli: str | None = None,
    recipe: str | None = None,
    cli_options: dict[str, Any] | None = None,
    **field_options: Any,
) -> Any:
    return Field(
        default,
        json_schema_extra={
            RESOLUTION_METADATA_KEY: {
                "env": env,
                "cli": cli,
                "recipe": recipe,
                "cli_options": cli_options or {},
            }
        },
        **field_options,
    )


class BaseParameters(BaseModel):
    _method_parameter_types: ClassVar[dict[str, type[BaseParameters]]] = {}
    method_name: ClassVar[str | None] = None
    model_config = ConfigDict(populate_by_name=True)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.method_name is not None:
            BaseParameters._method_parameter_types[cls.method_name] = cls

    @classmethod
    def method_parameter_types(cls) -> tuple[type[BaseParameters], ...]:
        return tuple(cls._method_parameter_types.values())

    @classmethod
    def incompatible_method_error(
        cls,
        requested_method: str,
        cli_values: BaseModel | Mapping[str, Any],
        env: Mapping[str, str],
    ) -> str:
        parameter_names = [
            sources["env"]
            for name, field in cls.model_fields.items()
            if name in cls.__annotations__
            and (sources := cls._resolution_sources(field)) is not None
            and (cls._source_value(cli_values, sources["cli"] or name) is not None or sources["env"] in env)
        ]
        return f"Parameters {', '.join(parameter_names)} do not belong to the selected training method."

    @classmethod
    def resolve(
        cls,
        cli_values: BaseModel | Mapping[str, Any],
        env: Mapping[str, str],
        recipe_defaults: BaseModel | Mapping[str, Any] | None = None,
    ) -> Self:
        values: dict[str, Any] = {}
        for name, field in cls.model_fields.items():
            sources = cls._resolution_sources(field)
            if sources is None:
                continue

            cli_name = sources["cli"] or name
            cli_value = cls._source_value(cli_values, cli_name)
            if cli_value is not None:
                values[name] = cli_value
                continue

            env_name = sources["env"]
            if env_name in env:
                values[name] = cls._decode_env_value(env[env_name])
                continue

            recipe_name = sources["recipe"] or name
            recipe_value = cls._source_value(recipe_defaults, recipe_name)
            if recipe_value is not None:
                values[name] = recipe_value

        return cls.model_validate(values)

    @classmethod
    def has_explicit_values(
        cls,
        cli_values: BaseModel | Mapping[str, Any],
        env: Mapping[str, str],
    ) -> bool:
        for name, field in cls.model_fields.items():
            if name not in cls.__annotations__:
                continue
            sources = cls._resolution_sources(field)
            if sources is None:
                continue
            if cls._source_value(cli_values, sources["cli"] or name) is not None:
                return True
            if sources["env"] in env:
                return True
        return False

    @staticmethod
    def _resolution_sources(field: FieldInfo) -> dict[str, Any] | None:
        extra = field.json_schema_extra
        if not isinstance(extra, dict):
            return None
        sources = extra.get(RESOLUTION_METADATA_KEY)
        return sources if isinstance(sources, dict) else None

    @staticmethod
    def _source_value(source: BaseModel | Mapping[str, Any] | None, name: str) -> Any:
        if source is None:
            return None
        if isinstance(source, Mapping):
            return source.get(name)
        return getattr(source, name, None)

    @staticmethod
    def _decode_env_value(raw_value: str) -> Any:
        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            return raw_value
