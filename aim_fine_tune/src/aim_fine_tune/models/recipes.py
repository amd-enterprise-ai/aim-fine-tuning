# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, model_validator

NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


class RecipeDefaults(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    path: Path
    recipe_id: NonEmptyStr
    recipe_digest: NonEmptyStr
    model_id: NonEmptyStr
    num_accelerators: int = Field(gt=0)
    use_adapter: bool | None = None
    epochs: int | None = None
    seed: int | None = None
    evaluation_interval: int | None = None
    logging_interval: int | None = None
    learning_rate: float | None = None
    learning_rate_warmup_steps: int | None = None
    per_accelerator_batch_size: int | None = None
    gradient_accumulation: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    lora_rank: int | None = None
    max_context_length: int | None = None
    save_interval: int | None = None
    backend_options: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_recipe_runtime_defaults(
        cls,
        *,
        path: Path,
        recipe_id: str,
        recipe_digest: str,
        model_id: str,
        num_accelerators: int,
        use_adapter: bool | None,
        runtime_defaults: RecipeRuntimeDefaults,
        backend_options: dict[str, Any],
    ) -> RecipeDefaults:
        return cls(
            path=path,
            recipe_id=recipe_id,
            recipe_digest=recipe_digest,
            model_id=model_id,
            num_accelerators=num_accelerators,
            use_adapter=use_adapter,
            backend_options=dict(backend_options),
            **runtime_defaults.model_dump(),
        )


class RecipeSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    recipe_id: NonEmptyStr
    task: NonEmptyStr
    aim_id: NonEmptyStr
    model_id: NonEmptyStr
    defaults: dict[str, Any] = Field(default_factory=dict)
    compatible_accelerators: tuple[str, ...]
    accelerator_count: int = Field(gt=0)
    is_adapter: bool
    engine: NonEmptyStr
    is_compatible: bool
    validation: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        metadata = {
            "compatibleAccelerators": list(self.compatible_accelerators),
            "acceleratorCount": self.accelerator_count,
            "isAdapter": self.is_adapter,
            "engine": self.engine,
            "isCompatible": self.is_compatible,
        }
        return {
            "recipeId": self.recipe_id,
            "task": self.task,
            "aimId": self.aim_id,
            "modelId": self.model_id,
            "defaults": _json_field_names(self.defaults),
            "metadata": metadata,
        }

    def to_pretty_string(self) -> str:
        compatibility = "compatible" if self.is_compatible else "not compatible"
        accelerator_ids = ", ".join(self.compatible_accelerators) or "none"
        training_type = "adapter" if self.is_adapter else "full-parameter"
        lines = [
            f"{self.recipe_id} [{compatibility}]",
            f"  model: {self.model_id}",
            f"  aim_id: {self.aim_id}",
            f"  task: {self.task}",
            f"  engine: {self.engine}",
            f"  training_type: {training_type}",
            f"  accelerator_count: {self.accelerator_count}",
            f"  compatible_accelerators: {accelerator_ids}",
        ]
        if self.defaults:
            lines.append("  defaults:")
            lines.extend(f"    {key}: {value}" for key, value in self.defaults.items())
        if self.validation is not None:
            lines.append(f"  validation_status: {self.validation}")
        return "\n".join(lines)


class RecipeMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    recipe_id: NonEmptyStr = Field(alias="recipeId")
    backend: NonEmptyStr
    method: NonEmptyStr
    is_adapter: bool = Field(alias="isAdapter")
    accelerator_count: int = Field(gt=0, alias="acceleratorCount")
    model_id: NonEmptyStr | None = Field(default=None, alias="modelId")
    aim_model_id: NonEmptyStr | None = Field(default=None, alias="aimModelId")
    aim_id: NonEmptyStr | None = Field(default=None, alias="aimId")
    extras: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_mapping(cls, metadata: dict[str, Any], *, recipe_path: Path) -> RecipeMetadata:
        known_metadata_keys = {
            "recipeId",
            "backend",
            "method",
            "isAdapter",
            "acceleratorCount",
            "modelId",
            "aimModelId",
            "aimId",
        }

        try:
            return cls.model_validate(
                {
                    **metadata,
                    "extras": {key: value for key, value in metadata.items() if key not in known_metadata_keys},
                }
            )
        except ValidationError as exc:
            raise ValueError(f"Recipe '{recipe_path}' has invalid metadata: {exc}") from exc


class RecipeRuntimeDefaults(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    epochs: int | None = None
    seed: int | None = None
    evaluation_interval: int | None = None
    logging_interval: int | None = None
    learning_rate: float | None = None
    learning_rate_warmup_steps: int | None = None
    per_accelerator_batch_size: int | None = None
    gradient_accumulation: int | None = None
    lora_alpha: int | None = None
    lora_dropout: float | None = None
    lora_rank: int | None = None
    max_context_length: int | None = None
    save_interval: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _sanitize_numeric_values(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        int_fields = {
            "epochs",
            "seed",
            "evaluation_interval",
            "logging_interval",
            "learning_rate_warmup_steps",
            "per_accelerator_batch_size",
            "gradient_accumulation",
            "lora_alpha",
            "lora_rank",
            "max_context_length",
            "save_interval",
        }
        float_fields = {"learning_rate", "lora_dropout"}

        sanitized = dict(value)
        for key in int_fields:
            raw = sanitized.get(key)
            if raw is not None and (isinstance(raw, bool) or not isinstance(raw, int)):
                sanitized[key] = None
        for key in float_fields:
            raw = sanitized.get(key)
            if raw is not None and (isinstance(raw, bool) or not isinstance(raw, (int, float))):
                sanitized[key] = None
        return sanitized

    @classmethod
    def from_mapping(cls, defaults: dict[str, Any]) -> RecipeRuntimeDefaults:
        try:
            return cls.model_validate(defaults)
        except ValidationError as exc:
            raise ValueError(f"Invalid recipe defaults: {exc}") from exc


class _RecipeLoadedPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    metadata: dict[str, Any]
    defaults: dict[str, Any]
    engine_args: dict[str, Any] = Field(alias="engine_args")
    model: NonEmptyStr | None = None


class RecipeData(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: Path
    metadata: RecipeMetadata
    defaults: RecipeRuntimeDefaults
    engine_args: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_loaded_mapping(cls, recipe_path: Path, loaded: dict[str, Any]) -> RecipeData:
        try:
            payload = _RecipeLoadedPayload.model_validate(loaded)
        except ValidationError as exc:
            raise ValueError(
                f"Recipe '{recipe_path}' must define 'metadata', 'defaults', and 'engine_args' as mappings: {exc}"
            ) from exc

        recipe_metadata = RecipeMetadata.from_mapping(payload.metadata, recipe_path=recipe_path)
        runtime_defaults = RecipeRuntimeDefaults.from_mapping(payload.defaults)

        if not recipe_metadata.model_id and not payload.model:
            raise ValueError(
                f"Recipe '{recipe_path}' must define either metadata.modelId or top-level model as a non-empty string."
            )

        return cls(
            path=recipe_path,
            metadata=recipe_metadata,
            defaults=runtime_defaults,
            engine_args=dict(payload.engine_args),
            raw=loaded,
        )

    def digest(self) -> str:
        canonical = json.dumps(self.raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def top_level_string(self, key: str) -> str | None:
        value = self.raw[key] if key in self.raw else None
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    def nested_string(self, section: str, key: str) -> str | None:
        container = self.raw[section] if section in self.raw else None
        if not isinstance(container, dict):
            return None
        value = container[key] if key in container else None
        if isinstance(value, str) and value.strip():
            return value.strip()
        return None

    def model_ids(self) -> set[str]:
        model_ids: set[str] = set()
        if self.metadata.model_id:
            model_ids.add(self.metadata.model_id)
        if self.metadata.aim_model_id:
            model_ids.add(self.metadata.aim_model_id)

        top_level_model = self.top_level_string("model")
        if top_level_model:
            model_ids.add(top_level_model)

        for key in ("modelId", "aimId"):
            value = self.nested_string("aimManifest", key)
            if value:
                model_ids.add(value)
        return model_ids

    def primary_model_id(self) -> str | None:
        if self.metadata.model_id:
            return self.metadata.model_id

        top_level_model = self.top_level_string("model")
        if top_level_model:
            return top_level_model

        for key in ("modelId", "aimId"):
            value = self.nested_string("aimManifest", key)
            if value:
                return value
        return None

    def summary_aim_id(self, *, fallback_model_id: str) -> str:
        for value in (self.metadata.aim_id, self.metadata.model_id):
            if value:
                return value

        for key in ("aimId", "modelId"):
            value = self.nested_string("aimManifest", key)
            if value:
                return value
        return fallback_model_id

    def validation_status(self) -> str | None:
        for key in ("validation", "validationStatus"):
            value = self.top_level_string(key)
            if value:
                return value

        for key in ("validation", "validationStatus"):
            value = self.metadata.extras[key] if key in self.metadata.extras else None
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    @staticmethod
    def normalize_accelerator_id(value: Any) -> str:
        text = str(value).strip().lower()
        return text if text.startswith("0x") else f"0x{text}"


class RecipeCompatibilityEntry(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    compatible_accelerators: list[NonEmptyStr] = Field(alias="compatibleAccelerators")
    recipe_digest: NonEmptyStr | None = Field(default=None, alias="recipeDigest")


class RecipeCompatibility(BaseModel):
    model_config = ConfigDict(frozen=True)

    recipes: dict[str, RecipeCompatibilityEntry] = Field(default_factory=dict)


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
