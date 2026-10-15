# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml
from aim_runtime.accelerator_detector import AcceleratorDetectionResult, AcceleratorDetector
from aim_runtime.object_model import AcceleratorModel
from pydantic import ValidationError

from aim_fine_tune.models.recipes import RecipeCompatibility, RecipeData, RecipeDefaults, RecipeSummary

BUILTIN_RECIPES_ROOT = Path("/opt/aim_fine_tune/builtin_recipes")
RECIPE_COMPATIBILITY_FILE_NAME = "aimft-recipe-compatibility.yaml"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogRecipe:
    path: Path
    data: RecipeData
    compatible_accelerators: frozenset[str]


def load_recipe(recipe_path: Path) -> RecipeData:
    with recipe_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Recipe '{recipe_path}' must contain a mapping at the top level.")
    return RecipeData.from_loaded_mapping(recipe_path, loaded)


def accelerator_product(value: str) -> str | None:
    """Canonical product name (e.g. MI300X) for a product name or GPU device ID, or None if unknown."""
    text = str(value).strip()
    model = AcceleratorModel.from_string_with_default(text) or AcceleratorModel.from_string_with_default(
        RecipeData.normalize_accelerator_id(text)
    )
    return model.value if model is not None else None


def detected_accelerators(detection: AcceleratorDetectionResult) -> set[str]:
    """Products of the detected GPUs; unrecognized devices keep their device ID and match no product."""
    accelerators: set[str] = set()
    for gpu in detection.gpu_info:
        device_id = RecipeData.normalize_accelerator_id(gpu.device_id)
        accelerators.add(accelerator_product(device_id) or device_id)
    return accelerators


def load_recipe_compatibility(recipes_root: Path) -> RecipeCompatibility:
    compatibility_path = recipes_root / RECIPE_COMPATIBILITY_FILE_NAME
    if not compatibility_path.is_file():
        logger.warning("Recipe compatibility file does not exist: %s", compatibility_path)
        return RecipeCompatibility()

    with compatibility_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    try:
        return RecipeCompatibility.model_validate(loaded)
    except ValidationError as exc:
        raise ValueError(f"Invalid recipe compatibility file '{compatibility_path}': {exc}") from exc


def load_catalog(extra_recipes: Path | None = None) -> list[CatalogRecipe]:
    """The built-in recipes plus any extra recipes; an extra recipe replaces a built-in one with the same ID."""
    catalog = {recipe.data.metadata.recipe_id: recipe for recipe in load_recipe_dir(BUILTIN_RECIPES_ROOT)}
    for recipe in load_recipe_dir(extra_recipes) if extra_recipes is not None else ():
        recipe_id = recipe.data.metadata.recipe_id
        if recipe_id in catalog:
            logger.info("Extra recipe '%s' replaces built-in recipe '%s'.", recipe.path, catalog[recipe_id].path)
        catalog[recipe_id] = recipe
    return list(catalog.values())


def load_recipe_dir(root: Path) -> list[CatalogRecipe]:
    if not root.is_dir():
        logger.warning("Recipe directory does not exist: %s", root)
        return []

    compatibility = load_recipe_compatibility(root)
    catalog: list[CatalogRecipe] = []
    paths_by_id: dict[str, Path] = {}
    for recipe_path in sorted(root.rglob("*.yaml")):
        if recipe_path == root / RECIPE_COMPATIBILITY_FILE_NAME:
            continue
        recipe_data = load_recipe(recipe_path)
        recipe_id = recipe_data.metadata.recipe_id
        if recipe_id in paths_by_id:
            raise ValueError(
                f"Duplicate recipe ID '{recipe_id}' found in '{paths_by_id[recipe_id]}' and '{recipe_path}'."
            )
        paths_by_id[recipe_id] = recipe_path

        entry = compatibility.recipes.get(recipe_id)
        if entry is not None and entry.recipe_digest is not None and entry.recipe_digest != recipe_data.digest():
            raise ValueError(
                f"Recipe '{recipe_path}' does not match the recipe digest listed in {RECIPE_COMPATIBILITY_FILE_NAME}."
            )
        catalog.append(
            CatalogRecipe(
                path=recipe_path,
                data=recipe_data,
                compatible_accelerators=frozenset(
                    _compatibility_product(value, recipe_path)
                    for value in (entry.compatible_accelerators if entry is not None else ())
                ),
            )
        )
    return catalog


def _compatibility_product(value: str, recipe_path: Path) -> str:
    product = accelerator_product(value)
    if product is None:
        raise ValueError(
            f"Unknown accelerator '{value}' listed for '{recipe_path}' in {RECIPE_COMPATIBILITY_FILE_NAME}."
        )
    return product


def select_recipe_defaults(
    *,
    backend: str,
    method: str,
    aim_model_id: str,
    num_accelerators: int,
    use_adapter: bool | None = None,
    recipe_id: str | None = None,
    recipe_path: Path | None = None,
    extra_recipes: Path | None = None,
    accelerator_detector: AcceleratorDetectionResult | None = None,
) -> RecipeDefaults | None:
    if recipe_path is not None:
        return _select_recipe_from_path(
            recipe_path,
            backend=backend,
            method=method,
            aim_model_id=aim_model_id,
            num_accelerators=num_accelerators,
            use_adapter=use_adapter,
            recipe_id=recipe_id,
        )

    catalog = load_catalog(extra_recipes)
    detection = accelerator_detector or AcceleratorDetector().detect()
    if detection.accelerator_count < num_accelerators:
        raise ValueError(
            "AIMFT_NUM_ACCELERATORS exceeds the number of detected accelerators available "
            f"for recipe selection ({num_accelerators} requested, {detection.accelerator_count} detected)."
        )

    accelerators = detected_accelerators(detection)
    if not accelerators:
        logger.error(
            "No accelerators were detected for recipe selection for model '%s' and %d accelerator(s).",
            aim_model_id,
            num_accelerators,
        )
        return None

    candidates = [recipe for recipe in catalog if recipe_id is None or recipe.data.metadata.recipe_id == recipe_id]
    matching = [
        recipe
        for recipe in candidates
        if not _request_mismatch_reasons(
            recipe.data,
            backend=backend,
            method=method,
            aim_model_id=aim_model_id,
            num_accelerators=num_accelerators,
            use_adapter=use_adapter,
        )
        and accelerators.issubset(recipe.compatible_accelerators)
    ]
    if matching:
        selected = min(matching, key=lambda recipe: recipe.path)
        logger.info(
            "Selected recipe '%s' for model '%s' with %d accelerator(s) on %s.",
            selected.path,
            aim_model_id,
            num_accelerators,
            sorted(accelerators),
        )
        return _build_recipe_defaults(selected.path, selected.data)

    if recipe_id is not None:
        if not candidates:
            raise ValueError(f"AIMFT_RECIPE_ID did not match any recipe in the recipe catalog: '{recipe_id}'.")
        candidate = candidates[0]
        reasons = _request_mismatch_reasons(
            candidate.data,
            backend=backend,
            method=method,
            aim_model_id=aim_model_id,
            num_accelerators=num_accelerators,
            use_adapter=use_adapter,
        )
        if not accelerators.issubset(candidate.compatible_accelerators):
            reasons.append(
                f"accelerator mismatch: detected {sorted(accelerators)}, "
                f"{RECIPE_COMPATIBILITY_FILE_NAME} lists {sorted(candidate.compatible_accelerators)}"
            )
        raise ValueError(
            "AIMFT_RECIPE_ID matched a recipe, but it is not compatible with the requested "
            f"model or environment: {'; '.join(reasons)}."
        )

    logger.error(
        "No recipe matched model '%s' for %d accelerator(s) on %s in the recipe catalog.",
        aim_model_id,
        num_accelerators,
        sorted(accelerators),
    )
    return None


def list_recipe_summaries(
    *,
    extra_recipes: Path | None = None,
    accelerator_detector: AcceleratorDetectionResult | None = None,
    compatible_only: bool = False,
    aim_id: str | None = None,
) -> list[RecipeSummary]:
    catalog = load_catalog(extra_recipes)
    detection = accelerator_detector or AcceleratorDetector().detect()
    accelerators = detected_accelerators(detection)

    summaries: list[RecipeSummary] = []
    for recipe in catalog:
        if aim_id is not None and aim_id.strip() != recipe.data.metadata.aim_id:
            continue
        summary = _build_recipe_summary(
            recipe,
            accelerators=accelerators,
            detected_accelerator_count=detection.accelerator_count,
        )
        if compatible_only and not summary.is_compatible:
            continue
        summaries.append(summary)
    return summaries


def _select_recipe_from_path(
    recipe_path: Path,
    *,
    backend: str,
    method: str,
    aim_model_id: str,
    num_accelerators: int,
    use_adapter: bool | None,
    recipe_id: str | None,
) -> RecipeDefaults:
    recipe_data = load_recipe(recipe_path)
    reasons = _request_mismatch_reasons(
        recipe_data,
        backend=backend,
        method=method,
        aim_model_id=aim_model_id,
        num_accelerators=num_accelerators,
        use_adapter=use_adapter,
    )
    if recipe_id is not None and recipe_id != recipe_data.metadata.recipe_id:
        reasons.append(f"recipe_id mismatch: requested '{recipe_id}', recipe is '{recipe_data.metadata.recipe_id}'")
    if reasons:
        raise ValueError(f"AIMFT_RECIPE_PATH recipe does not match the request: {'; '.join(reasons)}.")

    logger.warning(
        "Using recipe '%s' from AIMFT_RECIPE_PATH without recipe catalog or accelerator compatibility checks.",
        recipe_path,
    )
    return _build_recipe_defaults(recipe_path, recipe_data)


def _request_mismatch_reasons(
    recipe_data: RecipeData,
    *,
    backend: str,
    method: str,
    aim_model_id: str,
    num_accelerators: int,
    use_adapter: bool | None,
) -> list[str]:
    reasons: list[str] = []

    recipe_backend = recipe_data.metadata.backend
    if recipe_backend != backend:
        reasons.append(f"backend mismatch: requested '{backend}', recipe uses '{recipe_backend}'")

    recipe_method = recipe_data.metadata.method.upper()
    if recipe_method != method.upper():
        reasons.append(f"method mismatch: requested '{method}', recipe uses '{recipe_method}'")

    recipe_model_ids = sorted(recipe_data.model_ids())
    if aim_model_id not in recipe_model_ids:
        reasons.append(
            f"model_id mismatch: requested '{aim_model_id}', recipe supports {recipe_model_ids or ['<none>']}"
        )

    recipe_accelerator_count = recipe_data.metadata.accelerator_count
    if recipe_accelerator_count != num_accelerators:
        reasons.append(
            "accelerator_count mismatch: " f"requested {num_accelerators}, recipe requires {recipe_accelerator_count!r}"
        )

    if use_adapter is not None:
        recipe_is_adapter = recipe_data.metadata.is_adapter
        if recipe_is_adapter != use_adapter:
            requested_mode = "adapter" if use_adapter else "full-parameter"
            recipe_mode = "adapter" if recipe_is_adapter else "full-parameter"
            reasons.append(f"adapter_type mismatch: requested '{requested_mode}', recipe uses '{recipe_mode}'")

    return reasons


def _build_recipe_summary(
    recipe: CatalogRecipe,
    *,
    accelerators: set[str],
    detected_accelerator_count: int,
) -> RecipeSummary:
    recipe_data = recipe.data
    model_id = recipe_data.primary_model_id() or recipe.path.stem
    aim_id = recipe_data.summary_aim_id(fallback_model_id=model_id)
    accelerator_count = recipe_data.metadata.accelerator_count
    validation = recipe_data.validation_status()
    is_compatible = (
        detected_accelerator_count >= accelerator_count
        and bool(accelerators)
        and accelerators.issubset(recipe.compatible_accelerators)
    )

    extra_metadata = {key: value for key, value in recipe_data.metadata.extras.items() if key != "engine"}
    extra_metadata.setdefault("backend", recipe_data.metadata.backend)
    if validation is not None:
        extra_metadata.setdefault("validation_status", validation)

    return RecipeSummary(
        path=recipe.path,
        recipe_id=recipe_data.metadata.recipe_id,
        task=recipe_data.metadata.method.upper(),
        aim_id=aim_id,
        model_id=model_id,
        defaults=recipe_data.defaults.model_dump(exclude_none=True),
        compatible_accelerators=tuple(sorted(recipe.compatible_accelerators)),
        accelerator_count=accelerator_count,
        is_adapter=recipe_data.metadata.is_adapter,
        engine=recipe_data.metadata.backend,
        is_compatible=is_compatible,
        validation=validation,
        metadata=extra_metadata,
    )


def _build_recipe_defaults(recipe_path: Path, recipe_data: RecipeData) -> RecipeDefaults:
    model_id = recipe_data.primary_model_id()
    if model_id is None:
        raise ValueError(f"Recipe '{recipe_path}' must define at least one model identifier.")

    return RecipeDefaults.from_recipe_runtime_defaults(
        path=recipe_path,
        recipe_id=recipe_data.metadata.recipe_id,
        recipe_digest=recipe_data.digest(),
        model_id=model_id,
        num_accelerators=recipe_data.metadata.accelerator_count,
        use_adapter=recipe_data.metadata.is_adapter,
        runtime_defaults=recipe_data.defaults,
        backend_options=dict(recipe_data.engine_args),
    )
