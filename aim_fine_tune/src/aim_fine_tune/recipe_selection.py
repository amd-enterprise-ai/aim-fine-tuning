# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

import yaml
from aim_runtime.accelerator_detector import AcceleratorDetectionResult, AcceleratorDetector

from aim_fine_tune.models import RecipeData, RecipeDefaults, RecipeSummary

CONTAINER_RECIPES_ROOT = Path("/input/recipes")
PACKAGED_RECIPES_ROOT = Path(__file__).resolve().parents[2] / "recipes"
DEFAULT_RECIPES_ROOTS = (CONTAINER_RECIPES_ROOT, PACKAGED_RECIPES_ROOT)
logger = logging.getLogger(__name__)


def _resolve_recipes_root(recipes_root: Path | None) -> Path:
    if recipes_root is not None:
        return recipes_root

    for candidate in DEFAULT_RECIPES_ROOTS:
        if candidate.exists():
            return candidate

    return DEFAULT_RECIPES_ROOTS[0]


def select_recipe_defaults(
    *,
    backend: str,
    aim_model_id: str,
    num_accelerators: int,
    use_adapter: bool | None = None,
    recipe_id: str | None = None,
    allow_incompatible_hardware: bool = False,
    recipes_root: Path | None = None,
    accelerator_detector: AcceleratorDetectionResult | None = None,
) -> RecipeDefaults | None:
    root = _resolve_recipes_root(recipes_root)
    if not root.exists():
        logger.error("Recipe root does not exist: %s", root)
        return None

    detection = accelerator_detector or AcceleratorDetector().detect()
    if allow_incompatible_hardware:
        if recipe_id is None:
            raise ValueError("Ignoring recipe hardware compatibility requires AIMFT_RECIPE_ID or --recipe-id.")

        recipe_path = _find_recipe_by_id(root, recipe_id)
        if recipe_path is None:
            raise ValueError(f"AIMFT_RECIPE_ID did not match any recipe: '{recipe_id}'.")

        recipe_data = _load_recipe(recipe_path)
        mismatch_reasons = _recipe_mismatch_reasons(
            recipe_data,
            backend=backend,
            aim_model_id=aim_model_id,
            accelerator_ids=None,
            num_accelerators=num_accelerators,
            use_adapter=use_adapter,
            include_hardware_ids=False,
        )
        if mismatch_reasons:
            mismatch_text = "; ".join(mismatch_reasons)
            raise ValueError(
                "AIMFT_RECIPE_ID matched a recipe, but it is not compatible with the requested "
                f"model or accelerator count: {mismatch_text}."
            )

        logger.warning(
            "Selected recipe '%s' for dry-run testing without enforcing detected hardware compatibility.",
            recipe_path,
        )
        return _build_recipe_defaults(recipe_path, recipe_data, num_accelerators)

    if detection.accelerator_count < num_accelerators:
        raise ValueError(
            "AIMFT_NUM_ACCELERATORS exceeds the number of detected accelerators available "
            f"for recipe selection ({num_accelerators} requested, {detection.accelerator_count} detected)."
        )

    accelerator_ids = {RecipeData.normalize_accelerator_id(acc.device_id) for acc in detection.gpu_info}
    if not accelerator_ids:
        logger.error(
            "No accelerators were detected for recipe selection for model '%s' and %d accelerator(s).",
            aim_model_id,
            num_accelerators,
        )
        return None

    matching_recipes: list[tuple[Path, RecipeData]] = []
    for recipe_path in sorted(_iter_recipe_files(root)):
        recipe_data = _load_recipe(recipe_path)
        if recipe_id is not None and not _recipe_matches_id(recipe_data, recipe_id):
            continue
        if not _recipe_matches(
            recipe_data,
            backend=backend,
            aim_model_id=aim_model_id,
            accelerator_ids=accelerator_ids,
            num_accelerators=num_accelerators,
            use_adapter=use_adapter,
        ):
            continue
        matching_recipes.append((recipe_path, recipe_data))

    if matching_recipes:
        recipe_path, recipe_data = min(matching_recipes, key=_recipe_selection_priority)
        logger.info(
            "Selected recipe '%s' for model '%s' with %d accelerator(s) on %s.",
            recipe_path,
            aim_model_id,
            num_accelerators,
            sorted(accelerator_ids),
        )
        return _build_recipe_defaults(recipe_path, recipe_data, num_accelerators)

    if recipe_id is not None:
        for recipe_path in sorted(_iter_recipe_files(root)):
            recipe_data = _load_recipe(recipe_path)
            if not _recipe_matches_id(recipe_data, recipe_id):
                continue
            mismatch_reasons = _recipe_mismatch_reasons(
                recipe_data,
                backend=backend,
                aim_model_id=aim_model_id,
                accelerator_ids=accelerator_ids,
                num_accelerators=num_accelerators,
                use_adapter=use_adapter,
            )
            mismatch_text = "; ".join(mismatch_reasons) if mismatch_reasons else "unknown mismatch"
            raise ValueError(
                "AIMFT_RECIPE_ID matched a recipe, but it is not compatible with the requested "
                f"model or environment: {mismatch_text}."
            )
        raise ValueError(f"AIMFT_RECIPE_ID did not match any recipe: '{recipe_id}'.")

    logger.error(
        "No recipe matched model '%s' for %d accelerator(s) on %s under %s.",
        aim_model_id,
        num_accelerators,
        sorted(accelerator_ids),
        root,
    )
    return None


def list_recipe_summaries(
    *,
    recipes_root: Path | None = None,
    accelerator_detector: AcceleratorDetectionResult | None = None,
    compatible_only: bool = False,
    aim_id: str | None = None,
) -> list[RecipeSummary]:
    root = _resolve_recipes_root(recipes_root)
    if not root.exists():
        logger.error("Recipe root does not exist: %s", root)
        return []

    detection = accelerator_detector or AcceleratorDetector().detect()
    accelerator_ids = {RecipeData.normalize_accelerator_id(acc.device_id) for acc in detection.gpu_info}
    detected_accelerator_count = detection.accelerator_count

    recipes_by_id: dict[str, RecipeSummary] = {}
    for recipe_path in sorted(_iter_recipe_files(root)):
        recipe_data = _load_recipe(recipe_path)
        if aim_id is not None and aim_id.strip() != recipe_data.metadata.aim_id:
            continue
        recipe_id = recipe_data.metadata.recipe_id
        existing = recipes_by_id.get(recipe_id)
        if existing is not None:
            raise ValueError(f"Duplicate recipe ID '{recipe_id}' found in '{existing.path}' and '{recipe_path}'.")
        recipe = _build_recipe_summary(
            recipe_path,
            recipe_data,
            accelerator_ids=accelerator_ids,
            detected_accelerator_count=detected_accelerator_count,
        )
        if compatible_only and not recipe.is_compatible:
            continue
        recipes_by_id[recipe.recipe_id] = recipe

    return list(recipes_by_id.values())


def _iter_recipe_files(recipes_root: Path) -> Iterable[Path]:
    return recipes_root.rglob("*.yaml")


def _load_recipe(recipe_path: Path) -> RecipeData:
    with recipe_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Recipe '{recipe_path}' must contain a mapping at the top level.")
    return RecipeData.from_loaded_mapping(recipe_path, loaded)


def _recipe_matches(
    recipe_data: RecipeData,
    *,
    backend: str,
    aim_model_id: str,
    accelerator_ids: set[str],
    num_accelerators: int,
    use_adapter: bool | None,
) -> bool:
    if recipe_data.metadata.backend != backend:
        return False

    if aim_model_id not in _recipe_model_ids(recipe_data):
        return False

    if recipe_data.metadata.accelerator_count != num_accelerators:
        return False

    if use_adapter is not None and recipe_data.metadata.is_adapter != use_adapter:
        return False

    compatible_accelerators = _recipe_compatible_accelerators(recipe_data)
    return accelerator_ids.issubset(compatible_accelerators)


def _recipe_mismatch_reasons(
    recipe_data: RecipeData,
    *,
    backend: str,
    aim_model_id: str,
    accelerator_ids: set[str] | None,
    num_accelerators: int,
    use_adapter: bool | None,
    include_hardware_ids: bool = True,
) -> list[str]:
    reasons: list[str] = []

    recipe_backend = recipe_data.metadata.backend
    if recipe_backend != backend:
        reasons.append(f"backend mismatch: requested '{backend}', recipe uses '{recipe_backend}'")

    recipe_model_ids = sorted(_recipe_model_ids(recipe_data))
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

    compatible_accelerators = sorted(_recipe_compatible_accelerators(recipe_data))
    if (
        include_hardware_ids
        and accelerator_ids is not None
        and not accelerator_ids.issubset(set(compatible_accelerators))
    ):
        reasons.append(
            "accelerator_ids mismatch: "
            f"detected {sorted(accelerator_ids)}, recipe supports {compatible_accelerators}"
        )

    return reasons


def _find_recipe_by_id(recipes_root: Path, recipe_id: str) -> Path | None:
    matches: list[Path] = []
    for recipe_path in sorted(_iter_recipe_files(recipes_root)):
        recipe_data = _load_recipe(recipe_path)
        if _recipe_matches_id(recipe_data, recipe_id):
            matches.append(recipe_path)

    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            f"AIMFT_RECIPE_ID '{recipe_id.strip()}' matched multiple recipes: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _recipe_matches_id(recipe_data: RecipeData, recipe_id: str) -> bool:
    requested_id = recipe_id.strip()
    if not requested_id:
        return False
    return requested_id == recipe_data.metadata.recipe_id


def _recipe_model_ids(recipe_data: RecipeData) -> set[str]:
    return recipe_data.model_ids()


def _recipe_compatible_accelerators(recipe_data: RecipeData) -> set[str]:
    normalized: set[str] = set()
    for value in recipe_data.metadata.compatible_accelerators:
        normalized.add(RecipeData.normalize_accelerator_id(value))
    return normalized


def _recipe_engine_args(recipe_data: RecipeData) -> dict[str, Any]:
    return dict(recipe_data.engine_args)


def _recipe_selection_priority(candidate: tuple[Path, RecipeData]) -> tuple[int, Path]:
    recipe_path, recipe_data = candidate
    status_priority = 0 if recipe_data.metadata.status == "validated" else 1
    return status_priority, recipe_path


def _build_recipe_summary(
    recipe_path: Path,
    recipe_data: RecipeData,
    *,
    accelerator_ids: set[str],
    detected_accelerator_count: int,
) -> RecipeSummary:
    metadata = recipe_data.metadata.extras

    model_id = recipe_data.primary_model_id() or recipe_path.stem
    aim_id = recipe_data.summary_aim_id(fallback_model_id=model_id)

    compatible_accelerators = tuple(sorted(_recipe_compatible_accelerators(recipe_data)))
    accelerator_count = recipe_data.metadata.accelerator_count
    is_adapter = recipe_data.metadata.is_adapter
    validation = recipe_data.validation_status()
    engine = recipe_data.metadata.backend
    is_compatible = _recipe_is_compatible_with_environment(
        compatible_accelerators=set(compatible_accelerators),
        accelerator_count=accelerator_count,
        accelerator_ids=accelerator_ids,
        detected_accelerator_count=detected_accelerator_count,
    )

    extra_metadata = {key: value for key, value in metadata.items() if key != "engine"}
    extra_metadata.setdefault("backend", recipe_data.metadata.backend)
    if validation is not None:
        extra_metadata.setdefault("validation_status", validation)

    return RecipeSummary(
        path=recipe_path,
        recipe_id=recipe_data.metadata.recipe_id,
        status=recipe_data.metadata.status,
        task=recipe_data.metadata.method.upper(),
        aim_id=aim_id,
        model_id=model_id,
        defaults=recipe_data.defaults.model_dump(exclude_none=True),
        compatible_accelerators=compatible_accelerators,
        accelerator_count=accelerator_count,
        is_adapter=is_adapter,
        engine=engine,
        is_compatible=is_compatible,
        validation=validation,
        metadata=extra_metadata,
    )


def _recipe_is_compatible_with_environment(
    *,
    compatible_accelerators: set[str],
    accelerator_count: int,
    accelerator_ids: set[str],
    detected_accelerator_count: int,
) -> bool:
    if accelerator_count < 1 or detected_accelerator_count < accelerator_count:
        return False
    if not compatible_accelerators or not accelerator_ids:
        return False
    return accelerator_ids.issubset(compatible_accelerators)


def _build_recipe_defaults(recipe_path: Path, recipe_data: RecipeData, num_accelerators: int) -> RecipeDefaults:
    model_id = recipe_data.primary_model_id()
    if model_id is None:
        raise ValueError(f"Recipe '{recipe_path}' must define at least one model identifier.")

    backend_options = _recipe_engine_args(recipe_data)

    return RecipeDefaults.from_recipe_runtime_defaults(
        path=recipe_path,
        model_id=model_id,
        num_accelerators=recipe_data.metadata.accelerator_count,
        use_adapter=recipe_data.metadata.is_adapter,
        runtime_defaults=recipe_data.defaults,
        backend_options=backend_options,
    )
