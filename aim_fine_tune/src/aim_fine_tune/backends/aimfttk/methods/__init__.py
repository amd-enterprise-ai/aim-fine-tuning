# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from aim_fine_tune.backends.aimfttk.methods.base import AimfttkTrainingMethod
from aim_fine_tune.backends.aimfttk.methods.sft import AimfttkSftMethod

_METHODS: dict[str, AimfttkTrainingMethod] = {method.name: method for method in (AimfttkSftMethod(),)}


def get_method_names() -> tuple[str, ...]:
    return tuple(sorted(_METHODS))


def get_method(name: str) -> AimfttkTrainingMethod:
    normalized = name.upper()
    try:
        return _METHODS[normalized]
    except KeyError as exc:
        supported = ", ".join(get_method_names())
        raise ValueError(f"AIMFTTK does not support method '{name}'. Supported methods: {supported}.") from exc
