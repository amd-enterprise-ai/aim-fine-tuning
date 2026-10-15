# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from aim_fine_tune.backends.aimfttk.adapter import AimfttkBackendAdapter
from aim_fine_tune.backends.base import BackendAdapter

_aimfttk_adapter = AimfttkBackendAdapter()

_ADAPTERS: dict[str, BackendAdapter] = {
    "aimfttk": _aimfttk_adapter,
}


def get_backend_names() -> tuple[str, ...]:
    return tuple(sorted(_ADAPTERS))


def get_backend_method_names(name: str) -> tuple[str, ...]:
    return get_backend_adapter(name).supported_methods


def get_training_method_names() -> tuple[str, ...]:
    return tuple(sorted({method for adapter in _ADAPTERS.values() for method in adapter.supported_methods}))


def get_backend_adapter(name: str) -> BackendAdapter:
    try:
        return _ADAPTERS[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_ADAPTERS))
        raise ValueError(f"Unsupported backend '{name}'. Supported backends: {supported}.") from exc
