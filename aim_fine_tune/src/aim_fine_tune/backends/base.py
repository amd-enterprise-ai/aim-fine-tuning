# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, Protocol, TypeVar

if TYPE_CHECKING:
    from aim_fine_tune.models.reports import TrainingReport
    from aim_fine_tune.models.training import TrainRequest


@dataclass(frozen=True)
class LaunchSpec:
    backend: str
    command: tuple[str, ...]
    env: dict[str, str]
    files: dict[str, str]
    config_mount_destination: PurePosixPath = PurePosixPath("/configs")

    def to_jsonable(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "command": list(self.command),
            "env": self.env,
            "files": sorted(self.files),
            "config_mount_destination": str(self.config_mount_destination),
        }


@dataclass(frozen=True)
class RawStdoutChunk:
    data: bytes


@dataclass(frozen=True)
class RawStderrChunk:
    data: bytes


@dataclass(frozen=True)
class StdoutClosed:
    pass


@dataclass(frozen=True)
class StderrClosed:
    pass


@dataclass(frozen=True)
class StdoutLine:
    text: str


@dataclass(frozen=True)
class StderrLine:
    text: str


@dataclass(frozen=True)
class LogMessage:
    message: str
    level: str = "info"
    event: Literal["log"] = "log"

    def to_jsonable(self) -> dict[str, object]:
        return {
            "event": self.event,
            "level": self.level,
            "message": self.message,
        }


EventT = TypeVar("EventT")
PublishedEvents = object | tuple[object, ...] | None


@dataclass(frozen=True)
class EventSubscription(Generic[EventT]):
    event_type: type[EventT]
    callback: Callable[[EventT], PublishedEvents]


class EventActor(Protocol):
    def subscriptions(self) -> tuple[EventSubscription[Any], ...]:
        pass


class EventSource(Protocol):
    """An independently scheduled producer of pipeline events."""

    def run(self, publish: Callable[[object], None]) -> None:
        pass


class BackendAdapter(Protocol):
    @property
    def name(self) -> str:
        pass

    @property
    def supported_methods(self) -> tuple[str, ...]:
        pass

    def prepare(self, request: TrainRequest) -> LaunchSpec:
        pass

    def create_output_observers(self, request: TrainRequest) -> tuple[EventActor, ...]:
        pass

    def parse_run_report(
        self,
        *,
        request: TrainRequest,
        duration_seconds: float,
        accelerators: tuple[str, ...] = (),
    ) -> TrainingReport:
        pass


def parse_engine_args(value: Any, *, source: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source} must be a valid JSON object.") from exc
        if not isinstance(decoded, dict):
            raise ValueError(f"{source} must decode to a JSON object.")
        return decoded
    raise ValueError(f"{source} must be a JSON object.")


def merge_engine_args(*engine_args: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in engine_args:
        merged = _merge_engine_arg_mappings(merged, item)
    return merged


def _merge_engine_arg_mappings(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_engine_arg_mappings(existing, value)
            continue
        merged[key] = value
    return merged
