# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class EngineImage(BaseModel):
    model_config = ConfigDict(frozen=True)

    registry: str
    org: str
    name: str
    tag: str | None = None
    digest: str | None = None

    @classmethod
    def parse(cls, reference: str) -> EngineImage:
        name_part, _, digest = reference.strip().partition("@")
        components = name_part.split("/")
        registry = "docker.io"
        if len(components) > 1 and ("." in components[0] or ":" in components[0] or components[0] == "localhost"):
            registry = components.pop(0)
        name, _, tag = components.pop().partition(":")
        return cls(registry=registry, org="/".join(components), name=name, tag=tag or None, digest=digest or None)

    @property
    def reference(self) -> str:
        reference = "/".join(part for part in (self.registry, self.org, self.name) if part)
        if self.tag:
            reference += f":{self.tag}"
        if self.digest:
            reference += f"@{self.digest}"
        return reference
