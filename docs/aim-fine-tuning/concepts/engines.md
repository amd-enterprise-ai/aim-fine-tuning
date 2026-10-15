<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Engines

An AIMFT engine adapter converts an AIM Fine-tuning training request into the files,
environment variables, and command needed to run a specific fine-tuning engine.

The `aim_fine_tune` CLI owns the common workflow. Each engine owns the
engine-specific configuration and launch details.

## Engine contract

Every engine adapter accepts a normalized training request and produces a
launch specification containing:

- The engine name.
- The command to invoke in the execution environment.
- Environment variables required by the engine.
- Generated configuration files.
- The destination where configuration files are mounted, `/configs` by default.

This boundary lets the CLI support additional engines without exposing each
engine's configuration format as part of the common command-line interface.

## Available engines

Available engines and engine-specific parameters are documented in the [engine
reference](../reference/engines.md).
