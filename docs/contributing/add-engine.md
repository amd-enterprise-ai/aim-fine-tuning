<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Add an engine

An engine connects the common `aim_fine_tune` workflow to a fine-tuning engine.
It owns engine-specific configuration, launch files, runtime environment
variables, and the container image used to execute the job.

## 1. Implement the adapter

Create `aim_fine_tune/src/aim_fine_tune/backends/<backend>.py`. Implement the
`BackendAdapter` protocol from `backends/base.py`:

- Expose a stable `name` matching the backend name used in recipes and CLI
  requests.
- Implement `prepare(request)`.
- Implement `create_output_observers()` and return the backend's event
  actors. Each actor declares its subscriptions and may return derived events
  for the controller to publish. The controller runs matching actors
  synchronously; actors do not own threads or queues. For example, the AIMFTTK
  metrics actor subscribes to `StdoutLine` and publishes `TrainMetricsMessage`.
- Keep observers independent of output destinations. Execution publishes exact
  `RawStdoutChunk` and `RawStderrChunk` events; the runtime derives distinct
  line events and republishes backend-produced messages. New event classes can
  be published and subscribed to without changing the controller. Event sources
  own their scheduling; future sources such as TensorBoard followers may run
  independently and publish into the same controller. `AIMFT_JSON_LOGS`
  selects JSON message output instead of raw streams and does not change
  observation. `runtime/train.stdout.log` and `runtime/train.stderr.log` are
  written by destination actors that subscribe to their respective chunks.
- Implement `parse_run_report(...)` for final figures from the backend's
  persisted training state.
- Return a `LaunchSpec` containing the command, environment, generated files,
  and configuration mount destination.
- Validate engine-specific `engine_args` and reject unknown or invalid values.

Keep common request parsing and recipe selection in the existing CLI and recipe
modules. The adapter should only translate the normalized request into the
engine's launch workspace.

## 2. Register the engine

Update `aim_fine_tune/src/aim_fine_tune/backends/__init__.py`:

1. Import the adapter.
2. Instantiate it.
3. Add it to `_ADAPTERS` under the exact backend name.

The registry supplies the values accepted by `--backend` and by backend
validation. A backend that is not registered cannot be selected even if its
adapter module exists.

Update `cli.py` only where the new engine changes common CLI behavior:

- Change `DEFAULT_BACKEND` only if the new engine should be the default.
- Extend the `--method` choices and method validation if the engine supports a
  method other than the currently supported SFT method.
- Add a common CLI option only when the option applies to every engine. Keep
  engine-specific options in recipe `engine_args` or `AIMFT_ENGINE_ARGS`.

## 3. Build the engine image

Add an engine image definition under `docker/`, following the existing
`docker/Dockerfile.aimfttk` pattern:

- Start from the engine or accelerator base image.
- Set `AIMFT_BACKEND` to the registered engine name.
- Set `AIMFT_METHOD` when the image supports a specific method.
- Copy the packaged recipes and their `aimft-recipe-compatibility.yaml` into
  `/opt/aim_fine_tune/builtin_recipes` in the image.
- Install the `aim_fine_tune` package and all engine dependencies.
- Provide an entrypoint that invokes `aim_fine_tune train --execute`.

Document the build context, base image, required runtime mounts, and published
image tag alongside the Dockerfile. Update deployment manifests or Helm values
that select the image when the new engine is used by a deployed workflow.

## 4. Add tests

Add focused tests for:

- engine registration and name discovery.
- Request-to-`LaunchSpec` translation.
- Valid and invalid `engine_args`.
- Generated configuration files and launch commands.
- CLI engine and method selection, when those choices change.

Run the affected package tests from the repository root:

```bash
cd aim_fine_tune
uv run pytest
```
