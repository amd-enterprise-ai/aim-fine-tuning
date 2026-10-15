<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Add a training method

Add a training method to an existing engine when the engine already knows how
to launch the engine, but needs to support another training workflow such as
preference optimization or reinforcement learning. This work extends the engine and its image; it
does not require registering a new engine name.

## 1. Define the method contract

Describe the method before changing the implementation:

- The method name and its CLI spelling.
- The required training and validation data format.
- Common parameters and method-specific parameters.
- The recipe `defaults` required by the method, including any method-specific
  fields and values.
- Whether full-parameter and adapter training are supported.
- Expected runtime outputs and checkpoint format.
- The engine and image versions that support the method.

Add a method concept page or section under
`docs/aim-fine-tuning/concepts/training-methods.md`. Add a method-specific
standard interface under `docs/standard-interfaces/` and expose it through the
corresponding Reference page when the image contract is stable.

## 2. Extend the CLI

Update `aim_fine_tune/src/aim_fine_tune/cli.py`:

- Add the method to the `--method` `click.Choice` values.
- Update `DEFAULT_METHOD` only if the CLI default method itself should change.
- Extend `_coerce_method` so the environment variable `AIMFT_METHOD` accepts
  the method.
- Update method-specific request validation and help text where necessary.

Keep options common to every method in the common CLI. Put method- or
engine-specific settings in recipes and `AIMFT_ENGINE_ARGS` instead of adding
one-off flags to the common command.

## 3. Extend the engine adapter

Update the existing adapter in
`aim_fine_tune/src/aim_fine_tune/engines/<engine>.py`:

- Accept the new method in `prepare` and related validation.
- Keep the existing method behavior unchanged.
- Add method-specific configuration generation and launch commands.
- Validate method-specific `engine_args` and reject unsupported combinations.
- Return a complete `LaunchSpec` for the new method.

If the method needs substantially different engine dependencies or entrypoint
behavior, keep that logic in a focused helper or separate module while retaining
the same registered engine name.

## 4. Update the container image

Update the existing engine image definition, for example
`docker/Dockerfile.aimfttk`, when the method requires:

- Additional engine or data-processing dependencies.
- A different entrypoint or command.
- Method-specific environment defaults.
- New runtime directories or mounts.

Set `AIMFT_METHOD` only when publishing a method-specific image. Otherwise,
ensure the image can select the method through the CLI or environment variable.
Update the image documentation and deployment configuration with the supported
method and image tag.

## 5. Add method-specific recipes

Add recipes under the existing engine directory:

```text
aim_fine_tune/recipes/<engine>/
```

Set `metadata.method` to the new method and keep the common fields in
`metadata` and `defaults`. Put engine and method-specific values in
`engine_args`, following the [recipe schema reference](../aim-fine-tuning/reference/recipe-schema.md)
and the engine reference.

Always set `metadata.method` explicitly. Define the `defaults` mapping for that
method; its fields and values may differ from the defaults used by SFT or other
methods.

Use recipe IDs that include the method, for example:

```text
<engine>/<method>/<model>/<adapter-mode>/<accelerator-count>/<variant>
```

## 6. Add tests

Add focused coverage for:

- CLI `--method` and `AIMFT_METHOD` acceptance and rejection.
- engine method validation.
- Method-specific data and configuration serialization.
- The generated `LaunchSpec`, command, environment, and files.
- Recipe selection by engine, model, method, adapter mode, and accelerator
  compatibility.
- Existing SFT behavior remaining unchanged.

Run the affected package tests:

```bash
cd aim_fine_tune
uv run pytest
```
