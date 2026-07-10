# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, TypedDict

from aim_fine_tune.backends.base import LaunchSpec, merge_engine_args
from aim_fine_tune.models import TrainRequest

DEFAULT_CHAT_TEMPLATE = "mistral-with-system"
DEFAULT_MISSING_PAD_TOKEN_STRATEGY = "bos-repurpose"


class AimfttkDistributedType(str, Enum):
    AUTO_DEEPSPEED_STAGE1 = "auto-deepspeed-stage1"
    AUTO_DEEPSPEED_STAGE3 = "auto-deepspeed-stage3"
    AUTO_DDP = "auto-ddp"
    AUTO_SINGLE_PROCESS = "auto-single-process"


SUPPORTED_DISTRIBUTED_TYPES = (
    AimfttkDistributedType.AUTO_DEEPSPEED_STAGE1.value,
    AimfttkDistributedType.AUTO_DEEPSPEED_STAGE3.value,
    AimfttkDistributedType.AUTO_DDP.value,
    AimfttkDistributedType.AUTO_SINGLE_PROCESS.value,
)
DEFAULT_LEARNING_RATE_WARMUP_STEPS = 100
USE_CACHE_INCOMPATIBLE_MODEL_PREFIXES = ("google/gemma-3",)
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
)


class AimfttkBackendOptions(TypedDict):
    distributed_type: AimfttkDistributedType | str
    merge_adapter: bool
    finetuning_config: dict[str, Any]


class AimfttkBackendAdapter:
    name = "aimfttk"

    def prepare(self, request: TrainRequest) -> LaunchSpec:
        metadata = request.metadata
        data = request.data
        if metadata.method != "SFT":
            raise ValueError("The aimfttk adapter currently supports only SFT.")

        backend_options = self._parse_backend_options(
            request.engine_args,
            num_accelerators=metadata.accelerator_count,
        )
        merge_adapter = backend_options["merge_adapter"]
        model_destination = _to_posix_path(data.model_path)
        train_data_destination = _to_posix_path(data.train_data_path)
        validation_data_destination = (
            _to_posix_path(data.validation_data_path) if data.validation_data_path is not None else None
        )
        output_root = _to_posix_path(data.output_path)
        config_mount_destination = output_root / "configs"

        accelerate_config = self._build_accelerate_config(
            distributed_type=backend_options["distributed_type"],
            num_accelerators=metadata.accelerator_count,
        )
        finetuning_config = self._build_finetuning_config(
            request=request,
            backend_options=backend_options,
            model_destination=model_destination,
            train_data_destination=train_data_destination,
            validation_data_destination=validation_data_destination,
            output_root=output_root,
        )
        entrypoint_script = self._build_entrypoint_script(
            merge_adapter=merge_adapter,
            model_destination=model_destination,
            output_root=output_root,
            config_mount_destination=config_mount_destination,
        )
        accelerate_config_yaml = (
            accelerate_config if isinstance(accelerate_config, str) else _to_yaml(accelerate_config)
        )
        if not accelerate_config_yaml.endswith("\n"):
            accelerate_config_yaml += "\n"

        return LaunchSpec(
            backend=self.name,
            command=("/bin/bash", str(config_mount_destination / "entrypoint.sh")),
            env={
                "ACCELERATE_LOG_LEVEL": "info",
                "TRANSFORMERS_VERBOSITY": "info",
            },
            files={
                "configs/accelerate_config.yaml": accelerate_config_yaml,
                "configs/finetuning_config.yaml": _to_yaml(finetuning_config),
                "configs/entrypoint.sh": entrypoint_script,
            },
            config_mount_destination=config_mount_destination,
        )

    def _build_accelerate_config(
        self, *, distributed_type: AimfttkDistributedType | str, num_accelerators: int
    ) -> dict[str, Any] | str:
        if isinstance(distributed_type, str) and not isinstance(distributed_type, AimfttkDistributedType):
            return Path(distributed_type).read_text(encoding="utf-8")

        if distributed_type in (
            AimfttkDistributedType.AUTO_DEEPSPEED_STAGE1,
            AimfttkDistributedType.AUTO_DEEPSPEED_STAGE3,
        ):
            is_stage3 = distributed_type == AimfttkDistributedType.AUTO_DEEPSPEED_STAGE3
            deepspeed_config: dict[str, Any] = {
                "zero_stage": 3 if is_stage3 else 1,
                "bf16": {"enabled": True},
            }
            if is_stage3:
                deepspeed_config.update(
                    {
                        "offload_optimizer_device": "none",
                        "offload_param_device": "none",
                        "zero3_init_flag": True,
                        "zero3_save_16bit_model": True,
                    }
                )
            return {
                "compute_environment": "LOCAL_MACHINE",
                "distributed_type": "DEEPSPEED",
                "deepspeed_config": deepspeed_config,
                "fsdp_config": {},
                "machine_rank": 0,
                "main_process_ip": None,
                "main_process_port": None,
                "mixed_precision": "bf16",
                "num_machines": 1,
                "num_processes": num_accelerators,
                "use_cpu": False,
            }
        if distributed_type == AimfttkDistributedType.AUTO_SINGLE_PROCESS:
            return {
                "compute_environment": "LOCAL_MACHINE",
                "distributed_type": "NO",
                "fsdp_config": {},
                "machine_rank": 0,
                "main_process_ip": None,
                "main_process_port": None,
                "mixed_precision": "bf16",
                "num_machines": 1,
                "num_processes": 1,
                "use_cpu": False,
            }
        if distributed_type == AimfttkDistributedType.AUTO_DDP:
            return {
                "compute_environment": "LOCAL_MACHINE",
                "debug": False,
                "distributed_type": "MULTI_GPU",
                "downcast_bf16": "no",
                "enable_cpu_affinity": False,
                "machine_rank": 0,
                "main_training_function": "main",
                "mixed_precision": "bf16",
                "num_machines": 1,
                "num_processes": num_accelerators,
                "rdzv_backend": "static",
                "same_network": True,
                "tpu_env": [],
                "tpu_use_cluster": False,
                "tpu_use_sudo": False,
                "use_cpu": False,
            }
        raise ValueError(f"Unsupported aimfttk distributed type '{distributed_type}'.")

    def _build_finetuning_config(
        self,
        *,
        request: TrainRequest,
        backend_options: AimfttkBackendOptions,
        model_destination: PurePosixPath,
        train_data_destination: PurePosixPath,
        validation_data_destination: PurePosixPath | None,
        output_root: PurePosixPath,
    ) -> dict[str, Any]:
        metadata = request.metadata
        defaults = request.defaults
        checkpoints_dir = output_root / "checkpoints"
        training_args: dict[str, Any] = {
            "per_device_train_batch_size": defaults.per_accelerator_batch_size or 1,
            "per_device_eval_batch_size": defaults.per_accelerator_batch_size or 1,
            "gradient_accumulation_steps": defaults.gradient_accumulation or 1,
            "learning_rate": defaults.learning_rate if defaults.learning_rate is not None else 0.00005,
            "max_grad_norm": 7.0,
            "weight_decay": 0.000001 if metadata.is_adapter else 0.00001,
            "optim": "adamw_torch",
            "num_train_epochs": defaults.epochs,
            "output_dir": str(checkpoints_dir),
            "lr_scheduler_type": "cosine",
            "logging_strategy": "steps",
            "logging_steps": defaults.logging_interval,
            "save_strategy": "steps" if defaults.save_interval != -1 else "no",
            "save_steps": defaults.save_interval if defaults.save_interval != -1 else None,
            "seed": defaults.seed,
            "bf16": True,
            "report_to": ["none"],
            "push_to_hub": False,
            "gradient_checkpointing": False,
        }
        training_args = {k: v for k, v in training_args.items() if v is not None}

        training_args["warmup_steps"] = (
            defaults.learning_rate_warmup_steps
            if defaults.learning_rate_warmup_steps is not None
            else DEFAULT_LEARNING_RATE_WARMUP_STEPS
        )

        validation_data: dict[str, Any]
        if validation_data_destination is None:
            validation_data = {"type": "NONE"}
            training_args["eval_strategy"] = "no"
        else:
            validation_data = {
                "type": "CONCATENATION",
                "datasets": [{"path": str(validation_data_destination)}],
            }
            training_args.update(
                {
                    "eval_steps": defaults.evaluation_interval,
                    "eval_strategy": "steps",
                    "metric_for_best_model": "loss",
                    "greater_is_better": False,
                }
            )

        peft_conf: dict[str, Any]
        if metadata.is_adapter:
            peft_conf = {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "peft_kwargs": {
                    "r": defaults.lora_rank or 64,
                    "lora_alpha": float(defaults.lora_alpha or 16),
                    "lora_dropout": defaults.lora_dropout if defaults.lora_dropout is not None else 0.0,
                    "target_modules": list(DEFAULT_TARGET_MODULES),
                },
            }
        else:
            peft_conf = {"peft_type": "NO_PEFT"}

        model_args: dict[str, Any] = {
            "torch_dtype": "bfloat16",
            "attn_implementation": "flash_attention_2",
        }
        if _supports_use_cache_model_arg(metadata.aim_id):
            model_args["use_cache"] = False

        base_finetuning_config = {
            "method": "sft",
            "data_conf": {
                "training_data": {
                    "type": "CONCATENATION",
                    "datasets": [{"path": str(train_data_destination)}],
                },
                "validation_data": validation_data,
                "chat_template_name": DEFAULT_CHAT_TEMPLATE,
                "missing_pad_token_strategy": DEFAULT_MISSING_PAD_TOKEN_STRATEGY,
            },
            "training_args": training_args,
            "overrides": {
                "lr_multiplier": 1.0,
                "lr_batch_size_scaling": "none",
            },
            "batchsize_conf": {"enable_auto_batch_size": False},
            "peft_conf": peft_conf,
            "quant_conf": {"quantization_type": "no-quantization"},
            "run_conf": {
                "model": str(model_destination),
                "final_checkpoint_name": ("checkpoint-final-adapter" if metadata.is_adapter else "checkpoint-final"),
                "model_args": model_args,
                "resume_from_checkpoint": "auto",
            },
            "sft_args": {
                "max_seq_length": defaults.max_context_length or 2048,
            },
        }

        finetuning_config_override = backend_options.get("finetuning_config")
        if not isinstance(finetuning_config_override, dict):
            return base_finetuning_config
        return merge_engine_args(base_finetuning_config, finetuning_config_override)

    def _parse_backend_options(
        self,
        backend_options: dict[str, Any],
        *,
        num_accelerators: int,
    ) -> AimfttkBackendOptions:
        supported_keys = {
            "distributed_type",
            "merge_adapter",
            "finetuning_config",
        }
        unknown_keys = sorted(set(backend_options) - supported_keys)
        if unknown_keys:
            formatted = ", ".join(unknown_keys)
            raise ValueError(f"Unsupported {self.name} engine args: {formatted}.")

        distributed_type_raw = self._optional_string(
            backend_options.get("distributed_type"),
            key="distributed_type",
        ) or (
            AimfttkDistributedType.AUTO_SINGLE_PROCESS.value
            if num_accelerators == 1
            else AimfttkDistributedType.AUTO_DEEPSPEED_STAGE1.value
        )
        distributed_type: AimfttkDistributedType | str
        try:
            distributed_type = AimfttkDistributedType(distributed_type_raw)
        except ValueError as exc:
            custom_accelerate_config_path = Path(distributed_type_raw).expanduser()
            if custom_accelerate_config_path.is_file():
                distributed_type = str(custom_accelerate_config_path)
            else:
                supported = ", ".join(SUPPORTED_DISTRIBUTED_TYPES)
                raise ValueError(
                    f"{self.name} engine arg 'distributed_type' must be one of: {supported}, "
                    f"or a path to an existing accelerate config file."
                ) from exc

        parsed: AimfttkBackendOptions = {
            "distributed_type": distributed_type,
            "merge_adapter": self._optional_bool(
                backend_options.get("merge_adapter"),
                key="merge_adapter",
            )
            or False,
            "finetuning_config": self._optional_mapping(
                backend_options.get("finetuning_config"),
                key="finetuning_config",
            )
            or {},
        }
        return parsed

    def _optional_string(self, value: Any, *, key: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{self.name} engine arg '{key}' must be a string.")
        stripped = value.strip()
        if not stripped:
            raise ValueError(f"{self.name} engine arg '{key}' must not be empty.")
        return stripped

    def _optional_bool(self, value: Any, *, key: str) -> bool | None:
        if value is None:
            return None
        if not isinstance(value, bool):
            raise ValueError(f"{self.name} engine arg '{key}' must be a boolean.")
        return value

    def _optional_mapping(self, value: Any, *, key: str) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError(f"{self.name} engine arg '{key}' must be an object.")
        return dict(value)

    def _build_entrypoint_script(
        self,
        *,
        merge_adapter: bool,
        model_destination: PurePosixPath,
        output_root: PurePosixPath,
        config_mount_destination: PurePosixPath,
    ) -> str:
        checkpoints_dir = output_root / "checkpoints"
        final_checkpoint_dir = output_root / "checkpoint-final"
        final_adapter_dir = output_root / "checkpoint-final-adapter"
        checkpoint_new_basemodel_dir = checkpoints_dir / "checkpoint-new-basemodel"
        final_adapter_from_checkpoint_dir = checkpoints_dir / "checkpoint-final-adapter"
        final_checkpoint_from_checkpoint_dir = checkpoints_dir / "checkpoint-final"
        merge_adapter_block = ""
        if merge_adapter:
            merge_adapter_block = f"""
    merge_base={model_destination}
    if [ -d {checkpoint_new_basemodel_dir} ]; then
        merge_base={checkpoint_new_basemodel_dir}
  fi
        merge_adapter "$merge_base" {final_adapter_dir} {final_checkpoint_dir}
"""
        return f"""#!/usr/bin/env bash
set -euo pipefail

replace_output_dir() {{
    src="$1"
    dst="$2"

    if [ ! -d "$src" ]; then
        return
    fi

    rm -rf "$dst"
    mv "$src" "$dst"
}}

if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi
fi

echo 'Starting training process'
accelerate launch \
    --config_file {config_mount_destination / "accelerate_config.yaml"} \
    --no-python \
    aimfttk --num-preprocess-workers 4 sft {config_mount_destination / "finetuning_config.yaml"}

if [ -d {final_adapter_from_checkpoint_dir} ]; then
  echo 'Running adapter post-processing'
        replace_output_dir {final_adapter_from_checkpoint_dir} {final_adapter_dir}
        create_vllm_compatible_adapter --training-config {config_mount_destination / "finetuning_config.yaml"} {final_adapter_dir}
{merge_adapter_block}fi

replace_output_dir {final_checkpoint_from_checkpoint_dir} {final_checkpoint_dir}

if [ -f {model_destination / "preprocessor_config.json"} ]; then
    copied_preprocessor=false
    if [ -d {final_checkpoint_dir} ]; then
        cp {model_destination / "preprocessor_config.json"} {final_checkpoint_dir / "preprocessor_config.json"}
        echo 'Copied preprocessor_config.json to checkpoint-final'
        copied_preprocessor=true
    fi

    if [ -d {final_adapter_dir} ]; then
        cp {model_destination / "preprocessor_config.json"} {final_adapter_dir / "preprocessor_config.json"}
        echo 'Copied preprocessor_config.json to checkpoint-final-adapter'
        copied_preprocessor=true
    fi

    if [ "$copied_preprocessor" = false ]; then
        echo 'No final checkpoint directory found, skipping preprocessor_config.json copy'
    fi
else
  echo 'No preprocessor_config.json found in basemodel, skipping copy'
fi

echo 'Training complete'
"""


def _to_yaml(value: Any) -> str:
    return "\n".join(_yaml_lines(value)) + "\n"


def _to_posix_path(path: PurePosixPath | Any) -> PurePosixPath:
    return PurePosixPath(path.as_posix())


def _supports_use_cache_model_arg(aim_model_id: str) -> bool:
    normalized_model_id = aim_model_id.strip().lower()
    return not normalized_model_id.startswith(USE_CACHE_INCOMPATIBLE_MODEL_PREFIXES)


def _yaml_lines(value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        if not value:
            return [f"{prefix}{{}}"]

        lines: list[str] = []
        for key, nested_value in value.items():
            if _is_scalar(nested_value):
                lines.append(f"{prefix}{key}: {_yaml_scalar(nested_value)}")
                continue

            if isinstance(nested_value, list) and not nested_value:
                lines.append(f"{prefix}{key}: []")
                continue

            if isinstance(nested_value, dict) and not nested_value:
                lines.append(f"{prefix}{key}: {{}}")
                continue

            lines.append(f"{prefix}{key}:")
            lines.extend(_yaml_lines(nested_value, indent + 2))
        return lines

    if isinstance(value, list):
        if not value:
            return [f"{prefix}[]"]

        lines = []
        for item in value:
            if _is_scalar(item):
                lines.append(f"{prefix}- {_yaml_scalar(item)}")
                continue

            if isinstance(item, list) and not item:
                lines.append(f"{prefix}- []")
                continue

            if isinstance(item, dict) and not item:
                lines.append(f"{prefix}- {{}}")
                continue

            lines.append(f"{prefix}-")
            lines.extend(_yaml_lines(item, indent + 2))
        return lines

    return [f"{prefix}{_yaml_scalar(value)}"]


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value)
