# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import os
import random
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import click

from aim_fine_tune.errors import DownloadError
from aim_fine_tune.models import DownloadRequest


def run_download_request(request: DownloadRequest) -> dict[str, str]:
    downloaded: dict[str, str] = {}

    if request.hf_model_id is not None:
        download_hf_model(request)
        assert request.model_output_path is not None
        downloaded["model_path"] = str(request.model_output_path)

    if request.dataset_s3_uri is not None:
        download_dataset_from_s3(request)
        assert request.dataset_output_path is not None
        downloaded["dataset_path"] = str(request.dataset_output_path)

        if request.split_ratio is not None and request.split_ratio > 0:
            if request.validation_data_output_path is None:
                raise DownloadError("AIMFT_VALIDATION_DATA_OUTPUT_PATH is required when AIMFT_SPLIT_RATIO is set.")
            split_dataset(request)
            downloaded["validation_path"] = str(request.validation_data_output_path)

    return downloaded


def download_hf_model(request: DownloadRequest) -> None:
    if request.hf_model_id is None or request.model_output_path is None:
        raise DownloadError("Hugging Face model id and output path are required.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise DownloadError(
            "huggingface_hub is required for model downloads. Install the CLI package with its runtime dependencies."
        ) from exc

    request.model_output_path.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": request.hf_model_id,
        "local_dir": str(request.model_output_path),
        "local_dir_use_symlinks": False,
    }
    if request.model_cache_dir is not None:
        request.model_cache_dir.mkdir(parents=True, exist_ok=True)
        kwargs["cache_dir"] = str(request.model_cache_dir)
    if request.hf_token is not None:
        kwargs["token"] = request.hf_token

    snapshot_download(**kwargs)


def download_dataset_from_s3(request: DownloadRequest) -> None:
    if request.dataset_s3_uri is None or request.dataset_output_path is None:
        raise DownloadError("Dataset S3 URI and output path are required.")

    bucket, key_prefix = parse_s3_uri(request.dataset_s3_uri)
    client = build_s3_client(request)
    object_keys = list_s3_keys(client, bucket, key_prefix)
    if not object_keys:
        raise DownloadError(f"No dataset objects found at '{request.dataset_s3_uri}'.")

    request.dataset_output_path.mkdir(parents=True, exist_ok=True)
    for object_key in object_keys:
        target_path = request.dataset_output_path / relative_download_path(key_prefix, object_key)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, object_key, str(target_path))


def split_dataset(request: DownloadRequest) -> None:
    if request.dataset_output_path is None or request.validation_data_output_path is None:
        raise DownloadError("Dataset and validation output paths are required for splitting.")
    if request.split_ratio is None or request.split_ratio <= 0 or request.split_ratio >= 1:
        raise DownloadError("Split ratio must be between 0 and 1 (exclusive).")

    random.seed(request.seed)
    request.validation_data_output_path.mkdir(parents=True, exist_ok=True)

    jsonl_files = list(request.dataset_output_path.glob("*.jsonl"))
    if not jsonl_files:
        raise DownloadError(f"No JSONL files found in {request.dataset_output_path} for splitting.")

    for jsonl_file in jsonl_files:
        lines = jsonl_file.read_text(encoding="utf-8").strip().split("\n")
        if not lines:
            continue

        random.shuffle(lines)
        split_index = int(len(lines) * (1 - request.split_ratio))

        train_lines = lines[:split_index]
        validation_lines = lines[split_index:]

        validation_file = request.validation_data_output_path / jsonl_file.name
        jsonl_file.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
        validation_file.write_text("\n".join(validation_lines) + "\n", encoding="utf-8")


def build_s3_client(request: DownloadRequest) -> Any:
    try:
        import boto3
    except ImportError as exc:
        raise DownloadError(
            "boto3 is required for dataset downloads. Install the CLI package with its runtime dependencies."
        ) from exc

    if request.s3_endpoint_url is None:
        raise DownloadError("An S3 endpoint URL is required for dataset downloads.")
    if request.s3_access_key_id is None or request.s3_secret_access_key is None:
        raise DownloadError("S3 access key id and secret access key are required for dataset downloads.")

    return boto3.client(
        "s3",
        endpoint_url=request.s3_endpoint_url,
        aws_access_key_id=request.s3_access_key_id,
        aws_secret_access_key=request.s3_secret_access_key,
        aws_session_token=request.s3_session_token,
        region_name=request.s3_region,
    )


def list_s3_keys(client: Any, bucket: str, key_prefix: str) -> list[str]:
    paginator = client.get_paginator("list_objects_v2")
    normalized_prefix = key_prefix.strip("/")
    prefix = normalized_prefix if not normalized_prefix else f"{normalized_prefix}/"
    candidate_prefixes = [normalized_prefix] if normalized_prefix else [""]
    if prefix and prefix != normalized_prefix:
        candidate_prefixes.append(prefix)

    object_keys: list[str] = []
    seen: set[str] = set()
    for candidate in candidate_prefixes:
        for page in paginator.paginate(Bucket=bucket, Prefix=candidate):
            for entry in page.get("Contents", []):
                object_key = entry["Key"]
                if object_key.endswith("/"):
                    continue
                if normalized_prefix and object_key != normalized_prefix and not object_key.startswith(prefix):
                    continue
                if object_key in seen:
                    continue
                seen.add(object_key)
                object_keys.append(object_key)
    return object_keys


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise DownloadError(f"Expected an S3 URI like s3://bucket/path, got '{uri}'.")
    key_prefix = parsed.path.lstrip("/")
    if not key_prefix:
        raise DownloadError("Dataset S3 URI must include a bucket path.")
    return parsed.netloc, key_prefix


def relative_download_path(request_prefix: str, object_key: str) -> Path:
    normalized_prefix = request_prefix.strip("/")
    if object_key == normalized_prefix:
        return Path(PurePosixPath(object_key).name)

    relative_key = object_key
    if normalized_prefix:
        prefix_with_slash = f"{normalized_prefix}/"
        if object_key.startswith(prefix_with_slash):
            relative_key = object_key[len(prefix_with_slash) :]

    if not relative_key:
        relative_key = PurePosixPath(object_key).name
    return Path(PurePosixPath(relative_key))


# ---------------------------------------------------------------------------
# CLI helpers for the debug-download command
# ---------------------------------------------------------------------------

_DOWNLOAD_STRING_ENV_KEYS = {
    "AIMFT_HF_MODEL_ID",
    "AIMFT_MODEL_OUTPUT_PATH",
    "AIMFT_MODEL_CACHE_DIR",
    "AIMFT_HF_TOKEN",
    "HF_TOKEN",
    "AIMFT_DATASET_S3_URI",
    "AIMFT_DATASET_OUTPUT_PATH",
    "AIMFT_VALIDATION_DATA_OUTPUT_PATH",
    "AIMFT_S3_ENDPOINT_URL",
    "AWS_ENDPOINT_URL",
    "BUCKET_STORAGE_HOST",
    "AIMFT_S3_ACCESS_KEY_ID",
    "AWS_ACCESS_KEY_ID",
    "BUCKET_STORAGE_ACCESS_KEY",
    "AIMFT_S3_SECRET_ACCESS_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "BUCKET_STORAGE_SECRET_KEY",
    "AIMFT_S3_SESSION_TOKEN",
    "AWS_SESSION_TOKEN",
    "AIMFT_S3_REGION",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
}


def _merged_value(cli_value: Any, env: dict[str, str], env_key: str, default: Any = None) -> Any:
    if cli_value is not None:
        return cli_value
    if env_key in env:
        return _decode_env_value(env_key, env[env_key])
    return default


def _merged_values(cli_value: Any, env: dict[str, str], *env_keys: str, default: Any = None) -> Any:
    if cli_value is not None:
        return cli_value
    for env_key in env_keys:
        if env_key in env:
            return _decode_env_value(env_key, env[env_key])
    return default


def _decode_env_value(env_key: str, raw_value: str) -> Any:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        if env_key in _DOWNLOAD_STRING_ENV_KEYS:
            return raw_value
        raise DownloadError(f"Environment variable {env_key} must be valid JSON or a raw string value.")


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DownloadError("Expected a string value.")
    text = value.strip()
    return text or None


def _coerce_path(value: Any, name: str) -> Path:
    if isinstance(value, Path):
        return value.expanduser()
    if not isinstance(value, str):
        raise DownloadError(f"{name} must be a string path.")
    text = value.strip()
    if not text:
        raise DownloadError(f"{name} must not be empty.")
    return Path(text).expanduser()


def _coerce_optional_directory_path(value: Any, name: str) -> Path | None:
    if value is None:
        return None
    path = _coerce_path(value, name)
    if path.exists() and not path.is_dir():
        raise DownloadError(f"{name} must point to a directory.")
    return path


def _coerce_optional_float(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DownloadError(f"{name} must be a float.")
    return float(value)


def _coerce_int(value: Any, name: str) -> int:
    if value is None:
        raise DownloadError(f"{name} is required.")
    if isinstance(value, bool) or not isinstance(value, int):
        raise DownloadError(f"{name} must be an integer.")
    return value


def build_download_request(cli_values: dict[str, Any], env: dict[str, str]) -> DownloadRequest:
    hf_model_id = _coerce_optional_string(_merged_value(cli_values["hf_model_id"], env, "AIMFT_HF_MODEL_ID"))
    model_output_path = _coerce_optional_directory_path(
        _merged_value(cli_values["model_output_path"], env, "AIMFT_MODEL_OUTPUT_PATH"),
        "AIMFT_MODEL_OUTPUT_PATH",
    )
    model_cache_dir = _coerce_optional_directory_path(
        _merged_value(cli_values["model_cache_dir"], env, "AIMFT_MODEL_CACHE_DIR"),
        "AIMFT_MODEL_CACHE_DIR",
    )
    hf_token = _coerce_optional_string(
        _merged_values(
            cli_values["hf_token"],
            env,
            "AIMFT_HF_TOKEN",
            "HF_TOKEN",
        )
    )
    dataset_s3_uri = _coerce_optional_string(_merged_value(cli_values["dataset_s3_uri"], env, "AIMFT_DATASET_S3_URI"))
    dataset_output_path = _coerce_optional_directory_path(
        _merged_value(cli_values["dataset_output_path"], env, "AIMFT_DATASET_OUTPUT_PATH"),
        "AIMFT_DATASET_OUTPUT_PATH",
    )
    validation_data_output_path = _coerce_optional_directory_path(
        _merged_value(cli_values.get("validation_data_output_path"), env, "AIMFT_VALIDATION_DATA_OUTPUT_PATH"),
        "AIMFT_VALIDATION_DATA_OUTPUT_PATH",
    )
    s3_endpoint_url = _coerce_optional_string(
        _merged_values(
            cli_values["s3_endpoint_url"],
            env,
            "AIMFT_S3_ENDPOINT_URL",
            "AWS_ENDPOINT_URL",
            "BUCKET_STORAGE_HOST",
        )
    )
    s3_access_key_id = _coerce_optional_string(
        _merged_values(
            cli_values["s3_access_key_id"],
            env,
            "AIMFT_S3_ACCESS_KEY_ID",
            "AWS_ACCESS_KEY_ID",
            "BUCKET_STORAGE_ACCESS_KEY",
        )
    )
    s3_secret_access_key = _coerce_optional_string(
        _merged_values(
            cli_values["s3_secret_access_key"],
            env,
            "AIMFT_S3_SECRET_ACCESS_KEY",
            "AWS_SECRET_ACCESS_KEY",
            "BUCKET_STORAGE_SECRET_KEY",
        )
    )
    s3_session_token = _coerce_optional_string(
        _merged_values(
            cli_values["s3_session_token"],
            env,
            "AIMFT_S3_SESSION_TOKEN",
            "AWS_SESSION_TOKEN",
        )
    )
    s3_region = _coerce_optional_string(
        _merged_values(
            cli_values["s3_region"],
            env,
            "AIMFT_S3_REGION",
            "AWS_REGION",
            "AWS_DEFAULT_REGION",
        )
    )
    split_ratio = _coerce_optional_float(
        _merged_value(cli_values.get("split_ratio"), env, "AIMFT_SPLIT_RATIO"),
        "AIMFT_SPLIT_RATIO",
    )
    seed = _coerce_int(
        _merged_value(cli_values.get("seed"), env, "AIMFT_SEED", 42),
        "AIMFT_SEED",
    )

    if hf_model_id is None and dataset_s3_uri is None:
        raise DownloadError("At least one of AIMFT_HF_MODEL_ID or AIMFT_DATASET_S3_URI is required.")
    if hf_model_id is not None and model_output_path is None:
        raise DownloadError("AIMFT_MODEL_OUTPUT_PATH is required when AIMFT_HF_MODEL_ID is set.")
    if hf_model_id is None and model_output_path is not None:
        raise DownloadError("AIMFT_HF_MODEL_ID is required when AIMFT_MODEL_OUTPUT_PATH is set.")
    if dataset_s3_uri is not None and dataset_output_path is None:
        raise DownloadError("AIMFT_DATASET_OUTPUT_PATH is required when AIMFT_DATASET_S3_URI is set.")
    if dataset_s3_uri is None and dataset_output_path is not None:
        raise DownloadError("AIMFT_DATASET_S3_URI is required when AIMFT_DATASET_OUTPUT_PATH is set.")
    if dataset_s3_uri is not None and s3_endpoint_url is None:
        raise DownloadError(
            "An S3 endpoint URL is required for dataset downloads. Set AIMFT_S3_ENDPOINT_URL, AWS_ENDPOINT_URL, or BUCKET_STORAGE_HOST."
        )
    if dataset_s3_uri is not None and (s3_access_key_id is None or s3_secret_access_key is None):
        raise DownloadError(
            "S3 credentials are required for dataset downloads. Set AIMFT_S3_ACCESS_KEY_ID/AIMFT_S3_SECRET_ACCESS_KEY, AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, or BUCKET_STORAGE_ACCESS_KEY/BUCKET_STORAGE_SECRET_KEY."
        )

    return DownloadRequest(
        hf_model_id=hf_model_id,
        model_output_path=model_output_path,
        model_cache_dir=model_cache_dir,
        hf_token=hf_token,
        dataset_s3_uri=dataset_s3_uri,
        dataset_output_path=dataset_output_path,
        validation_data_output_path=validation_data_output_path,
        s3_endpoint_url=s3_endpoint_url,
        s3_access_key_id=s3_access_key_id,
        s3_secret_access_key=s3_secret_access_key,
        s3_session_token=s3_session_token,
        s3_region=s3_region,
        split_ratio=split_ratio,
        seed=seed,
    )


@click.command("debug-download")
@click.option("--hf-model-id", default=None)
@click.option(
    "--model-output-path",
    type=click.Path(path_type=Path, exists=False, file_okay=False, dir_okay=True),
    default=None,
)
@click.option(
    "--model-cache-dir",
    type=click.Path(path_type=Path, exists=False, file_okay=False, dir_okay=True),
    default=None,
)
@click.option("--hf-token", default=None)
@click.option("--dataset-s3-uri", default=None)
@click.option(
    "--dataset-output-path",
    type=click.Path(path_type=Path, exists=False, file_okay=False, dir_okay=True),
    default=None,
)
@click.option(
    "--validation-data-output-path",
    type=click.Path(path_type=Path, exists=False, file_okay=False, dir_okay=True),
    default=None,
)
@click.option("--s3-endpoint-url", default=None)
@click.option("--s3-access-key-id", default=None)
@click.option("--s3-secret-access-key", default=None)
@click.option("--s3-session-token", default=None)
@click.option("--s3-region", default=None)
@click.option("--split-ratio", type=float, default=None)
@click.option("--seed", type=int, default=None)
def debug_download_command(
    hf_model_id: str | None,
    model_output_path: Path | None,
    model_cache_dir: Path | None,
    hf_token: str | None,
    dataset_s3_uri: str | None,
    dataset_output_path: Path | None,
    validation_data_output_path: Path | None,
    s3_endpoint_url: str | None,
    s3_access_key_id: str | None,
    s3_secret_access_key: str | None,
    s3_session_token: str | None,
    s3_region: str | None,
    split_ratio: float | None,
    seed: int | None,
) -> None:
    """Download model and dataset assets into local paths."""
    cli_values = {
        "hf_model_id": hf_model_id,
        "model_output_path": model_output_path,
        "model_cache_dir": model_cache_dir,
        "hf_token": hf_token,
        "dataset_s3_uri": dataset_s3_uri,
        "dataset_output_path": dataset_output_path,
        "validation_data_output_path": validation_data_output_path,
        "s3_endpoint_url": s3_endpoint_url,
        "s3_access_key_id": s3_access_key_id,
        "s3_secret_access_key": s3_secret_access_key,
        "s3_session_token": s3_session_token,
        "s3_region": s3_region,
        "split_ratio": split_ratio,
        "seed": seed,
    }

    try:
        request = build_download_request(cli_values, dict(os.environ))
        results = run_download_request(request)
        if "model_path" in results:
            click.echo(f"Downloaded model to {results['model_path']}")
        if "dataset_path" in results:
            click.echo(f"Downloaded dataset to {results['dataset_path']}")
        if "validation_path" in results:
            click.echo(f"Split validation dataset to {results['validation_path']}")
    except (DownloadError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
