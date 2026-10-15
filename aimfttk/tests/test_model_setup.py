# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import pytest
import torch
import transformers

from aimfttk.model import _init_new_rows_by_avg_sample, get_model

try:
    # Use a deterministic SPD matrix so availability detection is stable across runs.
    torch.linalg.cholesky(torch.eye(3))
    CHOLESKY_AVAILABLE = True
except RuntimeError:
    CHOLESKY_AVAILABLE = False


@pytest.mark.parametrize("model_class_name", ["AutoModelForCausalLM", "AutoModelForImageTextToText"])
def test_get_model_uses_configured_model_class(monkeypatch, model_class_name):
    calls = []

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, model_name_or_path, **kwargs):
            calls.append((model_name_or_path, kwargs))
            return cls

    monkeypatch.setattr(transformers, model_class_name, FakeModelClass)

    model = get_model("model-path", {"model_class": model_class_name, "dtype": "auto"})

    assert model is FakeModelClass
    assert calls == [("model-path", {"dtype": "auto"})]


@pytest.mark.skipif(not CHOLESKY_AVAILABLE, reason="Cholesky decomposition not available in this PyTorch installation")
def test_init_new_rows_by_avg_sample_constant_first_rows():
    weights = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.2, 1.2, 1.2],
            [0.8, 0.8, 0.8],
        ]
    )
    new_weights = torch.zeros((5, 3))
    new_weights[0:3, :] = weights
    _init_new_rows_by_avg_sample(new_weights, num_new_rows=2)
    assert torch.equal(weights, new_weights[0:3, :])


@pytest.mark.skipif(not CHOLESKY_AVAILABLE, reason="Cholesky decomposition not available in this PyTorch installation")
def test_init_new_rows_by_avg_sample_changed_last_rows():
    weights = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.2, 1.2, 1.2],
            [0.8, 0.8, 0.8],
        ]
    )
    new_weights = torch.zeros((5, 3))
    new_weights[0:3, :] = weights
    _init_new_rows_by_avg_sample(new_weights, num_new_rows=2)
    assert not torch.equal(new_weights[3:4, :], torch.zeros((2, 3)))


@pytest.mark.skipif(not CHOLESKY_AVAILABLE, reason="Cholesky decomposition not available in this PyTorch installation")
def test_init_new_rows_by_avg_sample_close_to_mean():
    # NOTE: this test could technically fail as the new rows are sampled. Statistically this should happen like once in
    # a million years
    weights = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.2, 1.2, 1.2],
            [0.8, 0.8, 0.8],
        ]
    )
    new_weights = torch.zeros((4, 3))
    new_weights[0:3, :] = weights
    _init_new_rows_by_avg_sample(new_weights, num_new_rows=1)
    assert torch.allclose(new_weights[3, :], torch.ones((1, 3)), atol=10.0)
