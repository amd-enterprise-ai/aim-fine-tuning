# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import filecmp
import os

import pytest
import safetensors
import torch
from peft import LoraConfig, get_peft_model_state_dict, inject_adapter_in_model

from aimfttk.utils.vllm_compatibility import remove_non_lora_layers


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 10)
        self.linear = torch.nn.Linear(10, 10)
        self.lm_head = torch.nn.Linear(10, 10)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        x = self.lm_head(x)
        return x


@pytest.fixture(scope="function")
def setup_model(tmpdir):
    adapter_model_folder = f"{tmpdir}"
    adapter_model_file_path = f"{tmpdir}/adapter_model.safetensors"
    old_adapter_file_path = f"{tmpdir}/old_with_extra_layers_adapter_model.safetensors"
    model = DummyModel()

    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        target_modules=["linear"],
    )

    model = inject_adapter_in_model(lora_config, model)

    peft_state_dict = get_peft_model_state_dict(model)
    # artificially add some extra layers to the adapter state dict
    peft_state_dict["embedding.weight"] = model.embedding.weight
    safetensors.torch.save_file(peft_state_dict, adapter_model_file_path)

    return adapter_model_folder, adapter_model_file_path, old_adapter_file_path


def test_remove_non_lora_layers(setup_model):
    """Test if the non-lora layers are removed from the model."""
    adapter_model_folder, adapter_model_file_path, old_adapter_file_path = setup_model

    remove_non_lora_layers(adapter_model_folder)

    assert os.path.isfile(old_adapter_file_path)
    state_dict = safetensors.torch.load_file(adapter_model_file_path)
    for key in state_dict.keys():
        assert "lora" in key
        assert "lm_head" not in key
        assert "embedding" not in key


def test_remove_non_lora_layers_noop(setup_model):
    """Test that the adapter is not updated if it exists."""
    adapter_model_folder, adapter_model_file_path, old_adapter_file_path = setup_model

    remove_non_lora_layers(adapter_model_folder)
    # Run the function again to check if it is a noop
    remove_non_lora_layers(adapter_model_folder)

    # assert that both files are not the same
    assert not filecmp.cmp(adapter_model_file_path, old_adapter_file_path)
