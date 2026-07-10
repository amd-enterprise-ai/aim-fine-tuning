# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""This file contains tests that run the full E2E setups and train for a few updates

NOTE: These tests should move to our E2E testing framework once we have the necessary things in place (workload engine,
etc.)

TIPS:
- Inject something like this in the tests at the start to inspect the intermediate results:
    import pathlib
    tmpdir = pathlib.Path("sft_tests")
    tmpdir.mkdir()
- Note: To test DPO, we first run SFT
"""

import json
from unittest.mock import patch

import numpy as np
import yaml

import aimfttk.cli
from aimfttk.data.tokenizer import subsetup_tokenizer

sftdata = [
    {
        "dataset": "test-1",
        "id": "test-1-1",
        "messages": [
            {"role": "system", "content": "A B C"},
            {"role": "user", "content": "A B C"},
            {"role": "assistant", "content": "A B C"},
        ],
    },
]

dpodata = [
    {
        "dataset": "test-1",
        "id": "test-1-1",
        "prompt_messages": [
            {"role": "system", "content": "A B C"},
            {"role": "user", "content": "A B C"},
        ],
        "chosen_messages": [
            {"role": "assistant", "content": "A B C"},
        ],
        "rejected_messages": [
            {"role": "assistant", "content": "D E F"},
        ],
    },
]


def test_sft_llama_lora(tmpdir, tiny_llama_model_and_tokenizer):
    # 0. Config definition:
    config = """\
data_conf:
  chat_template_name: "keep-original"
  training_data:
    type: CONCATENATION
    datasets:
      - path: {train_data_path}
  validation_data:
    type: NONE
batchsize_conf:
  total_train_batch_size: 1
  max_per_device_train_batch_size: 1
training_args:
  use_cpu: True
  output_dir: {output_dir}
  bf16: true
  gradient_checkpointing: false
  gradient_checkpointing_kwargs:
    use_reentrant: false
  learning_rate: 3.0
  logging_steps: 1
  logging_strategy: "steps"
  lr_scheduler_type: "constant"
  optim: sgd
  max_steps: {num_steps}
  num_train_epochs: -1
  overwrite_output_dir: true
  push_to_hub: False
  report_to: ["none"]
  save_strategy: "steps"
  save_steps: {num_steps}
  eval_strategy: "no"
sft_args:
  max_seq_length: 128
peft_conf:
  peft_type: "LORA"
  task_type: "CAUSAL_LM"
  peft_kwargs:
    r: 8
    lora_alpha: 4
    lora_dropout: 0.1
    target_modules:
      - up_proj
      - down_proj
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - lm_head
      - embed_tokens
run_conf:
  model: {model_path}
  model_args:
    attn_implementation: "eager"
    use_cache: False
  resume_from_checkpoint: False
  determinism: "full"
"""
    num_steps = 25
    # 1. setup
    # Set paths
    train_data_path = tmpdir / "data.json"
    ckpt_dir = tmpdir / "outputs"
    config_path = tmpdir / "config.yaml"
    # Save config and data to disk
    with open(train_data_path, "w") as fo:
        fo.write(json.dumps(sftdata))
    config_filled_in = config.format(
        output_dir=str(ckpt_dir),
        train_data_path=str(train_data_path),
        num_steps=str(num_steps),
        model_path=tiny_llama_model_and_tokenizer,
    )
    with open(config_path, "w") as fo:
        fo.write(config_filled_in)

    # 2. Run
    aimfttk.cli.aimfttk_main_cli(argv=["--num-preprocess-workers", "1", "sft", str(config_path)])

    # 3. Inspect results!
    with open(ckpt_dir / f"checkpoint-{num_steps}" / "trainer_state.json") as fi:
        trainer_state = json.loads(fi.read())
    # At the start, loss should be around -log(1/1024)~6.93, (1024 is the vocab size of our tiny model)
    #  see: http://karpathy.github.io/2019/04/25/recipe/  -> verify loss @ init
    assert np.isclose(trainer_state["log_history"][0]["loss"], 6.9, atol=0.8)
    # In current setup, loss goes at least below 2.2 by 25 steps (relaxed for platform differences)
    assert trainer_state["log_history"][-1]["loss"] < 2.2

    # LoRA setup should lead to adapter checkpoint
    assert (ckpt_dir / "checkpoint-final" / "adapter_model.safetensors").isfile()


def test_sft_llama(tmpdir, tiny_llama_model_and_tokenizer):
    # 0. Config definition:
    config = """\
data_conf:
  chat_template_name: "keep-original"
  training_data:
    type: "PRECOMPUTE_WEIGHTED_MIX"
    datasets:
      - path: {train_data_path}
  validation_data:
    type: "CONCATENATION"
    datasets:
      - path: {train_data_path}
batchsize_conf:
  total_train_batch_size: 1
  max_per_device_train_batch_size: 1
training_args:
  use_cpu: True
  output_dir: {output_dir}
  bf16: true
  gradient_checkpointing: false
  gradient_checkpointing_kwargs:
    use_reentrant: false
  learning_rate: 1.0
  logging_steps: 1
  logging_strategy: "steps"
  lr_scheduler_type: "constant"
  optim: sgd
  max_steps: {num_steps}
  num_train_epochs: -1
  overwrite_output_dir: true
  push_to_hub: False
  report_to: ["none"]
  save_strategy: "steps"
  save_steps: 1
  eval_strategy: "steps"
  eval_steps: 10
sft_args:
  max_seq_length: 128
peft_conf:
  peft_type: "NO_PEFT"
run_conf:
  model: {model_path}
  model_args:
    attn_implementation: "eager"
    use_cache: False
  resume_from_checkpoint: False
  determinism: "full"
"""
    num_steps = 25
    # 1. setup
    # Set paths

    train_data_path = tmpdir / "data.json"
    ckpt_dir = tmpdir / "outputs"
    config_path = tmpdir / "config.yaml"
    # Save config and data to disk
    with open(train_data_path, "w") as fo:
        fo.write(json.dumps(sftdata))
    config_filled_in = config.format(
        output_dir=str(ckpt_dir),
        train_data_path=str(train_data_path),
        num_steps=str(num_steps),
        model_path=tiny_llama_model_and_tokenizer,
    )
    with open(config_path, "w") as fo:
        fo.write(config_filled_in)

    # 2. Run
    aimfttk.cli.aimfttk_main_cli(argv=["--num-preprocess-workers", "1", "sft", str(config_path)])

    # 3. Inspect results!
    with open(ckpt_dir / f"checkpoint-{num_steps}" / "trainer_state.json") as fi:
        trainer_state = json.loads(fi.read())

    # At the start, loss should be around -log(1/1024)~6.93, (1024 is the vocab size of our tiny model)
    #  see: http://karpathy.github.io/2019/04/25/recipe/  -> verify loss @ init
    assert np.isclose(trainer_state["log_history"][0]["loss"], 6.9, atol=0.8)
    # In current setup, loss goes at least below 2.2 by 25 steps (relaxed for platform differences)
    assert trainer_state["log_history"][-1]["loss"] < 2.2


def intercept_tokenizer_call(*args, overwrite_chat_template=False, **kwargs):
    return subsetup_tokenizer(*args, overwrite_chat_template=True, **kwargs)


@patch("aimfttk.dpo.subsetup_tokenizer", intercept_tokenizer_call)
def test_dpo_llama(tmpdir, tiny_llama_model_and_tokenizer):
    # 0. DPO Config definition:
    dpo_config = """\
data_conf:
  chat_template_name: "mistral-with-system"
  training_data:
    type: CONCATENATION
    datasets:
      - path: {train_data_path}
  validation_data:
    type: NONE
batchsize_conf:
  total_train_batch_size: 1
  max_per_device_train_batch_size: 1
training_args:
  use_cpu: True
  output_dir: {output_dir}
  bf16: true
  gradient_checkpointing: false
  gradient_checkpointing_kwargs:
    use_reentrant: false
  learning_rate: 1.5
  logging_steps: 1
  logging_strategy: "steps"
  lr_scheduler_type: "constant"
  optim: sgd
  max_steps: {num_steps}
  num_train_epochs: -1
  overwrite_output_dir: true
  push_to_hub: False
  report_to: ["none"]
  save_strategy: "steps"
  save_steps: 1
  eval_strategy: "no"
  remove_unused_columns: false
  beta: 0.01
  loss_type: "sigmoid"
peft_conf:
  peft_type: "NO_PEFT"
run_conf:
  model: {model_path}
  model_args:
    attn_implementation: "eager"
    use_cache: False
  resume_from_checkpoint: False
  determinism: "full"
"""
    num_steps = 25
    # 1. setup
    # Set paths
    dpotmpdir = tmpdir / "dpo"
    dpotmpdir.mkdir()
    dpo_train_data_path = dpotmpdir / "data.json"
    dpo_ckpt_dir = dpotmpdir / "outputs"
    dpo_config_path = dpotmpdir / "config.yaml"
    # Save config and data to disk
    with open(dpo_train_data_path, "w") as fo:
        fo.write(json.dumps(dpodata))
    dpo_config_filled_in = dpo_config.format(
        output_dir=str(dpo_ckpt_dir),
        train_data_path=str(dpo_train_data_path),
        num_steps=str(num_steps),
        model_path=tiny_llama_model_and_tokenizer,
    )
    with open(dpo_config_path, "w") as fo:
        fo.write(dpo_config_filled_in)

    # 2. Run DPO
    aimfttk.cli.aimfttk_main_cli(argv=["--num-preprocess-workers", "1", "dpo", str(dpo_config_path)])

    # 3. Inspect DPO results!
    with open(dpo_ckpt_dir / f"checkpoint-{num_steps}" / "trainer_state.json") as fi:
        trainer_state = json.loads(fi.read())
    # The margins start close to zero
    assert np.isclose(trainer_state["log_history"][0]["rewards/margins"], 0.0, atol=0.05)
    # But they rise to >0.4 by 25 steps (relaxed for platform differences)
    assert trainer_state["log_history"][-1]["rewards/margins"] > 0.4


def test_sft_and_dpo_llama(tmpdir, tiny_llama_model_and_tokenizer):
    # 0. Config definition:
    sft_config = """\
data_conf:
  chat_template_name: "keep-original"
  training_data:
    type: CONCATENATION
    datasets:
      - path: {train_data_path}
  validation_data:
    type: NONE
batchsize_conf:
  total_train_batch_size: 1
  max_per_device_train_batch_size: 1
training_args:
  use_cpu: True
  output_dir: {output_dir}
  bf16: true
  gradient_checkpointing: false
  gradient_checkpointing_kwargs:
    use_reentrant: false
  learning_rate: 0.5
  logging_steps: 1
  logging_strategy: "steps"
  lr_scheduler_type: "constant"
  optim: sgd
  max_steps: {num_steps}
  num_train_epochs: -1
  overwrite_output_dir: true
  push_to_hub: False
  report_to: ["none"]
  save_strategy: "steps"
  save_steps: 1
  eval_strategy: "no"
sft_args:
  max_seq_length: 128
peft_conf:
  peft_type: "NO_PEFT"
run_conf:
  model: {model_path}
  model_args:
    attn_implementation: "eager"
    use_cache: False
  resume_from_checkpoint: False
  determinism: "full"
"""
    num_steps = 25
    # 1. setup
    # Set paths
    sft_train_data_path = tmpdir / "data.json"
    sft_ckpt_dir = tmpdir / "outputs"
    sft_config_path = tmpdir / "config.yaml"
    # Save config and data to disk
    with open(sft_train_data_path, "w") as fo:
        fo.write(json.dumps(sftdata))
    sft_config_filled_in = sft_config.format(
        output_dir=str(sft_ckpt_dir),
        train_data_path=str(sft_train_data_path),
        num_steps=str(num_steps),
        model_path=tiny_llama_model_and_tokenizer,
    )
    with open(sft_config_path, "w") as fo:
        fo.write(sft_config_filled_in)

    # 2. Run SFT
    aimfttk.cli.aimfttk_main_cli(argv=["--num-preprocess-workers", "1", "sft", str(sft_config_path)])

    # # 3. Inspect SFT results!
    with open(sft_ckpt_dir / f"checkpoint-{num_steps}" / "trainer_state.json") as fi:
        trainer_state = json.loads(fi.read())
    # At the start, loss should be around -log(1/1024)~6.93, (1024 is the vocab size of our tiny model)
    #  see: http://karpathy.github.io/2019/04/25/recipe/  -> verify loss @ init
    assert np.isclose(trainer_state["log_history"][0]["loss"], 6.9, atol=0.8)
    # In current setup, loss goes at least below 1.0 by 25 steps
    assert trainer_state["log_history"][-1]["loss"] < 1.0

    # ############################# DPO STARTS HERE ################################ #
    # 4. DPO Config definition:
    dpo_config = """\
data_conf:
  chat_template_name: "keep-original"
  training_data:
    type: CONCATENATION
    datasets:
      - path: {train_data_path}
  validation_data:
    type: NONE
batchsize_conf:
  total_train_batch_size: 1
  max_per_device_train_batch_size: 1
training_args:
  use_cpu: True
  output_dir: {output_dir}
  bf16: true
  gradient_checkpointing: false
  gradient_checkpointing_kwargs:
    use_reentrant: false
  learning_rate: 0.5
  logging_steps: 1
  logging_strategy: "steps"
  lr_scheduler_type: "constant"
  optim: sgd
  max_steps: {num_steps}
  num_train_epochs: -1
  overwrite_output_dir: true
  push_to_hub: False
  report_to: ["none"]
  save_strategy: "steps"
  save_steps: 1
  eval_strategy: "no"
  remove_unused_columns: false
  beta: 0.01
  loss_type: "sigmoid"
peft_conf:
  peft_type: "NO_PEFT"
run_conf:
  model: {sft_checkpoint}
  model_args:
    attn_implementation: "eager"
    use_cache: False
  resume_from_checkpoint: False
  determinism: "full"
"""
    # note: potential to redefine
    num_steps = 25
    # 5. setup
    # Set paths
    dpotmpdir = tmpdir / "dpo"
    dpotmpdir.mkdir()
    dpo_train_data_path = dpotmpdir / "data.json"
    dpo_ckpt_dir = dpotmpdir / "outputs"
    dpo_config_path = dpotmpdir / "config.yaml"
    # Save config and data to disk
    with open(dpo_train_data_path, "w") as fo:
        fo.write(json.dumps(dpodata))
    dpo_config_filled_in = dpo_config.format(
        output_dir=str(dpo_ckpt_dir),
        train_data_path=str(dpo_train_data_path),
        num_steps=str(num_steps),
        sft_checkpoint=str(sft_ckpt_dir / "checkpoint-final"),
    )
    with open(dpo_config_path, "w") as fo:
        fo.write(dpo_config_filled_in)

    # 6. Run DPO
    aimfttk.cli.aimfttk_main_cli(argv=["--num-preprocess-workers", "1", "dpo", str(dpo_config_path)])

    # 7. Inspect DPO results!
    with open(dpo_ckpt_dir / f"checkpoint-{num_steps}" / "trainer_state.json") as fi:
        trainer_state = json.loads(fi.read())
    # The margins start close to zero
    assert np.isclose(trainer_state["log_history"][0]["rewards/margins"], 0.0, atol=0.05)
    # But they rise to >0.2 by 25 steps
    assert trainer_state["log_history"][-1]["rewards/margins"] > 0.2


def test_generate(tmpdir, tiny_gemma2_model_and_tokenizer):
    # 0. Config definition:
    generate_config = """\
model_conf:
  model: {model_path}
  model_args:
    attn_implementation: "eager"
    use_cache: False
    device_map: cpu
prompt_conf:
  type: "open-input"
  input: "Hi my name is "
hf_gen_params:
  do_sample: false
  max_new_tokens: 4
"""
    config_filled_in = generate_config.format(model_path=tiny_gemma2_model_and_tokenizer)
    with open(tmpdir / "gen_conf.yaml", "w") as fo:
        fo.write(config_filled_in)
    aimfttk.cli.generation_cli([str(tmpdir / "gen_conf.yaml"), str(tmpdir / "generated_output.yaml")])
    with open(tmpdir / "generated_output.yaml") as fin:
        list_of_outputs = yaml.safe_load(fin)
    # Our tiny random model generates an empty string (deterministic output)
    assert list_of_outputs[0]["answer"] == ""
