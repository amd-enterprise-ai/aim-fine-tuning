# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import peft
import pytest
import yaml
from pydantic import BaseModel, ValidationError

from aimfttk.config import (
    AIMFTTKTrainingArguments,
    DPOExperimentConfig,
    GenericPeftConfig,
    MissingPadTokenStrategy,
    RunConfig,
    SFTExperimentConfig,
    TokenizerArguments,
)
from aimfttk.config.base import BaseConfig, DisallowedInput


def test_validation():
    class TestConfig(BaseConfig):
        foo: str

    # Valid:
    assert TestConfig(**{"foo": "A string"}).foo == "A string"
    # Wrong key specified:
    with pytest.raises(ValidationError):
        TestConfig(**{"bar": "A string"})
    # Empty input:
    with pytest.raises(ValidationError):
        TestConfig()
    # Additional input:
    with pytest.raises(ValidationError):
        TestConfig(**{"foo": "A string", "bar": "Another string"})

    class NestedConfig(BaseConfig):
        test: TestConfig
        foo: str

    # Valid:
    assert NestedConfig(**{"test": {"foo": "A string"}, "foo": "A string"}).foo == "A string"
    # Nested structure is also enforced:
    with pytest.raises(ValidationError):
        NestedConfig(**{"test": "A string", "foo": "A string"})

    class DefaultConfig(BaseConfig):
        foo: str = "A string"

    # Default works:
    assert DefaultConfig().foo == "A string"
    # Can be changed:
    assert DefaultConfig(**{"foo": "Another string"}).foo == "Another string"


def test_yaml_instantiation(tmp_path):
    class TestConfig(BaseConfig):
        foo: str
        bar: int = 1

    class NestedConfig(BaseConfig):
        test: TestConfig
        fizz: int

    # Valid YAML:
    yamlstr = """
    test:
        foo: "A string"
    fizz: 1
    """
    assert NestedConfig.from_yaml(yamlstr).test.foo == "A string"
    assert NestedConfig.from_yaml(yamlstr).fizz == 1
    # Invalid YAML:
    yamlstr = """
    test:
        foo: "A string"
        buzz: 3
    fizz: 1
    """
    with pytest.raises(ValidationError):
        NestedConfig.from_yaml(yamlstr)

    # Back and forth:
    yamlstr = """
    test:
        foo: "A string"
    fizz: 1
    """
    assert NestedConfig.from_yaml(yaml.dump(NestedConfig.from_yaml(yamlstr).model_dump())).fizz == 1


def test_disallowed_input():
    class TestModel(BaseModel):
        foo: str
        bar: DisallowedInput = "Forbidden Fruit"

    # If not specified, all good:
    assert TestModel(foo="hi").foo == "hi"
    assert TestModel(foo="hi").bar == "Forbidden Fruit"

    # But User Input is not allowed
    with pytest.raises(ValidationError):
        TestModel(foo="hi", bar="world")

    # Even if it matches the default:
    with pytest.raises(ValidationError):
        TestModel(foo="hi", bar="Forbidden Fruit")


def test_hf_training_args_config():
    args = AIMFTTKTrainingArguments(output_dir="Dummy dir")
    args.model_dump()
    recreated = AIMFTTKTrainingArguments(**(args.model_dump(exclude_unset=True)))
    assert recreated.output_dir == "Dummy dir"
    assert recreated.model_dump() == args.model_dump()


def test_peft_config():
    config = GenericPeftConfig(peft_type="LORA", task_type="CAUSAL_LM")
    assert isinstance(config.get_peft_config(), peft.LoraConfig)
    config = GenericPeftConfig(
        peft_type="LORA", task_type="CAUSAL_LM", peft_kwargs={"r": 32, "target_modules": ["v_proj"]}
    )
    assert isinstance(config.get_peft_config(), peft.LoraConfig)


def test_tokenizer_args_config():
    config = TokenizerArguments(
        chat_template_name="chat-ml",
        padding_side="left",
        add_eos_token=False,
        add_bos_token=False,
        eos_token="<eos>",
        missing_pad_token_strategy="unk-repurpose",
        use_fast_tokenizer=False,
        fix_mistral_regex=True,
        tokenizer_extra_args={"foo": 1},
    )
    kwargs = config.get_tokenizer_load_kwargs()
    assert kwargs["padding_side"] == "left"
    assert kwargs["add_eos_token"] is False
    assert kwargs["add_bos_token"] is False
    assert kwargs["eos_token"] == "<eos>"
    assert kwargs["use_fast"] is False
    assert kwargs["fix_mistral_regex"] is True
    assert kwargs["foo"] == 1
    assert config.chat_template_name == "chat-ml"
    assert config.missing_pad_token_strategy == "unk-repurpose"


def test_run_config_legacy_tokenizer_fields_are_migrated():
    yamlstr = """
    model: mistralai/Mistral-7B-v0.1
    use_fast_tokenizer: false
    """
    config = RunConfig.from_yaml(yamlstr)
    assert config.tokenizer_args.use_fast_tokenizer is False


def test_legacy_data_conf_padding_side_is_migrated():
    yamlstr = """
    data_conf:
      padding_side: left
      training_data:
        type: PRECOMPUTE_WEIGHTED_MIX
        datasets:
          - path: "./Dolly-Silosplit"
            sampling_weight: 1.0
      validation_data:
        type: NONE
    batchsize_conf:
      total_train_batch_size: 1
      max_per_device_train_batch_size: 1
    training_args:
      output_dir: "./outputs"
      bf16: false
      report_to: ["none"]
    sft_args:
      max_seq_length: 128
    peft_conf:
      peft_type: "NO_PEFT"
    run_conf:
      model: mistralai/Mistral-7B-v0.1
    """
    config = SFTExperimentConfig.from_yaml(yamlstr)
    assert config.run_conf.tokenizer_args.padding_side == "left"


def test_legacy_data_conf_chat_template_name_is_migrated():
    yamlstr = """
    data_conf:
      chat_template_name: "chat-ml"
      training_data:
        type: PRECOMPUTE_WEIGHTED_MIX
        datasets:
          - path: "./Dolly-Silosplit"
            sampling_weight: 1.0
      validation_data:
        type: NONE
    batchsize_conf:
      total_train_batch_size: 1
      max_per_device_train_batch_size: 1
    training_args:
      output_dir: "./outputs"
      bf16: false
      report_to: ["none"]
    sft_args:
      max_seq_length: 128
    peft_conf:
      peft_type: "NO_PEFT"
    run_conf:
      model: mistralai/Mistral-7B-v0.1
    """
    config = SFTExperimentConfig.from_yaml(yamlstr)
    assert config.run_conf.tokenizer_args.chat_template_name == "chat-ml"


def test_legacy_data_conf_missing_pad_token_strategy_is_migrated():
    yamlstr = """
    data_conf:
      missing_pad_token_strategy: "unk-repurpose"
      training_data:
        type: PRECOMPUTE_WEIGHTED_MIX
        datasets:
          - path: "./Dolly-Silosplit"
            sampling_weight: 1.0
      validation_data:
        type: NONE
    batchsize_conf:
      total_train_batch_size: 1
      max_per_device_train_batch_size: 1
    training_args:
      output_dir: "./outputs"
      bf16: false
      report_to: ["none"]
    sft_args:
      max_seq_length: 128
    peft_conf:
      peft_type: "NO_PEFT"
    run_conf:
      model: mistralai/Mistral-7B-v0.1
    """
    config = SFTExperimentConfig.from_yaml(yamlstr)
    assert config.run_conf.tokenizer_args.missing_pad_token_strategy == "unk-repurpose"


def test_full_sft_config():
    # Written in YAML here for convenience and ease of updating
    yamlstr = """
    data_conf:
      training_data:
        type: PRECOMPUTE_WEIGHTED_MIX
        datasets:
          - path: "./Dolly-Silosplit"
            sampling_weight: 1.0
          - path: "./OASST1"
            sampling_weight: 0.5
      validation_data:
        type: CONCATENATION
        datasets:
          - path: "Dolly-Silosplit"
    batchsize_conf:
      total_train_batch_size: 512
      max_per_device_train_batch_size: 4
    training_args:
      output_dir: "/experiments/SFT_DOLLY_BASIC/Config-Development-Experiments/EXP-B-replicate-with-new-data"
      bf16: true
      gradient_checkpointing: true
      gradient_checkpointing_kwargs: {"use_reentrant": false}
      learning_rate: 2.0e-05
      logging_steps: 5
      logging_strategy: "steps"
      lr_scheduler_type: "cosine"
      max_steps: 50
      num_train_epochs: 23
      push_to_hub: False
      report_to: ["none"]
      save_strategy: "steps"
      save_steps: 5
      eval_strategy: "epoch"
    sft_args:
      max_seq_length: 2048
    peft_conf:
      peft_type: "LORA"
      task_type: "CAUSAL_LM"
      peft_kwargs:
        r: 64
        lora_alpha: 16
        lora_dropout: 0.1
        target_modules:
          - q_proj
          - k_proj
          - v_proj
          - o_proj
    run_conf:
      model: mistralai/Mistral-7B-v0.1
      tokenizer_args:
        chat_template_name: "chat-ml"
        use_fast_tokenizer: true
        fix_mistral_regex: true
      model_args:
        attn_implementation: "flash_attention_2"
        use_cache: False
      resume_from_checkpoint: False
    tracking:
      mlflow_server_uri: "file:///mlruns"
      experiment_name: default
      hf_mlflow_log_artifacts: "False"
    """
    # This just tests the loading - any failure will raise a validation error
    SFTExperimentConfig.from_yaml(yamlstr)


def test_full_config_resolve_training_arguments():
    # This test is for the special resolve_training_args() call that is done when ready to start an experiment
    # with a full experiment config.
    # Written in YAML here for convenience and ease of updating
    yamlstr = """
    data_conf:
      training_data:
        type: PRECOMPUTE_WEIGHTED_MIX
        datasets:
          - path: "./Dolly-Silosplit"
            sampling_weight: 1.0
          - path: "./OASST1"
            sampling_weight: 0.5
      validation_data:
        type: CONCATENATION
        datasets:
          - path: "Dolly-Silosplit"
    batchsize_conf:
      total_train_batch_size: 512
      max_per_device_train_batch_size: 4
    training_args:
      output_dir: "/experiments/SFT_DOLLY_BASIC/Config-Development-Experiments/EXP-B-replicate-with-new-data"
      bf16: false
      gradient_checkpointing: true
      gradient_checkpointing_kwargs: {"use_reentrant": false}
      learning_rate: 2.0e-05
      logging_steps: 5
      logging_strategy: "steps"
      lr_scheduler_type: "cosine"
      max_steps: 50
      num_train_epochs: 23
      push_to_hub: False
      report_to: ["none"]
      save_strategy: "steps"
      save_steps: 5
      eval_strategy: "epoch"
    sft_args:
      max_seq_length: 2048
    peft_conf:
      peft_type: "LORA"
      task_type: "CAUSAL_LM"
      peft_kwargs:
        r: 64
        lora_alpha: 16
        lora_dropout: 0.1
        target_modules:
          - q_proj
          - k_proj
          - v_proj
          - o_proj
    run_conf:
      model: mistralai/Mistral-7B-v0.1
      tokenizer_args:
        chat_template_name: "chat-ml"
        use_fast_tokenizer: true
      model_args:
        attn_implementation: "flash_attention_2"
        use_cache: False
      resume_from_checkpoint: False
    tracking:
      mlflow_server_uri: "file:///mlruns"
      experiment_name: default
      hf_mlflow_log_artifacts: "False"
    """
    config = SFTExperimentConfig.from_yaml(yamlstr)
    assert config.training_args.per_device_train_batch_size == 8  # Transformers default.
    config.resolve_training_args()
    assert config.training_args.per_device_train_batch_size == 4
    assert config.training_args.gradient_accumulation_steps == 128
    # We can find the HuggingFace extra keys, too:
    config.training_args.distributed_state


def test_dpo_config():
    # Written in YAML here for convenience and ease of updating
    yamlstr = """
    data_conf:
      training_data:
        type: PRECOMPUTE_WEIGHTED_MIX
        datasets:
          - path: "./Dolly-Silosplit"
            sampling_weight: 1.0
          - path: "./OASST1"
            sampling_weight: 0.5
      validation_data:
        type: CONCATENATION
        datasets:
          - path: "Dolly-Silosplit"
    batchsize_conf:
      total_train_batch_size: 512
      max_per_device_train_batch_size: 4
    training_args:
      output_dir: "/experiments/SFT_DOLLY_BASIC/Config-Development-Experiments/EXP-B-replicate-with-new-data"
      bf16: true
      gradient_checkpointing: true
      gradient_checkpointing_kwargs: {"use_reentrant": false}
      learning_rate: 2.0e-05
      logging_steps: 5
      logging_strategy: "steps"
      lr_scheduler_type: "cosine"
      max_steps: 50
      num_train_epochs: 23
      push_to_hub: False
      report_to: ["none"]
      save_strategy: "steps"
      save_steps: 5
      eval_strategy: "epoch"
      beta: 0.01
      loss_type: ["sigmoid"]
      max_prompt_length: 123
    peft_conf:
      peft_type: "LORA"
      task_type: "CAUSAL_LM"
      peft_kwargs:
        r: 64
        lora_alpha: 16
        lora_dropout: 0.1
        target_modules:
          - q_proj
          - k_proj
          - v_proj
          - o_proj
    run_conf:
      model: mistralai/Mistral-7B-v0.1
      tokenizer_args:
        chat_template_name: "chat-ml"
        fix_mistral_regex: true
      model_args:
        attn_implementation: "flash_attention_2"
        use_cache: False
      resume_from_checkpoint: False
    """
    # This just tests the loading - any failure will raise a validation error
    DPOExperimentConfig.from_yaml(yamlstr)
