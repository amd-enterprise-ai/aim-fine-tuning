# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Huggingface-based generate utility

It is sometimes useful to use HuggingFace-based generation, as HuggingFace should support all the things that we also
support in training. For instance, bitsandbytes quantization is not available on vLLM

The GenerationConfig should look something like:

>>> yamlstr='''
... prompt_conf:
...   type: "eval-data"
...   data:
...     path: "/datasets/illy_qa_data_topk_2_evalfmt.jsonl"
... hf_gen_params:
...   max_length: 2048
...   max_new_tokens: 512
...   do_sample: false
... quant_conf:
...   quantization_type: "bits-and-bytes"
...   load_in_4bit: true
...   bnb_4bit_compute_dtype: bfloat16
...   bnb_4bit_quant_type: "nf4"
... model_conf:
...   model: "/models/Poro-34B"
...   adapter: "/experiments/SDX-359/poro-v1-D/checkpoint-3000"
...   model_args:
...     attn_implementation: "flash_attention_2"
...     use_cache: False
...     device_map: {'transformer.word_embeddings': 0, 'lm_head': 0, 'transformer.word_embeddings_layernorm': 0,
...       'transformer.h.0': 0, 'transformer.h.1': 0, 'transformer.h.2': 0, 'transformer.h.3': 0, 'transformer.h.4': 0,
...       'transformer.h.5': 0, 'transformer.h.6': 0, 'transformer.h.7': 0, 'transformer.h.8': 0, 'transformer.h.9': 0,
...       'transformer.h.10': 0, 'transformer.h.11': 0, 'transformer.h.12': 0, 'transformer.h.13': 0, 'transformer.h.14': 0,
...       'transformer.h.15': 0, 'transformer.h.16': 0, 'transformer.h.17': 0, 'transformer.h.18': 0, 'transformer.h.19': 0,
...       'transformer.h.20': 0, 'transformer.h.21': 0, 'transformer.h.22': 1, 'transformer.h.23': 1, 'transformer.h.24': 1,
...       'transformer.h.25': 1, 'transformer.h.26': 1, 'transformer.h.27': 1, 'transformer.h.28': 1, 'transformer.h.29': 1,
...       'transformer.h.30': 1, 'transformer.h.31': 1, 'transformer.h.32': 1, 'transformer.h.33': 1, 'transformer.h.34': 1,
...       'transformer.h.35': 1, 'transformer.h.36': 1, 'transformer.h.37': 1, 'transformer.h.38': 1, 'transformer.h.39': 1,
...       'transformer.h.40': 1, 'transformer.h.41': 1, 'transformer.h.42': 1, 'transformer.h.43': 1, 'transformer.h.44': 1,
...       'transformer.h.45': 1, 'transformer.h.46': 1, 'transformer.h.47': 1, 'transformer.h.48': 1, 'transformer.h.49': 1,
...       'transformer.h.50': 1, 'transformer.h.51': 1, 'transformer.h.52': 1, 'transformer.h.53': 0, 'transformer.ln_f': 0}'''
>>> GenerationConfig.from_yaml(yamlstr).model_conf.tokenizer
'/experiments/SDX-359/poro-v1-D/checkpoint-3000'

"""

import datasets
import torch
import tqdm
import transformers
import yaml
from accelerate import PartialState

from aimfttk.config import GenerationConfig
from aimfttk.config.generation import PromptType
from aimfttk.data.data_types import EvalConversation
from aimfttk.data.dataset import make_dataset
from aimfttk.data.tokenizer import subsetup_tokenizer_for_inference
from aimfttk.model import get_model, subsetup_peft_for_inference
from aimfttk.utils.serialization import ConfigDumper


def _make_open_prompt_dataset(input_txt, num_samples):
    """Handle sampling without prompt (sampling full conversations)"""
    samples = []
    for i in range(num_samples):
        example = {}
        example["txt_input"] = input_txt
        example["response"] = ""
        example["dataset"] = "_open_prompt_data_"
        example["id"] = str(i)
        samples.append(example)
    return datasets.Dataset.from_list(samples)


def subsetup_prompt_data(prompt_conf, tokenizer, max_total_length, max_new_tokens):
    """Subsetup step that prepares a dataset"""
    if prompt_conf.type == PromptType.EVALDATA:
        dataset = make_dataset(prompt_conf.data, data_type=EvalConversation)

        def apply_generation_chat_template(example):
            txt_input = tokenizer.apply_chat_template(
                example["prompt_messages"], tokenize=False, add_generation_prompt=True
            )
            return {"txt_input": txt_input}

        dataset = dataset.map(apply_generation_chat_template)
    elif prompt_conf.type == PromptType.OPEN:
        dataset = _make_open_prompt_dataset(prompt_conf.input, prompt_conf.num_samples)
    else:
        raise NotImplementedError(f"Not implemented prompt type: {prompt_conf.type}.")

    def tokenize_and_truncate(
        example, tokenizer=tokenizer, max_total_length=max_total_length, max_new_steps=max_new_tokens
    ):
        unconstrained_prefill_length = tokenizer(
            example["txt_input"],
            return_tensors="pt",
            truncation=False,
        )[
            "input_ids"
        ].shape[1]
        inputs = tokenizer(
            example["txt_input"], return_tensors="pt", truncation=True, max_length=max_total_length - max_new_tokens
        )
        truncated = bool(unconstrained_prefill_length > inputs["input_ids"].shape[1])
        return {"truncated": truncated, **inputs}

    dataset = dataset.map(tokenize_and_truncate)
    dataset = dataset.with_format("torch", device=PartialState().device)
    return dataset


def subsetup_generation_defaults(hf_gen_params, tokenizer):
    defaults = {
        # Add </s> and eos_token_id. They may be the same, but we often hack eos_token_id to message end token so that
        # inference engines that use that information will stop generating at the right token.
        "eos_token_id": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("</s>")],
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "max_length": 2048,
        "max_new_tokens": 512,
    }
    for key, value in defaults.items():
        hf_gen_params.setdefault(key, value)
    return hf_gen_params


def setup_generation(generation_conf: GenerationConfig):
    """Prepare the data, tokenizer, model, and generation configuration"""
    tokenizer = subsetup_tokenizer_for_inference(generation_conf.model_conf.tokenizer)
    gen_params = subsetup_generation_defaults(generation_conf.hf_gen_params, tokenizer)
    hf_gen_conf = transformers.GenerationConfig(**gen_params)
    dataset = subsetup_prompt_data(
        generation_conf.prompt_conf,
        tokenizer,
        max_total_length=hf_gen_conf.max_length,
        max_new_tokens=hf_gen_conf.max_new_tokens,
    )
    model = get_model(
        model_name_or_path=generation_conf.model_conf.model,
        model_load_kwargs=generation_conf.model_conf.model_args.get_model_load_kwargs(),
        quantization_config=generation_conf.quant_conf.get_hf_config(),
    )
    model = subsetup_peft_for_inference(model, generation_conf.model_conf.adapter)
    model.eval()
    return dataset, tokenizer, model, hf_gen_conf


def run_generation(dataset, tokenizer, model, hf_gen_conf):
    """Loop over the data and generate"""
    eval_generations = []
    with torch.no_grad():
        for i, example in enumerate(tqdm.tqdm(dataset)):
            outputs = model.generate(
                input_ids=example["input_ids"], attention_mask=example["attention_mask"], generation_config=hf_gen_conf
            )
            generated_text = tokenizer.batch_decode(
                outputs[:, example["input_ids"].shape[1] :].detach().cpu().numpy(), skip_special_tokens=True
            )[0]
            all_text_raw = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=False)[0]
            example_output = dict(
                dataset=example.get("dataset", "unknown"),
                id=example.get("id", i),
                prompt=example["txt_input"],
                answer=generated_text,
                correct_answers=[example["response"]],
                test_type="open-ended",
                data_collection=example["dataset"],
                truncated=example["truncated"].cpu().item(),
                all_text_raw=all_text_raw,
            )
            eval_generations.append(example_output)
    return eval_generations


def add_generation_params_to_outputs(outputs, generation_config, tokenizer):
    """Add generation parameters to the outputs"""
    gen_params = subsetup_generation_defaults(generation_config.hf_gen_params, tokenizer)
    gen_params["quant_conf"] = generation_config.quant_conf.model_dump(mode="json", exclude_unset=True)
    gen_params["model_conf"] = generation_config.model_conf.model_dump(mode="json", exclude_unset=True)
    for output in outputs:
        output["gen_params"] = gen_params


def _write_generated_output_yaml(outputs, outpath):
    """Utility for writing YAML output file to disk"""
    with open(outpath, "w") as fo:
        yaml.dump(outputs, fo, sort_keys=False, Dumper=ConfigDumper)


def generate(generation_conf: GenerationConfig, outpath):
    """Run setup, generate, save result"""
    dataset, tokenizer, model, hf_gen_conf = setup_generation(generation_conf)
    outputs = run_generation(dataset, tokenizer, model, hf_gen_conf)
    add_generation_params_to_outputs(outputs, generation_conf, tokenizer)
    _write_generated_output_yaml(outputs, outpath)
