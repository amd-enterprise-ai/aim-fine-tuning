# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Model setup"""

import peft
import torch
import transformers
from peft import PeftModel, prepare_model_for_kbit_training
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

import aimfttk
from aimfttk.config.hf_integration import GenericPeftConfig, NoPeftConfig, PretrainedPeftConfig
from aimfttk.utils.distributed import is_fsdp
from aimfttk.utils.model import is_kbit, is_quantized

logger = transformers.utils.logging.get_logger(__file__)


def get_model(model_name_or_path, model_load_kwargs, quantization_config=None):
    """Creates an instance of the desired model"""
    if quantization_config is not None:
        model_load_kwargs["quantization_config"] = quantization_config
    model = None
    try:
        model_class_name = model_load_kwargs.pop("model_class", "AutoModelForCausalLM")
        model_class = getattr(transformers, model_class_name)
        model = model_class.from_pretrained(model_name_or_path, **model_load_kwargs)
    except ImportError as e:
        # Try to intercept those misleading error messages from cases where the Flash Attention import fails
        # because of CUDA not being available,
        if "Flash Attention" in e.message and "is not available" in e.message:
            if not torch.cuda.is_available():
                raise ValueError(
                    "Running on CPU - is this intentional? Flash attention is not available on CPU."
                ) from e
        else:
            raise e
    return model


def subsetup_peft_for_inference(model, adapter_name_or_path):
    """Setup PEFT for inference"""
    if adapter_name_or_path is None:
        return model
    else:
        return PeftModel.from_pretrained(model, adapter_name_or_path, is_trainable=False)


def model_to_peft_model(model, peft_config, peft_kwargs={}):
    """Makes a PEFT Adapter Model from an existing model, based on peft config"""
    if isinstance(model, peft.PeftModel):
        raise ValueError("Model is already a PEFT model!")
    model = peft.get_peft_model(model, peft_config, **peft_kwargs)
    return model


def _init_new_rows_by_avg_sample(weights, num_new_rows, sigma_scale=1e-5):
    """In-place initialize new rows by sampling from multi-variate Gaussian modeled on old rows

    Useful for initialising new token embeddings.

    Code inspired by these:
    - https://nlp.stanford.edu/~johnhew/vocab-expansion.html
    - https://github.com/spyysalo/instruction-finetune/blob/main/train.py
    """
    n = weights.size()[0] - num_new_rows
    weights_dtype = weights.dtype
    init_weights = weights.to(torch.float32)
    mu = torch.mean(init_weights[:n, :], dim=0)
    sigma = ((init_weights[:n, :] - mu).T @ (init_weights[:n, :] - mu)) / n
    covariance = sigma_scale * sigma
    try:
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=covariance)
    except RuntimeError as e:
        if "Calling torch.linalg.cholesky on a CPU tensor requires compiling PyTorch with LAPACK." in str(e):
            raise ValueError("Embedding extension not available with the current PyTorch installation.") from e
        # covariance needs to be positive definite, add some noise to diagonal
        covariance = (
            covariance + torch.eye(covariance.size(0), device=covariance.device, dtype=covariance.dtype) * sigma_scale
        )
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=covariance)
    new_rows = torch.stack(tuple((dist.sample() for _ in range(num_new_rows))), dim=0).to(weights_dtype)
    if not is_deepspeed_zero3_enabled():
        weights[n:, :] = new_rows
    else:
        raise NotImplementedError("Deepspeed compatible new-embedding-row init is not yet implemented (commented out)")
        # Likely this code should work:
        # Add it when enabling deepspeed
        # import deepspeed
        #
        # with deepspeed.zero.GatheredParameters(weights, modifier_rank=0):
        #    if torch.distributed.get_rank() == 0:
        #        weights.data[n:, :] = new_rows


def grow_embeddings(model, new_num_tokens, approach="hf_default"):
    """Grows the embeddings to account for new tokens

    Approach can be
        "sample_avg_emb":
            Model the average embedding as a gaussian and samples the new embeddings from this distribution, see:
            https://nlp.stanford.edu/~johnhew/vocab-expansion.html
        "hf_default":
            Don't modify what HuggingFace does. This is essentially the same as sample_avg_emb, but perhaps
            implemented a bit different and maintained by HuggingFace
        "hf_legacy":
            This is likely a random init.

    Returns the names of the modules that should additionally be updated and saved during training even if using PEFT.
    """

    original_vocab_size = model.config.vocab_size
    num_new_tokens = new_num_tokens - original_vocab_size
    if num_new_tokens <= 0:
        logger.info(
            "New tokens added to tokenizer vocabulary, but the model already has space for the new tokens. No need to grow embeddings."
        )
        return []
    elif is_quantized(model):
        raise ValueError("Quantized layers can be grown, but they cannot be trained or saved, so this will not work!")
    try:
        model.resize_token_embeddings(new_num_tokens=new_num_tokens, mean_resizing=(approach != "hf_legacy"))
    except RuntimeError as e:
        if "Calling torch.linalg.cholesky on a CPU tensor requires compiling PyTorch with LAPACK." in str(e):
            raise ValueError("Embedding extension not available with the current PyTorch installation.") from e
        else:
            raise e
    if approach == "sample_avg_emb":
        with torch.no_grad():  # The operations may otherwise leave some gradient compute graph fluff behind.
            input_embeddings = model.get_input_embeddings()
            _init_new_rows_by_avg_sample(input_embeddings.weight.data, num_new_tokens)
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None:
                _init_new_rows_by_avg_sample(output_embeddings.weight.data, num_new_tokens)
    elif approach in ("hf_default", "hf_legacy"):
        pass
    else:
        raise ValueError(f"Unknown embedding init method {approach}")

    peft_extra_modules_to_save = ["embed_tokens"]
    if model.get_output_embeddings() is not None:
        peft_extra_modules_to_save.append("lm_head")
    return peft_extra_modules_to_save


def subsetup_handle_peft(
    peft_conf: NoPeftConfig | PretrainedPeftConfig | GenericPeftConfig,
    model,
    peft_extra_modules_to_save: list,
    training_args: transformers.TrainingArguments,
):
    """Setup step: PEFT

    Note: This should always be called in setup, and this will handle NO_PEFT too. In that case this will just return
    the same model as nothing else needs to be done.
    """
    if peft_conf.peft_type == aimfttk.config.hf_integration.NO_PEFT:
        return model

    # Make sure the base model is frozen (copied from peft.prepare_model_for_kbit_training):
    for name, param in model.named_parameters():
        param.requires_grad = False
    if is_kbit(model) and (not is_fsdp() and not is_deepspeed_zero3_enabled()):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=training_args.gradient_checkpointing_kwargs,
        )
    elif (
        is_quantized(model)
        and training_args.gradient_checkpointing
        and training_args.gradient_checkpointing_kwargs.get("use_reentrant", True)
    ):
        # With reentrant checkpointing, this seems to be needed, otherwise we get
        #  "UserWarning: None of the inputs have requires_grad=True. Gradients will be None"
        # This same hack is used in peft.prepare_model_for_kbit_training
        model.enable_input_require_grads()
    if peft_conf.peft_type == aimfttk.config.hf_integration.PRETRAINED_PEFT:
        if peft_extra_modules_to_save:
            raise NotImplementedError("Cannot handle saving extra modules (like Embeddings) with pretrained PEFT yet.")
        model = peft.PeftModel.from_pretrained(model, peft_conf.name_or_path, is_trainable=True)
    else:  # A new, initialized adapter
        peft_config = peft_conf.get_peft_config()
        if peft_extra_modules_to_save:
            modules_to_save = peft_config.modules_to_save
            if modules_to_save is None:
                modules_to_save = peft_extra_modules_to_save
            else:
                modules_to_save.extend(peft_extra_modules_to_save)
            peft_config.modules_to_save = modules_to_save
        model = model_to_peft_model(model, peft_config)
    # Get the parameter logging from here:
    # https://github.com/huggingface/peft/blob/6008f272a565f56c146c5d9fd78d00cb24392d7b/src/peft/peft_model.py#L492-L532
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(
        "PEFT model trainable params: "
        f"{trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )
    return model
