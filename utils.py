"""Gemma checkpoint conversion and training-only oracle/bootstrap execution."""

from __future__ import annotations

from copy import copy, deepcopy
from collections import UserDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional
import re

import torch
from torch import nn
from transformers import AutoConfig, Gemma4Config, Gemma4TextConfig
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer, DynamicSlidingWindowLayer
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4CausalLMOutputWithPast, Gemma4TextModelOutputWithPast,
    Gemma4TextDecoderLayer, Gemma4TextRotaryEmbedding,
)

from configuration_schema_mem import SchemaMemConfig
from modeling_schema_mem import SchemaCache, SchemaMemForCausalLM, SchemaMemModel


def _text_config(config):
    if isinstance(config, Gemma4Config):
        return config.text_config
    if isinstance(config, Gemma4TextConfig) and not isinstance(config, SchemaMemConfig):
        return config
    raise TypeError("Expected a Gemma4Config or Gemma4TextConfig")


def _gemma_view(config):
    view = copy(config)
    view.layer_types = [
        "full_attention" if kind == "trace_attention" else kind
        for kind in config.layer_types
    ]
    return view


def from_gemma_config(gemma_config, **schema_mem_kwargs):
    """Replace all globals in place; layer indices and packed PLE stay intact."""
    source = _text_config(gemma_config)
    values = deepcopy(source).to_dict()
    values.pop("model_type", None)
    values["layer_types"] = [
        "trace_attention" if kind == "full_attention" else kind
        for kind in source.layer_types
    ]
    values.update(schema_mem_kwargs)
    return SchemaMemConfig(**values)


def _validate_source(source, target):
    if source.num_hidden_layers != target.num_hidden_layers:
        raise ValueError("Gemma and SchemaMem configs differ at num_hidden_layers")
    if source.layer_types != _gemma_view(target).layer_types:
        raise ValueError("Gemma and SchemaMem configs differ at layer_types")
    for field in (
        "hidden_size", "intermediate_size", "num_attention_heads",
        "num_key_value_heads", "num_global_key_value_heads", "head_dim",
        "global_head_dim", "attention_k_eq_v", "use_double_wide_mlp",
        "num_kv_shared_layers", "hidden_size_per_layer_input", "vocab_size",
        "vocab_size_per_layer_input", "attention_bias", "enable_moe_block",
        "num_experts", "moe_intermediate_size", "top_k_experts",
    ):
        if getattr(source, field, None) != getattr(target, field, None):
            raise ValueError(f"Gemma and SchemaMem configs differ at {field}")


def convert_gemma_checkpoint(gemma_config, schema_mem_config, checkpoint, *, output_dir=None, **load_kwargs):
    """Load all original decoder weights at the same indices, preserving KV sharing.

    Only schema projections/bank and trace-input normalization are new.
    Original QKV geometry, FFN, norms, PLE and layer scalars are retained.
    Trace attention uses sliding RoPE and a commit-bounded window at runtime.
    """
    source = _text_config(gemma_config)
    _validate_source(source, schema_mem_config)
    config = deepcopy(schema_mem_config)
    if isinstance(checkpoint, Mapping):
        if load_kwargs:
            raise ValueError("Loader options apply only to checkpoint paths")
        model = SchemaMemForCausalLM(config)
        weights = {re.sub(r"^model\.language_model\.", "model.", k): v for k, v in checkpoint.items()}
        # Validate against the native decoder's expected keys, not all trace keys.
        for i in range(source.num_hidden_layers):
            with torch.device("meta"):
                native = Gemma4TextDecoderLayer(source, i)
            missing = [f"model.layers.{i}.{key}" for key in native.state_dict()
                       if f"model.layers.{i}.{key}" not in weights]
            if missing:
                raise ValueError(f"Checkpoint is missing backbone weights: {missing}")
        result = model.load_state_dict(weights, strict=False)
        allowed = tuple(f"model.layers.{i}." for i, k in enumerate(config.layer_types) if k == "trace_attention")
        missing = [k for k in result.missing_keys if not k.startswith(allowed)
                   and not (config.tie_word_embeddings and k == "lm_head.weight")]
        if missing or result.unexpected_keys:
            raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={result.unexpected_keys}")
    else:
        if "key_mapping" in load_kwargs:
            raise ValueError("key_mapping is managed by convert_gemma_checkpoint")
        mapping = {r"^model\.language_model\.": "model."} if isinstance(gemma_config, Gemma4Config) else {}
        model = SchemaMemForCausalLM.from_pretrained(
            str(checkpoint), config=config, key_mapping=mapping, **load_kwargs,
        )
    if config.tie_word_embeddings:
        model.tie_weights()
    if output_dir is not None:
        model.save_pretrained(output_dir, save_original_format=False)
    return model


def from_gemma_pretrained(pretrained_model_name_or_path, *, schema_mem_config=None, gemma_config_kwargs=None, **load_kwargs):
    source = AutoConfig.from_pretrained(pretrained_model_name_or_path, **(gemma_config_kwargs or {}))
    config = from_gemma_config(source, **(schema_mem_config or {}))
    return convert_gemma_checkpoint(source, config, pretrained_model_name_or_path, **load_kwargs)


@torch.no_grad()
def make_gemma_oracle_layers(model):
    """Snapshot ONLY replaced decoders before training; unchanged layers are shared.

    Keep this ModuleDict in the training wrapper, not the inference model.
    Reuse the same snapshot across bootstrap and assembled training.
    """
    core = model.model if isinstance(model, SchemaMemForCausalLM) else model
    config = _gemma_view(core.config)
    snapshots = nn.ModuleDict()
    for i, kind in enumerate(core.config.layer_types):
        if kind != "trace_attention":
            continue
        source = core.layers[i]
        with torch.device("meta"):
            decoder = Gemma4TextDecoderLayer(config, i)
        weights = source.state_dict()
        # Clone to prevent optimizer updates in the student changing its teacher.
        decoder.load_state_dict(
            {k: weights[k].detach().clone() for k in decoder.state_dict()}, assign=True,
        )
        snapshots[str(i)] = decoder.requires_grad_(False).eval()
    return snapshots


@torch.no_grad()
def run_gemma_oracle(
    model, input_ids=None, *, attention_mask=None, position_ids=None,
    past_key_values=None, inputs_embeds=None, per_layer_inputs=None,
    use_cache=False, output_hidden_states=True, logits_to_keep=0, oracle_layers=None,
):
    """Run native Gemma decoders, without schema compression or commits.

    Supply the pre-training snapshot for a stable teacher. Without it, this
    diagnostic snapshots the current decoder weights for this call only.
    """
    core = model.model if isinstance(model, SchemaMemForCausalLM) else model
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds")
    if input_ids is not None and per_layer_inputs is not None:
        raise ValueError("per_layer_inputs cannot be supplied together with input_ids")
    if past_key_values is not None and not use_cache:
        raise ValueError("past_key_values requires use_cache=True")
    oracle_layers = make_gemma_oracle_layers(core) if oracle_layers is None else oracle_layers
    layers = [oracle_layers[str(i)] if str(i) in oracle_layers else layer
              for i, layer in enumerate(core.layers)]
    config = _gemma_view(core.config)
    flags = [(module, module.training) for layer in layers for module in layer.modules()]
    windows = [(layer.self_attn, layer.self_attn.sliding_window) for layer in layers]
    try:
        for layer in layers:
            layer.eval()
            layer.self_attn.sliding_window = config.sliding_window if layer.self_attn.is_sliding else None
        cache = past_key_values
        if use_cache:
            if cache is None:
                cache = DynamicCache()
                cache.layers = [
                    DynamicSlidingWindowLayer(config.sliding_window) if k == "sliding_attention"
                    else DynamicLayer() for k in config.layer_types
                ]
            if not isinstance(cache, Cache) or len(cache.layers) != len(layers):
                raise ValueError("Oracle cache must match the execution stack")
            if list(cache.is_sliding) != [k == "sliding_attention" for k in config.layer_types]:
                raise ValueError("Oracle and SchemaMem caches must be kept separate")
        if inputs_embeds is None:
            inputs_embeds = core.embed_tokens(input_ids)
        if core.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = core.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = core.project_per_layer_inputs(inputs_embeds, per_layer_inputs)
        position_ids = core._position_ids(
            inputs_embeds, position_ids, past_length=cache.get_seq_length() if cache is not None else 0,
        )
        mask_args = dict(config=config, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                         past_key_values=cache, position_ids=position_ids)
        masks = {"full_attention": create_causal_mask(**mask_args),
                 "sliding_attention": create_sliding_window_causal_mask(**mask_args)}
        rotary = Gemma4TextRotaryEmbedding(config).to(inputs_embeds.device)
        positions = {k: rotary(inputs_embeds, position_ids, k) for k in set(config.layer_types)}
        hidden = inputs_embeds
        history = [hidden] if output_hidden_states else None
        shared = UserDict()
        for i, layer in enumerate(layers):
            kind = config.layer_types[i]
            hidden = layer(
                hidden, per_layer_input=per_layer_inputs[:, :, i] if per_layer_inputs is not None else None,
                shared_kv_states=shared, attention_mask=masks[kind], position_embeddings=positions[kind],
                position_ids=position_ids, past_key_values=cache,
            )
            if history is not None:
                history.append(hidden)
        hidden = core.norm(hidden)
        if history is not None:
            history.append(hidden)
        fields = dict(past_key_values=cache, shared_kv_states=shared,
                      hidden_states=tuple(history) if history is not None else None)
        if isinstance(model, SchemaMemForCausalLM):
            indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = model.lm_head(hidden[:, indices])
            if config.final_logit_softcapping is not None:
                cap = config.final_logit_softcapping
                logits = torch.tanh(logits / cap) * cap
            return Gemma4CausalLMOutputWithPast(logits=logits, **fields)
        return Gemma4TextModelOutputWithPast(last_hidden_state=hidden, **fields)
    finally:
        for module, training in flags:
            module.training = training
        for attention, window in windows:
            attention.sliding_window = window


@dataclass
class SchemaMemReplacementOutput:
    hidden_states: torch.Tensor


def run_replacement_layer(
    model, layer_idx, hidden_states, *, attention_mask=None, position_ids=None,
    per_layer_inputs=None, shared_kv_states=None, schema_state=None,
):
    """Bootstrap one replacement on the original global decoder's input."""
    core = model.model
    if not 0 <= layer_idx < len(core.layers) or core.config.layer_types[layer_idx] != "trace_attention":
        raise ValueError(f"layer {layer_idx} must be a trace layer for bootstrap")
    position_ids = core._position_ids(hidden_states, position_ids)
    if schema_state is not None and schema_state.ndim == 2:
        schema_state = schema_state.unsqueeze(0).expand(hidden_states.shape[0], -1, -1)
    cache = SchemaCache({layer_idx: schema_state} if schema_state is not None else None)
    # Bootstrap uses the teacher's donor K/V, just as it uses teacher inputs.
    # Assembled execution instead supplies the student's interval-wise trace K/V.
    shared = None
    if core.layers[layer_idx].self_attn.is_kv_shared_layer:
        if shared_kv_states is None or "full_attention" not in shared_kv_states:
            raise ValueError("Shared replacement bootstrap requires oracle donor K/V")
        if hidden_states.shape[1] > core.config.trace_sliding_window:
            raise ValueError("Shared replacement bootstrap must fit within one trace window")
        shared = {"trace_attention": (shared_kv_states["full_attention"],)}
    output = core.layers[layer_idx](
        hidden_states,
        per_layer_input=per_layer_inputs[:, :, layer_idx] if per_layer_inputs is not None else None,
        schema_cache=cache, padding_mask=attention_mask,
        shared_kv_states=shared,
        attention_mask=core._prepare_trace_attention_mask(
            core.config, hidden_states, attention_mask, None, position_ids, layer_idx=layer_idx,
        ),
        position_embeddings=core.trace_rotary_emb(hidden_states, position_ids, "sliding_attention"),
        past_key_values=None, use_cache=False,
    )
    return SchemaMemReplacementOutput(hidden_states=output)
