"""Gemma 4 with in-place schema-memory decoder replacements.

Trace layers reuse global attention weights, FFN, norms and PLE. Dense reads
compress phasor(E + S) into per-token traces, restored to hidden_size before
Gemma attention with a sliding window/RoPE. An independent writer uses pre-read
hidden states and static E to produce phase updates; each window boundary
commits them and resets trace K/V.
Episode state is caller-owned. No token-wise full-S trajectory is materialized.
Checkpoint conversion and frozen-teacher execution live in utils.py.
"""

from __future__ import annotations

from collections import UserDict
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    DynamicLayer,
    DynamicSlidingWindowLayer,
)
from transformers.generation import GenerationConfig, GenerationMixin, GenerationMode
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4CausalLMOutputWithPast,
    Gemma4PreTrainedModel,
    Gemma4RMSNorm,
    Gemma4TextDecoderLayer,
    Gemma4TextModel,
    Gemma4TextModelOutputWithPast,
    Gemma4TextRotaryEmbedding,
    Gemma4TextScaledWordEmbedding,
)

try:  # Package-style import used by Transformers remote/custom model loading.
    from .configuration_schema_mem import SchemaMemConfig
except ImportError:  # Direct execution from this standalone repository.
    from configuration_schema_mem import SchemaMemConfig


def _wrap_phase_angles(angles: torch.Tensor) -> torch.Tensor:
    # Reduce before casting: large accumulated angles can otherwise lose the
    # low-order phase bits. Persistent/readable state remains FP32.
    return torch.remainder(angles.to(torch.float64), 2 * torch.pi).to(torch.float32)


def _phase_features(angles: torch.Tensor) -> torch.Tensor:
    """Continuous real representation of unit phasors, including at wrap."""
    angles = angles.to(torch.float32)
    return torch.cat((angles.cos(), angles.sin()), dim=-1)


class _ScaleWriteGradient(torch.autograd.Function):
    """Exact forward identity with an explicitly rescaled surrogate backward."""

    @staticmethod
    def forward(ctx, value, divisor):
        ctx.divisor = divisor
        return value

    @staticmethod
    def backward(ctx, gradient):
        return gradient / ctx.divisor, None


def _schema_attention_weights(
    scores: torch.Tensor,
    top_k: int,
    *,
    dtype: torch.dtype,
    dropout: float,
    training: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Select top-k probabilities without renormalizing their retained mass.

    Indices are discrete, but the full softmax denominator connects selected
    probabilities to all logits. Only selected values receive write gradients.
    """
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or not 1 <= top_k <= scores.shape[-1]
    ):
        raise ValueError("schema_top_k must be between 1 and the number of schema slots")
    indices = None
    probabilities = F.softmax(scores.float(), dim=-1)
    if top_k < scores.shape[-1]:
        _, indices = torch.topk(scores, top_k, dim=-1)
        probabilities = probabilities.gather(-1, indices)
    weights = probabilities.to(dtype)
    return F.dropout(weights, p=dropout, training=training), indices


def _mix_schema_values(
    weights: torch.Tensor,
    indices: Optional[torch.Tensor],
    values: torch.Tensor,
) -> torch.Tensor:
    """Gather [B, T, k, D], never a per-token copy of the whole schema bank."""
    if indices is None:
        return torch.matmul(weights, values)
    batch = torch.arange(values.shape[0], device=values.device)[:, None, None]
    selected = values[batch, indices]
    return (weights.unsqueeze(-1) * selected).sum(dim=-2)


class SchemaCache:
    """Layer-indexed S and pending delta S; no token axis or model parameters.

    Independent of Transformers' K/V cache, one instance owns every memory
    layer by execution ``layer_idx``. Updates accumulate FP64 phase increments;
    only commit changes readable state, with out-of-place tensor replacement.

    Like DynamicCache this is a mutable, caller-owned episode container. Saved
    tensor references stay valid, but outputs sharing this cache are not frozen
    snapshots. ``fork()`` copies containers without copying/detaching tensors.
    ``detach()`` is explicit; normal reads, updates and commits retain autograd.
    """

    def __init__(self, states: Mapping[int, torch.Tensor] | None = None):
        self.states: dict[int, torch.Tensor] = {}
        self.pending: dict[int, torch.Tensor] = {}
        self.seen_tokens: dict[int, int] = {}
        self.pending_lengths: dict[int, int] = {}
        for layer_idx, state in (states or {}).items():
            self.set_state(state, layer_idx)

    @staticmethod
    def _validate_index(layer_idx: int) -> None:
        if not isinstance(layer_idx, int) or isinstance(layer_idx, bool) or layer_idx < 0:
            raise ValueError("layer_idx must be a nonnegative integer")

    @staticmethod
    def _validate_tensor(value: torch.Tensor) -> None:
        if (not isinstance(value, torch.Tensor) or value.ndim != 3
                or not value.is_floating_point() or any(size <= 0 for size in value.shape)):
            raise ValueError("Schema cache tensors must have shape [batch, schemas, hidden_size]")

    def get_state(self, layer_idx: int) -> torch.Tensor | None:
        self._validate_index(layer_idx)
        return self.states.get(layer_idx)

    def get_pending(self, layer_idx: int) -> torch.Tensor | None:
        self._validate_index(layer_idx)
        return self.pending.get(layer_idx)

    def _validate_entry(self, value: torch.Tensor, layer_idx: int) -> None:
        self._validate_index(layer_idx)
        self._validate_tensor(value)
        for previous in (self.states.get(layer_idx), self.pending.get(layer_idx)):
            if previous is not None and previous.shape != value.shape:
                raise ValueError("Schema cache batch/schema shapes changed; start a fresh cache")

    def set_state(self, state: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Install an initial/restored state without an implicit detach or wrap."""
        self._validate_entry(state, layer_idx)
        self.states[layer_idx] = state.to(torch.float32)
        return self.states[layer_idx]

    def get_seq_length(self, layer_idx: int) -> int:
        self._validate_index(layer_idx)
        return self.seen_tokens.get(layer_idx, 0)

    def get_pending_length(self, layer_idx: int) -> int:
        self._validate_index(layer_idx)
        return self.pending_lengths.get(layer_idx, 0)

    def update(self, phase_increments: torch.Tensor, layer_idx: int, *, num_tokens: int = 0) -> torch.Tensor:
        """Accumulate delta S; the readable S remains unchanged until commit."""
        self._validate_entry(phase_increments, layer_idx)
        if not isinstance(num_tokens, int) or isinstance(num_tokens, bool) or num_tokens < 0:
            raise ValueError("num_tokens must be a nonnegative integer")
        delta = phase_increments.to(torch.float64)
        previous = self.pending.get(layer_idx)
        self.pending[layer_idx] = delta if previous is None else previous.to(delta.device) + delta
        self.seen_tokens[layer_idx] = self.get_seq_length(layer_idx) + num_tokens
        self.pending_lengths[layer_idx] = self.get_pending_length(layer_idx) + num_tokens
        return self.pending[layer_idx]

    def commit(self, layer_idx: int | None = None) -> tuple[int, ...]:
        """Apply sum-then-wrap, clear committed increments, return changed layers."""
        if layer_idx is not None:
            self._validate_index(layer_idx)
        indices = tuple(self.pending) if layer_idx is None else ((layer_idx,) if layer_idx in self.pending else ())
        next_states = {}
        for idx in indices:
            delta = self.pending[idx]
            self._validate_entry(delta, idx)
            previous = self.states.get(idx)
            total = delta if previous is None else previous.to(
                device=delta.device, dtype=delta.dtype
            ) + delta
            next_states[idx] = _wrap_phase_angles(total)
        # Validate/compute all targets before changing the container.
        self.states.update(next_states)
        for idx in indices:
            del self.pending[idx]
            self.pending_lengths.pop(idx, None)
        return indices

    def fork(self) -> "SchemaCache":
        """Independent containers with shared immutable tensor history."""
        cache = type(self)()
        cache.states = dict(self.states)
        cache.pending = dict(self.pending)
        cache.seen_tokens = dict(self.seen_tokens)
        cache.pending_lengths = dict(self.pending_lengths)
        return cache

    def reset(self, layer_idx: int | None = None) -> None:
        if layer_idx is None:
            self.states.clear()
            self.pending.clear()
            self.seen_tokens.clear()
            self.pending_lengths.clear()
        else:
            self._validate_index(layer_idx)
            self.states.pop(layer_idx, None)
            self.pending.pop(layer_idx, None)
            self.seen_tokens.pop(layer_idx, None)
            self.pending_lengths.pop(layer_idx, None)

    def detach(self, layer_idx: int | None = None) -> "SchemaCache":
        """Explicit truncated-BPTT boundary; never detach during commit."""
        if layer_idx is not None:
            self._validate_index(layer_idx)
        for values in (self.states, self.pending):
            for idx in tuple(values):
                if layer_idx is None or idx == layer_idx:
                    values[idx] = values[idx].detach()
        return self

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        for values in (self.states, self.pending):
            for idx, value in values.items():
                values[idx] = value.index_select(0, indices.to(value.device))

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        self.batch_select_indices(beam_idx)

    def batch_repeat_interleave(self, repeats: int) -> None:
        for values in (self.states, self.pending):
            for idx, value in values.items():
                values[idx] = value.repeat_interleave(repeats, dim=0)


@dataclass
class SchemaMemModelOutputWithPast(Gemma4TextModelOutputWithPast):
    """Native Gemma outputs plus the caller-owned, episode-specific schema cache."""

    schema_cache: Optional[SchemaCache] = None


@dataclass
class SchemaMemCausalLMOutputWithPast(Gemma4CausalLMOutputWithPast):
    """Language-model outputs carrying the state for the next chunk."""

    schema_cache: Optional[SchemaCache] = None


class SchemaTraceAttentionLayer(Gemma4TextDecoderLayer):
    """Gemma global decoder with schema compression and chunk commits.

    Read: input norm -> dense attention over phasor(E + S) -> trace_size.
    Attend: restore hidden_size -> Gemma global-shaped attention using SWA
    RoPE/window -> original post-attention norm/residual.
    Write: pre-read hidden states + static E -> addresses + phase increments.
    The raw update features also have a learned local residual projection. The original
    Gemma FFN/MoE, PLE and layer scalar finish the decoder in their native order.

    Writes never read this layer's S, including indirectly through its traces.
    S is fixed within a chunk and committed by summing phase increments.
    The caller-owned SchemaCache stores S and pending updates per layer; trace
    K/V is bounded by the same window and evicted at commit. trace_size controls
    an information bottleneck, not the restored attention's K/V width.
    """

    def __init__(
        self,
        config: SchemaMemConfig,
        *,
        layer_idx: int = 0,
    ) -> None:
        config._validate_schema_mem_config()
        decoder_config = copy(config)
        decoder_config.layer_types = [
            "full_attention" if kind == "trace_attention" else kind
            for kind in config.layer_types
        ]
        decoder_config.layer_types[layer_idx] = "full_attention"
        super().__init__(decoder_config, layer_idx)
        # Retain native cross-layer K/V sharing, including absent K/V weights
        # in the shared suffix. Only the window/RoPE policy changes.
        decoder_config._attn_implementation = config._attn_implementation or "sdpa"
        self.self_attn.sliding_window = config.trace_sliding_window
        self.config = config

        self.hidden_size = config.hidden_size
        self.trace_size = config.trace_size
        self.schema_size = config.schema_size
        # Global weight geometry is independent of window/RoPE policy.
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = (
            config.num_global_key_value_heads
            if config.attention_k_eq_v else config.num_key_value_heads
        )
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )
        self.head_dim = self.self_attn.head_dim
        self.schema_read_query_proj = nn.Linear(
            config.hidden_size,
            self.hidden_size,
            bias=config.attention_bias,
        )

        # One learned bank: schema = schema_embedding + cached schema_state.
        # Reads use E + S; writes use E alone, with the same learned projections.
        self.schema_embedding = nn.Parameter(
            torch.empty(config.schema_size, self.hidden_size)
        )
        self.schema_key_proj = nn.Linear(
            2 * self.hidden_size, self.hidden_size, bias=False
        )
        self.schema_value_proj = nn.Linear(
            2 * self.hidden_size, self.trace_size, bias=False
        )
        self.trace_up_proj = nn.Linear(self.trace_size, self.hidden_size, bias=False)

        # Independent pre-read input -> schema projections select write addresses and
        # produce phase increments at the configured schema width.
        self.schema_write_query_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=config.attention_bias
        )
        self.schema_write_delta_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=config.attention_bias
        )
        # Raw update features and hidden residuals have equal width, not equal
        # semantics. Learn which update components to expose to the backbone.
        # No bias keeps a zero increment neutral. Use ordinary initialization:
        # zeroing this AND the delta projection would block local write learning.
        self.schema_update_output_proj = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False
        )

        self.trace_input_norm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # Gemma delegates ordinary Linear/norm initialization to this HF base
        # implementation. Use it for standalone layer construction as well.
        self.apply(lambda module: PreTrainedModel._init_weights(self, module))
        self._init_schema_parameters()

    @torch.no_grad()
    def _init_schema_parameters(self) -> None:
        # HF's init helpers protect weights already loaded from a checkpoint.
        # These are angles, not ordinary embedding coordinates. Near-zero
        # angles make cos(E) almost constant and collapse dense reads toward
        # the same trace regardless of the input. Cover the full phase circle.
        init.uniform_(self.schema_embedding, a=-math.pi, b=math.pi)
        if self.config.schema_write_zero_init_delta:
            init.zeros_(self.schema_write_delta_proj.weight)
            if self.schema_write_delta_proj.bias is not None:
                init.zeros_(self.schema_write_delta_proj.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: Optional[torch.Tensor] = None,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        schema_cache: Optional[SchemaCache] = None,
        attention_mask: Optional[tuple[Optional[torch.Tensor], ...]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        shared_kv_states: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Process consolidation intervals locally; Gemma layers stay parallel.

        The schedule counts token positions, not forward calls or valid-padding
        counts. Splitting a prefill into smaller calls keeps the same commits.
        No full [time, schemas, hidden_size] state trajectory is materialized.
        The model supplies one native causal mask per interval; the 2D
        padding_mask is used only to exclude padding from schema writes.
        The donor publishes one K/V pair per commit interval for this forward;
        shared suffix layers reuse those pairs without maintaining their own K/V.
        RoPE cos/sin are computed once by the model at absolute positions and
        sliced at commit boundaries even though trace K/V offsets reset.
        """
        if schema_cache is None:
            raise ValueError("Consolidation requires a caller-owned schema_cache")
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")
        self._validate_inputs(hidden_states, per_layer_input)
        if padding_mask is not None and padding_mask.ndim != 2:
            raise ValueError("padding_mask must be a 2D padding mask")
        self.config._validate_schema_mem_config()
        interval = self.config.trace_sliding_window
        pending_length = schema_cache.get_pending_length(self.layer_idx)
        if pending_length >= interval:
            raise ValueError("Consolidation interval changed mid-chunk; start a fresh cache")
        if attention_mask is None:
            raise ValueError("trace_attention masks must be prepared by SchemaMemModel")
        length = hidden_states.shape[1]
        expected_masks = (pending_length + length + interval - 1) // interval
        if len(attention_mask) != expected_masks:
            raise ValueError("trace_attention must provide one mask per commit interval")
        # Only K/V-producing layers need a cache. Retain donor interval tensors
        # until shared suffix layers finish, even if commit resets the donor cache.
        shared_kv_states = UserDict() if shared_kv_states is None else shared_kv_states
        is_shared = self.self_attn.is_kv_shared_layer
        self.self_attn.sliding_window = interval
        if is_shared:
            shared_intervals = shared_kv_states.get("trace_attention")
            if shared_intervals is None or len(shared_intervals) != expected_masks:
                raise ValueError("Shared trace attention requires donor K/V for every commit interval")
        elif self.self_attn.store_full_length_kv:
            shared_intervals = []
            shared_kv_states["trace_attention"] = shared_intervals
        trace_cache = None if is_shared else self._prepare_trace_cache(past_key_values)
        if trace_cache is not None and trace_cache.get_seq_length(self.layer_idx) != pending_length:
            raise ValueError("Trace KV cache and schema_cache must continue the same episode/chunk")
        outputs = []
        # Reuse static E within this forward, never across optimizer steps.
        # No detach: E and the writer still receive future-loss gradients.
        write_schema = self._project_schema(
            None, batch_size=hidden_states.shape[0],
            dtype=hidden_states.dtype, device=hidden_states.device,
        )
        start = 0
        while start < length:
            remaining = interval - schema_cache.get_pending_length(self.layer_idx)
            end = min(length, start + remaining)
            interval_kv = UserDict()
            if is_shared:
                interval_kv["full_attention"] = shared_intervals[len(outputs)]
            output, schema_updates = self._forward_chunk(
                hidden_states[:, start:end],
                per_layer_input[:, start:end] if per_layer_input is not None else None,
                schema_state=schema_cache.get_state(self.layer_idx),
                write_schema=write_schema,
                attention_mask=attention_mask[len(outputs)],
                padding_mask=(
                    padding_mask[:, -length:][:, start:end].to(torch.bool)
                    if padding_mask is not None else None
                ),
                position_embeddings=tuple(part[:, start:end] for part in position_embeddings),
                past_key_values=trace_cache,
                use_cache=not is_shared,
                shared_kv_states=interval_kv,
            )
            if self.self_attn.store_full_length_kv:
                shared_intervals.append(interval_kv["full_attention"])
            outputs.append(output)
            schema_cache.update(schema_updates, self.layer_idx, num_tokens=end - start)
            if schema_cache.get_pending_length(self.layer_idx) == interval:
                schema_cache.commit(self.layer_idx)
                if trace_cache is not None:
                    trace_cache.layers[self.layer_idx] = DynamicSlidingWindowLayer(
                        sliding_window=interval
                    )
            start = end
        return torch.cat(outputs, dim=1)

    def _forward_chunk(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: Optional[torch.Tensor] = None,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        schema_state: Optional[torch.Tensor] = None,
        write_schema: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        shared_kv_states: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stateful read/attention plus an independent additive writer.

        S stays fixed throughout this primitive. Standalone calls may supply
        a shared or batched S and an optional DynamicCache; only ``forward``
        schedules commits. RoPE cos/sin are supplied by the model, not inferred
        from the interval-local trace cache.
        attention_mask is already prepared by the model for the native backend;
        padding_mask separately excludes padded tokens from schema writes.
        """
        self._validate_inputs(hidden_states, per_layer_input)
        if past_key_values is not None and not use_cache:
            raise ValueError("past_key_values requires use_cache=True")

        query_length = hidden_states.shape[1]
        past_length = (
            past_key_values.get_seq_length(self.layer_idx)
            if past_key_values is not None
            else 0
        )
        # This primitive never crosses a commit boundary, even in inference.
        # Public forward splits longer inputs and resets K/V at each commit.
        if past_length + query_length > self.config.trace_sliding_window:
            raise ValueError("_forward_chunk cannot cross a commit boundary")
        if use_cache:
            past_key_values = self._prepare_trace_cache(past_key_values)

        residual = hidden_states
        traces, _, _ = self._encode_traces(
            hidden_states,
            schema_state,
        )
        normalized_traces = self.trace_input_norm(self.trace_up_proj(traces))
        contextualized_traces, _ = self.self_attn(
            hidden_states=normalized_traces,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=UserDict() if shared_kv_states is None else shared_kv_states,
            past_key_values=past_key_values,
        )
        if (
            past_key_values is not None
            and not self.self_attn.is_kv_shared_layer
            and self.config.trace_sliding_window == 1
        ):
            # Native [-window + 1:] retains everything at window=1.
            entry = past_key_values.layers[self.layer_idx]
            entry.keys = entry.keys[..., :0, :]
            entry.values = entry.values[..., :0, :]

        # Fully masked query rows have backend-dependent attention outputs.
        # Exclude them before both the decoder residual and schema writes.
        if padding_mask is not None:
            contextualized_traces = contextualized_traces.masked_fill(
                ~padding_mask.to(torch.bool).unsqueeze(-1), 0
            )
        # H contains lower-layer context but has not read this layer's S.
        # Feeding state-conditioned traces or E + S to the writer would create
        # a recurrent feedback Jacobian, despite the additive wrapped commit.
        if write_schema is None:
            write_schema = self._project_schema(
                None, batch_size=hidden_states.shape[0],
                dtype=hidden_states.dtype, device=hidden_states.device,
            )
        token_update_features, token_phase_updates, schema_weights, schema_indices = self._extract_schema_deltas(
            self.input_layernorm(hidden_states),
            schema_keys=write_schema[0],
            schema_values=write_schema[1],
            attention_mask=padding_mask,
        )
        # Local features retain their expressive scale; phase increments have
        # a separate bounded step size before accumulation into future memory.
        hidden_states = residual + self.post_attention_layernorm(contextualized_traces)
        hidden_states = hidden_states + self.schema_update_output_proj(token_update_features)
        output = self._finish_decoder(hidden_states, per_layer_input)
        schema_updates = self._aggregate_schema_updates(
            token_phase_updates, schema_weights, schema_indices
        )
        return output, schema_updates

    def _finish_decoder(self, hidden_states, per_layer_input):
        """Native Gemma FFN, optional MoE, PLE and layer scaling."""
        residual = hidden_states
        hidden_states = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        if self.enable_moe_block:
            dense = self.post_feedforward_layernorm_1(hidden_states)
            flat = residual.reshape(-1, residual.shape[-1])
            _, weights, indices = self.router(flat)
            experts = self.experts(self.pre_feedforward_layernorm_2(flat), indices, weights)
            hidden_states = dense + self.post_feedforward_layernorm_2(experts.reshape(residual.shape))
        hidden_states = residual + self.post_feedforward_layernorm(hidden_states)
        if self.hidden_size_per_layer_input and per_layer_input is not None:
            ple = self.act_fn(self.per_layer_input_gate(hidden_states)) * per_layer_input
            hidden_states = hidden_states + self.post_per_layer_input_norm(self.per_layer_projection(ple))
        return hidden_states * self.layer_scalar

    def _project_schema(
        self,
        schema_state: Optional[torch.Tensor],
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build fixed-chunk schema = E + S and project its phasor keys/values."""
        schema = self.schema_embedding.to(device=device, dtype=torch.float32).unsqueeze(0)
        schema = schema.expand(batch_size, -1, -1)
        expected_shared = (self.schema_size, self.hidden_size)
        expected_batched = (batch_size, self.schema_size, self.hidden_size)
        if schema_state is not None:
            if tuple(schema_state.shape) == expected_shared:
                schema_state = schema_state.unsqueeze(0).expand(batch_size, -1, -1)
            elif tuple(schema_state.shape) != expected_batched:
                raise ValueError(
                    "schema_state must have shape "
                    f"{expected_shared} or {expected_batched}, got "
                    f"{tuple(schema_state.shape)}"
                )
            schema = schema + schema_state.to(device=device, dtype=torch.float32)
        # Never feed raw wrapped angles into a Euclidean read path: the
        # numerical 0/2*pi boundary must not create a semantic discontinuity.
        schema_features = _phase_features(schema).to(self.schema_value_proj.weight.dtype)
        return (
            self.schema_key_proj(schema_features).to(dtype),
            self.schema_value_proj(schema_features).to(dtype),
        )

    def _encode_traces(
        self,
        hidden_states: torch.Tensor,
        schema_state: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Densely mix all schema values into traces, without top-k selection."""
        batch_size = hidden_states.shape[0]
        normalized = self.input_layernorm(hidden_states)
        schema_read_query = self.schema_read_query_proj(normalized)
        schema_keys, schema_values = self._project_schema(
            schema_state,
            batch_size=batch_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        schema_weights, schema_indices = self._address_schemas(
            schema_read_query, schema_keys,
            top_k=self.schema_size, dtype=hidden_states.dtype,
        )

        traces = _mix_schema_values(
            schema_weights, schema_indices, schema_values
        )
        return traces, schema_keys, schema_values

    def _extract_schema_deltas(
        self,
        write_inputs: torch.Tensor,
        *,
        schema_keys: torch.Tensor,
        schema_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Extract writes from normalized pre-read H and static E keys/values.

        This is pointwise on lower-layer context, not on this layer's attention.
        Keys and values use the read projections but exclude the live state S.
        Return local update features, bounded FP32 phase increments, weights and
        indices. Both branches retain gradients, but large local features must
        not become arbitrarily large rotations when summed over a chunk.
        """
        if write_inputs.ndim != 3 or write_inputs.shape[-1] != self.hidden_size:
            raise ValueError("write_inputs must have shape [batch, sequence, hidden_size]")
        batch, length, _ = write_inputs.shape
        if attention_mask is None:
            trace_valid_mask = torch.ones(batch, length, dtype=torch.bool, device=write_inputs.device)
        else:
            if tuple(attention_mask.shape) != (batch, length):
                raise ValueError("write attention_mask must have shape [batch, sequence]")
            trace_valid_mask = attention_mask.to(device=write_inputs.device, dtype=torch.bool)
        # Mask BEFORE projections as well as after softmax: all-padding queries
        # must not add phase increments or propagate padded activations.
        write_inputs = write_inputs.masked_fill(~trace_valid_mask.unsqueeze(-1), 0)
        # Projection weights are shared with reads, but this bank is phasor(E).
        if schema_values.ndim == 2:
            schema_values = schema_values.unsqueeze(0)
        if (
            schema_values.ndim != 3
            or schema_values.shape[0] not in (1, batch)
            or tuple(schema_values.shape[1:]) != (self.schema_size, self.trace_size)
        ):
            raise ValueError(
                "schema_values must have shape [schemas, trace_size] "
                "or [batch, schemas, trace_size]"
            )
        schema_values = schema_values.to(device=write_inputs.device, dtype=write_inputs.dtype)
        schema_values = schema_values.expand(batch, -1, -1)
        schema_keys = schema_keys.to(device=write_inputs.device, dtype=write_inputs.dtype)

        schema_write_query = self.schema_write_query_proj(write_inputs)
        schema_weights, schema_indices = self._address_schemas(
            schema_write_query, schema_keys,
            top_k=self.config.schema_top_k, dtype=write_inputs.dtype,
        )
        schema_weights = schema_weights.masked_fill(~trace_valid_mask.unsqueeze(-1), 0)
        schema_write_context = _mix_schema_values(schema_weights, schema_indices, schema_values)
        token_update_features = self.schema_write_delta_proj(
            write_inputs + self.trace_up_proj(schema_write_context)
        )
        token_update_features = token_update_features.masked_fill(~trace_valid_mask.unsqueeze(-1), 0)
        # Bound each write by schema_update_scale * tanh(raw), before weighting
        # and accumulation. Wrapping S alone is not a per-token write bound.
        token_phase_updates = self.config.schema_update_scale * torch.tanh(token_update_features.float())
        return token_update_features, token_phase_updates, schema_weights, schema_indices

    def _aggregate_schema_updates(
        self,
        token_phase_updates: torch.Tensor,
        schema_weights: torch.Tensor,
        schema_indices: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate token deltas; dense GEMM is FP32, pending updates FP64.

        Sums compose across forward fragments. No token-wise S history is kept,
        and this aggregation does not change the local residual's token deltas.
        """
        batch = token_phase_updates.shape[0]
        # Attention weights distribute a token's increment across schemas.
        # Do not divide by token count or schema mass. FP32 atomic reduction
        # Use FP32 for dense GEMM (including backward), then promote only its
        # slot-sized result for pending accumulation and phase wrapping. This
        # avoids FP64 token-by-slot GEMMs while retaining precise commit sums.
        # Sparse scatter keeps FP64 to limit order-dependent collision error.
        with torch.autocast(device_type=token_phase_updates.device.type, enabled=False):
            if schema_indices is None:
                schema_updates = torch.matmul(
                    schema_weights.float().transpose(1, 2), token_phase_updates.float()
                ).double()
            else:
                phases = token_phase_updates.double()
                weights = schema_weights.double()
                selected_schema_updates = weights.unsqueeze(-1) * phases.unsqueeze(-2)
                # Colliding token addresses add, rather than overwrite. This is
                # [B, T*k, D] -> [B, M, D], with no [B, T, M, D] S trajectory.
                targets = schema_indices.reshape(batch, -1, 1).expand(
                    -1, -1, self.hidden_size
                )
                schema_updates = torch.zeros(
                    batch, self.schema_size, self.hidden_size,
                    device=token_phase_updates.device, dtype=torch.float64,
                )
                schema_updates = schema_updates.scatter_add(
                    1, targets, selected_schema_updates.reshape(batch, -1, self.hidden_size)
                )
        power = self.config.schema_write_gradient_power
        if power and schema_updates.requires_grad:
            # Scale only newly written increments, never the S -> S identity
            # path or the local residual. Fixed period keeps fragmented forward
            # calls consistent. This is NOT forward averaging or true AD.
            schema_updates = _ScaleWriteGradient.apply(
                schema_updates, self.config.trace_sliding_window ** power,
            )
        return schema_updates

    def _address_schemas(
        self,
        schema_query: torch.Tensor,
        schema_keys: torch.Tensor,
        *,
        top_k: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Shared score math; reads use the full bank, writes use schema_top_k."""
        with torch.autocast(device_type=schema_query.device.type, enabled=False):
            scores = torch.matmul(schema_query.float(), schema_keys.float().transpose(-1, -2))
            scores = scores / math.sqrt(self.hidden_size)
        return _schema_attention_weights(
            scores,
            top_k,
            dtype=dtype,
            dropout=self.config.attention_dropout,
            training=self.training,
        )

    def _validate_inputs(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: Optional[torch.Tensor],
    ) -> None:
        """Reject malformed inputs before mutating either episode cache."""
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, seq, hidden]")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, got {hidden_states.shape[-1]}"
            )
        if hidden_states.shape[1] == 0:
            raise ValueError("hidden_states must contain at least one token")
        if self.hidden_size_per_layer_input and per_layer_input is not None:
            expected = (*hidden_states.shape[:2], self.hidden_size_per_layer_input)
            if tuple(per_layer_input.shape) != expected:
                raise ValueError(f"per_layer_input must have shape {expected}")

    def _prepare_trace_cache(self, cache: Optional[Cache]) -> DynamicCache:
        """Use the standard bounded cache, also for standalone layer calls."""
        if cache is None:
            cache = DynamicCache()
        if not isinstance(cache, DynamicCache):
            raise TypeError("Trace attention requires a standard DynamicCache")
        while len(cache.layers) <= self.layer_idx:
            cache.layers.append(DynamicLayer())
        current = cache.layers[self.layer_idx]
        window = self.config.trace_sliding_window
        self.self_attn.sliding_window = window
        if not isinstance(current, DynamicSlidingWindowLayer) or current.sliding_window != window:
            if current.get_seq_length() != 0:
                raise ValueError(
                    "The trace cache has a different window policy; change "
                    "trace_sliding_window before starting a fresh cache"
                )
            cache.layers[self.layer_idx] = DynamicSlidingWindowLayer(sliding_window=window)
        return cache


class SchemaMemPreTrainedModel(Gemma4PreTrainedModel):
    """Shared Hugging Face model metadata and initialization."""

    config_class = SchemaMemConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Gemma4TextDecoderLayer",
        "SchemaCache",
        "SchemaTraceAttentionLayer",
    ]
    _skip_keys_device_placement = ["past_key_values", "shared_kv_states"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_attention_backend = True

    def save_pretrained(self, *args, **kwargs):
        # A standalone SchemaMem checkpoint must not revert the source Gemma
        # multimodal key mapping (model.* -> model.language_model.*).
        kwargs.setdefault("save_original_format", False)
        return super().save_pretrained(*args, **kwargs)

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        super()._init_weights(module)
        if isinstance(module, SchemaTraceAttentionLayer):
            module._init_schema_parameters()


class SchemaMemModel(SchemaMemPreTrainedModel):
    """Gemma 4 text backbone with explicit trace layers."""

    def __init__(self, config: SchemaMemConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # Native decoder constructors use geometry types for projection sizes
        # and K/V sharing. Ordinary SWA layers only share with SWA donors;
        # trace layers construct their own native global-geometry config view.
        decoder_config = copy(config)
        decoder_config.layer_types = list(config.layer_types)

        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size**0.5,
        )

        self.layers = nn.ModuleList()
        for layer_idx, layer_type in enumerate(config.layer_types):
            if layer_type == "trace_attention":
                layer = SchemaTraceAttentionLayer(config, layer_idx=layer_idx)
            else:
                layer = Gemma4TextDecoderLayer(decoder_config, layer_idx)
            self.layers.append(layer)

        self.norm = Gemma4RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

        rotary_config = copy(decoder_config)
        rotary_config.layer_types = [
            "sliding_attention" if kind == "trace_attention" else kind
            for kind in decoder_config.layer_types
        ]
        rotary_config.rope_parameters = {
            kind: copy(config.rope_parameters[kind]) for kind in set(rotary_config.layer_types)
        }
        self.rotary_emb = Gemma4TextRotaryEmbedding(rotary_config)
        self.unique_layer_types = set(rotary_config.layer_types)
        # SWA RoPE at the global Q/K width retained by trace attention.
        trace_rope_config = copy(config)
        trace_rope_config.head_dim = config.global_head_dim or config.head_dim
        trace_rope_config.layer_types = ["sliding_attention"]
        trace_rope_config.rope_parameters = {
            "sliding_attention": copy(config.rope_parameters["sliding_attention"])
        }
        self.trace_rotary_emb = Gemma4TextRotaryEmbedding(trace_rope_config)

        # In-place replacements preserve every original packed PLE slice.
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                self.padding_idx,
                embed_scale=config.hidden_size_per_layer_input**0.5,
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_model_projection_scale = config.hidden_size**-0.5
            self.per_layer_projection_norm = Gemma4RMSNorm(
                config.hidden_size_per_layer_input,
                eps=config.rms_norm_eps,
            )
        self.gradient_checkpointing = False
        self.post_init()

    def iter_trace_layers(self):
        """Yield trace modules without registering a duplicate module tree."""
        return (layer for layer in self.layers if isinstance(layer, SchemaTraceAttentionLayer))

    # With PLE aligned to the full stack, the native Gemma helpers work as-is.
    get_per_layer_inputs = Gemma4TextModel.get_per_layer_inputs
    project_per_layer_inputs = Gemma4TextModel.project_per_layer_inputs

    def _prepare_cache(
        self,
        past_key_values: Optional[Cache],
        *,
        use_cache: bool,
    ) -> Optional[Cache]:
        if not use_cache:
            if past_key_values is not None:
                raise ValueError("past_key_values requires use_cache=True")
            return None
        if past_key_values is None:
            return self._new_dynamic_cache()
        if isinstance(past_key_values, Cache):
            # GenerationMixin knows the stack depth, but trace and backbone
            # layers have different window sizes. Install the right policies.
            if not any(layer.is_initialized for layer in past_key_values.layers):
                return self._new_dynamic_cache()
            for idx, kind in enumerate(self.config.layer_types):
                if idx < len(past_key_values.layers):
                    bounded = self.config.layer_types[idx] in {"sliding_attention", "trace_attention"}
                    if past_key_values.is_sliding[idx] != bounded:
                        raise ValueError("Cache window policy does not match SchemaMem layer_types")
            return past_key_values
        raise TypeError("past_key_values must be a transformers Cache")

    def _new_dynamic_cache(self) -> DynamicCache:
        """Build a standard cache with the runtime receptive-field policy.

        One entry per execution-layer index. All KV-sharing layers leave their
        entries unused; only donors/independent layers populate K/V tensors.
        Cache policies follow layer_types; trace layers use their own longer window.
        """

        cache = DynamicCache()
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, SchemaTraceAttentionLayer):
                entry = DynamicSlidingWindowLayer(
                    sliding_window=self.config.trace_sliding_window
                )
            else:
                kind = self.config.layer_types[idx]
                if kind == "sliding_attention":
                    entry = DynamicSlidingWindowLayer(sliding_window=self.config.sliding_window)
                else:
                    entry = DynamicLayer()
            cache.layers.append(entry)
        return cache

    def _validate_schema_cache(self, schema_cache: SchemaCache, batch_size: Optional[int] = None) -> None:
        if not isinstance(schema_cache, SchemaCache):
            raise TypeError("schema_cache must be a SchemaCache")
        for values in (schema_cache.states, schema_cache.pending):
            for idx, value in values.items():
                SchemaCache._validate_index(idx)
                if idx >= len(self.layers) or self.config.layer_types[idx] != "trace_attention":
                    raise ValueError(f"Unknown trace layer in schema_cache: {idx}")
                SchemaCache._validate_tensor(value)
                layer = self.layers[idx]
                if value.shape[1:] != (layer.schema_size, layer.hidden_size):
                    raise ValueError("schema_cache slot/feature dimensions do not match the model")
                if batch_size is not None and value.shape[0] != batch_size:
                    raise ValueError("schema_cache batch dimension does not match")

    @staticmethod
    def _position_ids(
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.LongTensor],
        *,
        past_length: int = 0,
    ) -> torch.LongTensor:
        """Prepare absolute positions once for every decoder and trace layer."""
        batch_size, sequence_length = hidden_states.shape[:2]
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + sequence_length,
                device=hidden_states.device,
            ).unsqueeze(0)
        elif position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.ndim != 2 or position_ids.shape[-1] != sequence_length:
            raise ValueError("position_ids must have one position per input token")
        if position_ids.shape[0] not in (1, batch_size):
            raise ValueError("position_ids batch dimension does not match")
        return position_ids.expand(batch_size, -1)

    @staticmethod
    def _prepare_trace_attention_mask(
        config: SchemaMemConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache],
        position_ids: Optional[torch.Tensor],
        *,
        layer_idx: int,
    ) -> tuple[Optional[torch.Tensor], ...]:
        """Prepare native causal masks for the upcoming commit intervals.

        Only the first interval can read retained trace K/V. Later intervals
        start with an empty cache, independently of absolute RoPE positions.
        Masks use only input shape/dtype; schema encoding is not required.
        """
        batch, length = inputs_embeds.shape[:2]
        past = past_key_values.get_seq_length(layer_idx) if past_key_values is not None else 0
        window = config.trace_sliding_window
        if past >= window:
            raise ValueError("Trace window changed mid-chunk; start a fresh cache")
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape[0] != batch:
                raise ValueError("attention_mask must be a 2D padding mask matching the batch")
            if attention_mask.shape[-1] == length and past:
                attention_mask = torch.cat((attention_mask.new_ones(batch, past), attention_mask), dim=-1)
            if attention_mask.shape[-1] < past + length:
                raise ValueError("2D attention_mask must cover all cached and current keys")
            attention_mask = attention_mask[:, -(past + length):]

        # A standalone config may not have selected an attention backend yet.
        # Match the native trace attention's default without mutating config.
        mask_config = copy(config)
        mask_config._attn_implementation = config._attn_implementation or "sdpa"
        empty_cache = DynamicCache()
        masks = []
        start, retained = 0, past
        while start < length:
            end = min(length, start + window - retained)
            masks.append(create_causal_mask(
                config=mask_config,
                inputs_embeds=inputs_embeds[:, start:end],
                attention_mask=(
                    attention_mask[:, past + start - retained:past + end]
                    if attention_mask is not None else None
                ),
                past_key_values=(past_key_values if start == 0 and past_key_values is not None else empty_cache),
                position_ids=position_ids[:, start:end] if position_ids is not None else None,
                layer_idx=layer_idx,
            ))
            start, retained = end, 0
        return tuple(masks)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        per_layer_inputs: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        schema_cache: Optional[SchemaCache] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> SchemaMemModelOutputWithPast:
        obsolete = {"schema_states", "pending_consolidation", "commit_consolidation",
                    "consolidation_enabled"} & kwargs.keys()
        if obsolete:
            raise TypeError(f"Use schema_cache and config.trace_sliding_window instead of {sorted(obsolete)}")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError("attention_mask must be a 2D padding mask")
        if input_ids is not None and per_layer_inputs is not None:
            raise ValueError(
                "per_layer_inputs cannot be supplied together with input_ids"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(
                    input_ids,
                    inputs_embeds,
                )
            expected = (*inputs_embeds.shape[:2], self.config.num_hidden_layers,
                        self.hidden_size_per_layer_input)
            if tuple(per_layer_inputs.shape) != expected:
                raise ValueError(f"per_layer_inputs must cover all layers with shape {expected}")
            per_layer_inputs = self.project_per_layer_inputs(
                inputs_embeds,
                per_layer_inputs,
            )

        self.config._validate_schema_mem_config()
        if schema_cache is None:
            schema_cache = SchemaCache()
        self._validate_schema_cache(schema_cache, inputs_embeds.shape[0])
        use_cache = self.config.use_cache if use_cache is None else use_cache
        output_hidden_states = (
            self.config.output_hidden_states
            if output_hidden_states is None
            else output_hidden_states
        )
        cache = self._prepare_cache(
            past_key_values,
            use_cache=use_cache,
        )
        # All trace layers share one schedule. Check the episode pairing once
        # before the backbone mutates K/V; each trace layer prepares its own cache.
        trace_layer = next(self.iter_trace_layers(), None)
        if trace_layer is not None:
            trace_length = cache.get_seq_length(trace_layer.layer_idx) if cache is not None else 0
            if trace_length != schema_cache.get_pending_length(trace_layer.layer_idx):
                raise ValueError("Trace KV cache and schema_cache must continue the same episode/chunk")
        # A trace-only stack resets its first KV entry at every commit; S's
        # token counter, unlike that KV length, retains absolute positions.
        past_length = (
            schema_cache.get_seq_length(0)
            if self.config.layer_types[0] == "trace_attention"
            else (cache.get_seq_length() if cache is not None else 0)
        )
        position_ids = self._position_ids(
            inputs_embeds,
            position_ids,
            past_length=past_length,
        )

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "position_ids": position_ids,
        }
        # All trace layers process the same tokens and commit together, so
        # their cache lengths and interval masks are identical.
        sliding_layer_idx = next((
            idx for idx, kind in enumerate(self.config.layer_types)
            if kind == "sliding_attention"
        ), None)
        causal_masks = {
            "sliding_attention": (
                create_sliding_window_causal_mask(**mask_kwargs, layer_idx=sliding_layer_idx)
                if sliding_layer_idx is not None else None
            ),
            "trace_attention": (
                self._prepare_trace_attention_mask(**mask_kwargs, layer_idx=trace_layer.layer_idx)
                if trace_layer is not None else None
            ),
        }

        hidden_states = inputs_embeds
        position_embeddings = {
            layer_type: self.rotary_emb(hidden_states, position_ids, layer_type)
            for layer_type in self.unique_layer_types
        }
        if trace_layer is not None:
            position_embeddings["trace_attention"] = self.trace_rotary_emb(hidden_states, position_ids, "sliding_attention")

        # Gemma 4 E2B shares K/V projections across its final layers. The
        # UserDict matches Transformers' FSDP-safe container and is populated
        # by the last non-sharing layer of each source attention type.
        shared_kv_states = UserDict()

        all_hidden_states: Optional[list[torch.Tensor]] = (
            [hidden_states] if output_hidden_states else None
        )
        for layer_idx, layer in enumerate(self.layers):
            per_layer_input = (
                per_layer_inputs[:, :, layer_idx, :]
                if per_layer_inputs is not None else None
            )
            layer_type = self.config.layer_types[layer_idx]
            hidden_states = layer(
                hidden_states,
                per_layer_input=per_layer_input,
                shared_kv_states=shared_kv_states,
                schema_cache=schema_cache,
                attention_mask=causal_masks[layer_type],
                padding_mask=attention_mask,
                position_embeddings=position_embeddings[layer_type],
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=use_cache,
                **kwargs,
            )
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        return SchemaMemModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=cache,
            shared_kv_states=shared_kv_states,
            hidden_states=(
                tuple(all_hidden_states) if all_hidden_states is not None else None
            ),
            schema_cache=schema_cache,
        )


class SchemaMemForCausalLM(SchemaMemPreTrainedModel, GenerationMixin):
    """Causal LM with native greedy/sampling generation and dynamic caches.

    For continuation across generate() calls, pass a caller-owned SchemaCache
    and retain it alongside the returned past_key_values. Generation outputs
    otherwise keep the standard Hugging Face format.
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: SchemaMemConfig) -> None:
        super().__init__(config)
        self.model = SchemaMemModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self._reset_new_output_projections()

    @torch.no_grad()
    def _reset_new_output_projections(self) -> None:
        """Initialize only the new schema-delta projection; retain Gemma readout."""

        if self.config.schema_write_zero_init_delta:
            for layer in self.model.iter_trace_layers():
                layer.schema_write_delta_proj.weight.zero_()
                if layer.schema_write_delta_proj.bias is not None:
                    layer.schema_write_delta_proj.bias.zero_()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.model = decoder

    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: dict,
        generation_mode: GenerationMode,
        batch_size: int,
        max_cache_length: int,
    ) -> None:
        # Reuse the native generation loop; only cache construction is special.
        # Beam/speculative decoding would also need schema reorder/rollback.
        if generation_mode not in (GenerationMode.GREEDY_SEARCH, GenerationMode.SAMPLE):
            raise NotImplementedError("SchemaMem generate supports greedy decoding and sampling only")
        if generation_config.cache_implementation not in (None, "dynamic"):
            raise ValueError("SchemaMem generate supports only dynamic K/V caches")
        past = model_kwargs.get("past_key_values")
        schema = model_kwargs.get("schema_cache")
        if not generation_config.use_cache:
            if past is not None or schema is not None:
                raise ValueError("Continuing supplied caches requires use_cache=True")
            # Native generation recomputes the full prefix, with fresh S each time.
            return
        if past is not None:
            if generation_config.cache_implementation is not None:
                raise ValueError("Pass either past_key_values or cache_implementation, not both")
            if not isinstance(past, DynamicCache):
                raise TypeError("SchemaMem generate requires a DynamicCache")
        if schema is not None:
            self.model._validate_schema_cache(schema, batch_size)
        if generation_config.num_return_sequences > 1 and (past is not None or schema is not None):
            raise NotImplementedError("Supplied caches require num_return_sequences=1")
        if past is None:
            model_kwargs["past_key_values"] = self.model._new_dynamic_cache()
        if schema is None:
            # This same mutable container survives every native generation step,
            # including chunked prefill, without storing episode data on the model.
            model_kwargs["schema_cache"] = SchemaCache()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        per_layer_inputs: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: int | torch.Tensor = 0,
        schema_cache: Optional[SchemaCache] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> SchemaMemCausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            use_cache=use_cache,
            schema_cache=schema_cache,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            softcap = self.config.final_logit_softcapping
            logits = torch.tanh(logits / softcap) * softcap

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits,
                labels,
                self.vocab_size,
                **kwargs,
            )

        return SchemaMemCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            shared_kv_states=outputs.shared_kv_states,
            schema_cache=outputs.schema_cache,
        )


__all__ = [
    "SchemaCache",
    "SchemaTraceAttentionLayer",
    "SchemaMemForCausalLM",
    "SchemaMemModel",
    "SchemaMemModelOutputWithPast",
    "SchemaMemCausalLMOutputWithPast",
]
