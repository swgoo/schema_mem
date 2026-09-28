"""Configuration classes for SchemaMem."""

from __future__ import annotations

from copy import copy
import math
from typing import Optional, Sequence

from transformers import Gemma4TextConfig


class SchemaMemConfig(Gemma4TextConfig):
    """Gemma decoder stack with in-place schema-memory replacements.

    layer_types is the execution plan: trace_attention replaces a global decoder without
    adding a layer or changing packed PLE indices. Global QKV geometry, FFN and
    PLE are retained; trace attention uses sliding RoPE and a bounded window.
    trace_size is an intermediate schema-read bottleneck, restored to
    hidden_size before attention. K/V use the original global head width.
    schema_size counts slots; dense reads use all slots, sparse writes use
    schema_top_k. trace_sliding_window is also the automatic commit period.
    schema_update_scale bounds each token's phase increment per coordinate;
    the default pi bounds the tanh-based increment to [-pi, pi].
    schema_write_gradient_power scales only write-branch backward by
    trace_sliding_window ** (-power); 0 disables, 0.5 uses sqrt, 1 uses length.
    """

    model_type = "schema_mem"

    def __init__(
        self,
        *,
        layer_types: Optional[Sequence[str]] = None,
        trace_size: int = 128,
        schema_size: int = 256,
        schema_top_k: int = 8,
        trace_sliding_window: int = 8_192,
        schema_write_zero_init_delta: bool = True,
        schema_update_scale: float = math.pi,
        schema_write_gradient_power: float = 0.0,
        **kwargs,
    ) -> None:
        requested_types = list(layer_types) if layer_types is not None else None
        if requested_types is not None:
            # Gemma's config initializer forces a final global layer and only
            # accepts native type names. Initialize defaults on a temporary
            # layout, then restore the caller's plan without inserting layers.
            if not requested_types:
                raise ValueError("layer_types must not be empty")
            kwargs["layer_types"] = ["sliding_attention"] + ["full_attention"] * max(1, len(requested_types) - 1)
            kwargs["num_hidden_layers"] = len(kwargs["layer_types"])
        super().__init__(**kwargs)
        self.layer_types = requested_types if requested_types is not None else [
            "trace_attention" if kind == "full_attention" else kind
            for kind in self.layer_types
        ]
        self.num_hidden_layers = len(self.layer_types)
        self.validate_layer_type()

        self.trace_size = trace_size
        self.schema_size = schema_size
        self.schema_top_k = schema_top_k
        self.trace_sliding_window = trace_sliding_window

        self.schema_write_zero_init_delta = schema_write_zero_init_delta
        self.schema_update_scale = schema_update_scale
        self.schema_write_gradient_power = schema_write_gradient_power

        self._validate_schema_mem_config()

    def validate_layer_type(self):
        """Validate this model's execution stack, without an original Gemma plan."""
        if not self.layer_types:
            raise ValueError("layer_types must not be empty")
        if self.num_hidden_layers != len(self.layer_types):
            raise ValueError("num_hidden_layers must match layer_types length")
        unknown = set(self.layer_types) - {
            "sliding_attention", "trace_attention",
        }
        if unknown:
            raise ValueError(f"Unsupported SchemaMem layer_types: {sorted(unknown)}")
        shared = self.num_kv_shared_layers
        if not isinstance(shared, int) or isinstance(shared, bool) or not 0 <= shared < self.num_hidden_layers:
            raise ValueError("num_kv_shared_layers must be a nonnegative suffix shorter than the stack")
        boundary = self.num_hidden_layers - shared
        for kind in set(self.layer_types[boundary:]):
            if kind not in self.layer_types[:boundary]:
                raise ValueError(f"Shared {kind} layers require an earlier non-sharing donor")

    def validate_rope(self):
        # Native validation assumes RoPE keys equal layer types. Here geometry
        # and the attention window intentionally differ, so validate each bank.
        view = copy(self)
        view.layer_types = list(view.rope_parameters or {})
        Gemma4TextConfig.validate_rope(view)

    def validate(self):
        # HF's strict Gemma dataclass captures base validator functions. Dispatch
        # by name to extend layout/RoPE validation while reusing native checks.
        for validator in Gemma4TextConfig.__class_validators__:
            getattr(self, validator.__name__)()
        self._validate_schema_mem_config()

    def _validate_schema_mem_config(self) -> None:
        if (
            isinstance(self.schema_write_gradient_power, bool)
            or not isinstance(self.schema_write_gradient_power, (int, float))
            or not math.isfinite(self.schema_write_gradient_power)
            or not 0 <= self.schema_write_gradient_power <= 1
        ):
            raise ValueError("schema_write_gradient_power must be between 0 and 1")
        if not isinstance(self.schema_write_zero_init_delta, bool):
            raise ValueError("schema_write_zero_init_delta must be a boolean")
        if (
            isinstance(self.schema_update_scale, bool)
            or not isinstance(self.schema_update_scale, (int, float))
            or not math.isfinite(self.schema_update_scale)
            or self.schema_update_scale <= 0
        ):
            raise ValueError("schema_update_scale must be finite and positive")
        integer_fields = {
            "hidden_size": self.hidden_size,
            "trace_size": self.trace_size,
            "schema_size": self.schema_size,
            "schema_top_k": self.schema_top_k,
            "trace_sliding_window": self.trace_sliding_window,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.schema_top_k > self.schema_size:
            raise ValueError("schema_top_k must be <= schema_size")
        if self.trace_size > self.hidden_size:
            raise ValueError(
                "trace_size must be <= Gemma hidden_size"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim % 2 or (self.global_head_dim or self.head_dim) % 2:
            raise ValueError("head_dim and global_head_dim must be even for RoPE")
        if not isinstance(self.attention_dropout, (int, float)) or not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")


__all__ = ["SchemaMemConfig"]
