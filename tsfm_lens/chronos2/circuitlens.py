from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.hooks import RemovableHandle

from tsfm_lens.chronos2.pipeline import Chronos2PipelineCustom
from tsfm_lens.circuitlens import BaseCircuitLens


@dataclass
class CircuitLensChronos2(Chronos2PipelineCustom, BaseCircuitLens):
    """
    CircuitLens utilities for Chronos-2 encoder-only models.

    Supports head ablation on time and group self-attention, MLP ablation,
    residual stream reading, and lightweight attribution captures.
    """

    # Ablation tracking
    head_ablation_positions: dict[int, dict[int, str]] = field(default_factory=dict)
    head_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    time_head_ablation_positions: dict[int, dict[int, str]] = field(default_factory=dict)
    time_head_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    group_head_ablation_positions: dict[int, dict[int, str]] = field(default_factory=dict)
    group_head_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    mlp_ablation_positions: list[int] = field(default_factory=list)
    mlp_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # Attribution
    head_attribution_positions: dict[int, list[int]] = field(default_factory=dict)
    head_attribution_outputs: dict[int, dict[int, list[torch.Tensor]]] = field(default_factory=dict)
    head_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    mlp_attribution_positions: list[int] = field(default_factory=list)
    mlp_attribution_inputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    mlp_attribution_outputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    mlp_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # Residual stream
    read_stream_inputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    read_stream_outputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    read_stream_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Chronos2PipelineCustom.__post_init__(self)
        cfg = self.model.config  # type: ignore[attr-defined]
        self.num_layers = cfg.num_layers  # type: ignore[attr-defined]
        self.num_heads = cfg.num_heads  # type: ignore[attr-defined]
        self.is_decoder_only = True

    def set_to_eval_mode(self) -> None:
        self.model.eval()

    # ---------- Hook fns ----------
    def _ablate_head_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        _output: Any,
        head_indices: list[int],
        ablation_method: Literal["zero", "mean"],
    ) -> Any:
        if not isinstance(input, tuple) or len(input) != 1:
            raise ValueError("Expected single-tensor input to attention out projection.")
        x = input[0]
        if x.dim() != 3:
            raise ValueError(f"Unexpected projection input shape {tuple(x.shape)}")

        b, s, d = x.shape
        if d % self.num_heads != 0:
            raise ValueError(f"d_model {d} not divisible by num_heads {self.num_heads}")
        head_dim = d // self.num_heads
        reshaped = x.view(b, s, self.num_heads, head_dim)
        for h in head_indices:
            if h >= self.num_heads:
                raise ValueError(f"Head index {h} out of range")
            if ablation_method == "zero":
                reshaped[:, :, h, :] = 0.0
            elif ablation_method == "mean":
                reshaped[:, :, h, :] = reshaped[:, :, h, :].mean(dim=-1, keepdim=True)
            else:
                raise ValueError(f"Invalid ablation method {ablation_method}")
        modified = reshaped.view(b, s, -1)
        return module.forward(modified)

    def _ablate_mlp_hook_fn(
        self,
        module: torch.nn.Module,
        _in: Any,
        out: Any,
        ablation_method: Literal["zero", "mean"],
    ) -> Any:
        y = out[0] if isinstance(out, tuple) else out
        if ablation_method == "zero":
            return torch.zeros_like(y)
        return y.mean(dim=-1, keepdim=True)

    def _attribute_head_hook_fn(
        self,
        module: torch.nn.Module,
        inputs: Any,
        _outputs: Any,
        head_indices: list[int],
        layer_idx: int,
    ) -> None:
        if not isinstance(inputs, tuple) or len(inputs) != 1:
            raise ValueError("Expected single-tensor input to attention out projection.")
        x = inputs[0]
        if x.dim() != 3:
            raise ValueError(f"Unexpected projection input shape {tuple(x.shape)}")
        b, s, d = x.shape
        head_dim = d // self.num_heads
        weight_t = module.weight.T.to(x.dtype)
        layer_store = self.head_attribution_outputs.setdefault(layer_idx, {})
        for h in head_indices:
            hs = slice(h * head_dim, (h + 1) * head_dim)
            x_h = x[:, :, hs]
            w_h = weight_t[hs, :]
            head_out = torch.einsum("bsd,df->bsf", x_h, w_h)
            layer_store.setdefault(h, []).append(head_out)

    def _attribute_mlp_hook_fn(self, module: torch.nn.Module, inputs: Any, outputs: Any, layer_idx: int) -> None:
        x_in = inputs[0] if isinstance(inputs, tuple) else inputs
        x_out = outputs[0] if isinstance(outputs, tuple) else outputs
        self.mlp_attribution_inputs.setdefault(layer_idx, []).append(x_in)
        self.mlp_attribution_outputs.setdefault(layer_idx, []).append(x_out)

    def _read_stream_hook_fn(self, module: torch.nn.Module, inputs: Any, outputs: Any, layer_idx: int) -> None:
        x_in = inputs[0] if isinstance(inputs, tuple) else inputs
        if hasattr(outputs, "hidden_states"):
            x_out = outputs.hidden_states  # Chronos2EncoderBlockOutput
        else:
            x_out = outputs[0] if isinstance(outputs, tuple) else outputs
        self.read_stream_inputs.setdefault(layer_idx, []).append(x_in)
        self.read_stream_outputs.setdefault(layer_idx, []).append(x_out)

    # ---------- Public API ----------
    def add_ablation_hooks_explicit(
        self,
        ablations_types: list[Literal["head", "mlp"]],
        layers_to_ablate_mlp: list[int],
        heads_to_ablate: list[tuple[int, int]],
        attention_type: Literal["time", "group"] = "time",
    ) -> None:
        self.remove_all_hooks()
        self.reset_attribution_inputs_and_outputs()
        if "head" in ablations_types:
            self.add_head_ablation_hooks(heads_to_ablate, attention_type=attention_type)
        if "mlp" in ablations_types:
            self.add_mlp_ablation_hooks(layers_to_ablate_mlp)

    def add_ablation_hooks(
        self,
        ablations_types: list[Literal["head", "mlp"]],
        layers_to_ablate: list[int],
        ablate_n_heads_per_layer: int | None,
        rng: np.random.Generator | None = None,
        verbose: bool = False,
        attention_type: Literal["time", "group"] = "time",
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        if verbose:
            print(f"Ablations: {ablations_types} on layers {layers_to_ablate}")
        self.remove_all_hooks()
        self.reset_attribution_inputs_and_outputs()
        if "head" in ablations_types:
            if ablate_n_heads_per_layer is None:
                heads_to_ablate = [(layer, h) for layer in layers_to_ablate for h in range(self.num_heads)]
            else:
                heads_to_ablate = []
                for layer in layers_to_ablate:
                    heads = rng.choice(list(range(self.num_heads)), size=ablate_n_heads_per_layer, replace=False)
                    heads_to_ablate.extend([(layer, int(h)) for h in heads])
            self.add_head_ablation_hooks(heads_to_ablate, attention_type=attention_type)
        if "mlp" in ablations_types:
            self.add_mlp_ablation_hooks(layers_to_ablate)

    def add_head_ablation_hooks(
        self,
        heads_to_ablate: list[tuple[int, int]],
        ablation_method: Literal["zero", "mean"] = "zero",
        attention_type: Literal["time", "group"] = "time",
    ) -> dict[int, RemovableHandle]:
        heads_by_layer: dict[int, list[int]] = {}
        for layer_idx, head_idx in heads_to_ablate:
            heads_by_layer.setdefault(layer_idx, []).append(head_idx)
            if attention_type == "time":
                self.time_head_ablation_positions.setdefault(layer_idx, {})[head_idx] = ablation_method
            else:
                self.group_head_ablation_positions.setdefault(layer_idx, {})[head_idx] = ablation_method
            self.head_ablation_positions.setdefault(layer_idx, {})[head_idx] = ablation_method

        handles = self.time_head_ablation_handles if attention_type == "time" else self.group_head_ablation_handles
        for layer_idx, head_indices in heads_by_layer.items():
            block = self.model.encoder.block[layer_idx]  # type: ignore[attr-defined]
            target = block.layer[0].self_attention.o if attention_type == "time" else block.layer[1].self_attention.o
            hook = target.register_forward_hook(
                partial(self._ablate_head_hook_fn, head_indices=head_indices, ablation_method=ablation_method)
            )
            handles[layer_idx] = hook
            self.head_ablation_handles[layer_idx] = hook
        return handles

    def add_mlp_ablation_hooks(
        self, layers_to_ablate: list[int], ablation_method: Literal["zero", "mean"] = "zero"
    ) -> dict[int, RemovableHandle]:
        handles: dict[int, RemovableHandle] = {}
        for layer_idx in layers_to_ablate:
            block = self.model.encoder.block[layer_idx]  # type: ignore[attr-defined]
            target = block.layer[2].mlp
            hook = target.register_forward_hook(partial(self._ablate_mlp_hook_fn, ablation_method=ablation_method))
            self.mlp_ablation_handles[layer_idx] = hook
            self.mlp_ablation_positions.append(layer_idx)
            handles[layer_idx] = hook
        return handles

    def add_head_attribution_hooks(self, heads_to_attribute: list[tuple[int, int]]) -> dict[int, RemovableHandle]:
        heads_by_layer: dict[int, list[int]] = {}
        for layer_idx, head_idx in heads_to_attribute:
            heads_by_layer.setdefault(layer_idx, []).append(head_idx)
            self.head_attribution_positions.setdefault(layer_idx, []).append(head_idx)
        handles = self.head_attribution_handles
        for layer_idx, head_indices in heads_by_layer.items():
            block = self.model.encoder.block[layer_idx]  # type: ignore[attr-defined]
            target = block.layer[0].self_attention.o
            hook = target.register_forward_hook(
                partial(self._attribute_head_hook_fn, head_indices=head_indices, layer_idx=layer_idx)
            )
            handles[layer_idx] = hook
        return handles

    def add_mlp_attribution_hooks(self, layers_to_attribute: list[int]) -> dict[int, RemovableHandle]:
        handles = self.mlp_attribution_handles
        for layer_idx in layers_to_attribute:
            block = self.model.encoder.block[layer_idx]  # type: ignore[attr-defined]
            target = block.layer[2].mlp
            hook = target.register_forward_hook(partial(self._attribute_mlp_hook_fn, layer_idx=layer_idx))
            handles[layer_idx] = hook
            if layer_idx not in self.mlp_attribution_positions:
                self.mlp_attribution_positions.append(layer_idx)
        return handles

    def add_read_stream_hooks(self, layer_idxs: list[int]) -> dict[int, RemovableHandle]:
        handles = {}
        for layer_idx in layer_idxs:
            block = self.model.encoder.block[layer_idx]  # type: ignore[attr-defined]
            hook = block.register_forward_hook(partial(self._read_stream_hook_fn, layer_idx=layer_idx))
            self.read_stream_handles[layer_idx] = hook
            handles[layer_idx] = hook
        return handles

    # ---------- Cleanup ----------
    def reset_attribution_inputs_and_outputs(self) -> None:
        self.head_attribution_outputs.clear()
        self.mlp_attribution_inputs.clear()
        self.mlp_attribution_outputs.clear()
        self.read_stream_inputs.clear()
        self.read_stream_outputs.clear()

    def remove_all_hooks(self) -> None:
        for h in (
            list(self.time_head_ablation_handles.values())
            + list(self.group_head_ablation_handles.values())
            + list(self.head_ablation_handles.values())
            + list(self.mlp_ablation_handles.values())
            + list(self.head_attribution_handles.values())
            + list(self.mlp_attribution_handles.values())
            + list(self.read_stream_handles.values())
        ):
            h.remove()
        self.time_head_ablation_handles.clear()
        self.group_head_ablation_handles.clear()
        self.head_ablation_handles.clear()
        self.mlp_ablation_handles.clear()
        self.head_attribution_handles.clear()
        self.mlp_attribution_handles.clear()
        self.read_stream_handles.clear()

        self.time_head_ablation_positions.clear()
        self.group_head_ablation_positions.clear()
        self.head_ablation_positions.clear()
        self.mlp_ablation_positions.clear()
        self.head_attribution_positions.clear()
        self.head_attribution_outputs.clear()
        self.mlp_attribution_inputs.clear()
        self.mlp_attribution_outputs.clear()
        self.read_stream_inputs.clear()
        self.read_stream_outputs.clear()

    def remove_head_attribution_hooks(self) -> None:
        for h in self.head_attribution_handles.values():
            h.remove()
        self.head_attribution_handles.clear()
        self.head_attribution_positions.clear()
        self.head_attribution_outputs.clear()

    def remove_mlp_attribution_hooks(self) -> None:
        for h in self.mlp_attribution_handles.values():
            h.remove()
        self.mlp_attribution_handles.clear()
        self.mlp_attribution_positions.clear()
        self.mlp_attribution_inputs.clear()
        self.mlp_attribution_outputs.clear()

    def remove_read_stream_hooks(self) -> None:
        for h in self.read_stream_handles.values():
            h.remove()
        self.read_stream_handles.clear()
        self.read_stream_inputs.clear()
        self.read_stream_outputs.clear()

    # ---------- Unsupported BaseCircuitLens pieces ----------
    def unembed_residual(self, residual: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Chronos-2 does not expose token unembedding.")

    def sample_tokens(self, residual: torch.Tensor, do_sample: bool = False) -> torch.Tensor:
        raise NotImplementedError("Chronos-2 is not a token generator.")
