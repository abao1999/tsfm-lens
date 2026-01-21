from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.hooks import RemovableHandle

from tsfm_lens.circuitlens import BaseCircuitLens
from tsfm_lens.moirai.pipeline import MoiraiPipelineCustom


@dataclass
class CircuitLensMoirai(MoiraiPipelineCustom, BaseCircuitLens):
    """
    CircuitLens utilities for Uni2TS Moirai (encoder-only) models.

    Supports:
    - Attention head ablation at the self-attention output projection
    - MLP ablation at the feed-forward block
    - Per-head and MLP attribution captures
    - Residual stream capture at each Transformer layer
    """

    head_ablation_positions: dict[int, dict[int, str]] = field(default_factory=dict)
    head_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    mlp_ablation_positions: list[int] = field(default_factory=list)
    mlp_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    head_attribution_positions: dict[int, list[int]] = field(default_factory=dict)
    head_attribution_outputs: dict[int, dict[int, list[torch.Tensor]]] = field(default_factory=dict)
    head_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    mlp_attribution_positions: list[int] = field(default_factory=list)
    mlp_attribution_inputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    mlp_attribution_outputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    mlp_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    read_stream_inputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    read_stream_outputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    read_stream_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    custom_handles: dict[str, RemovableHandle] = field(default_factory=dict)
    custom_outputs: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        MoiraiPipelineCustom.__post_init__(self)

        encoder = self.model.module.encoder
        self.num_layers = len(encoder.layers)
        self.num_heads = encoder.layers[0].self_attn.num_heads
        self.is_decoder_only = True

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
            raise ValueError("Expected attention projection input as a single-element tuple.")

        x = input[0]
        if x.dim() != 3:
            raise ValueError(f"Unexpected attention projection input shape {tuple(x.shape)}")

        batch, seq_len, d_model = x.shape
        if d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({self.num_heads})")
        head_dim = d_model // self.num_heads

        reshaped = x.view(batch, seq_len, self.num_heads, head_dim)
        for head_idx in head_indices:
            if head_idx >= self.num_heads:
                raise ValueError(f"Head index {head_idx} out of range for {self.num_heads} heads")
            if ablation_method == "zero":
                reshaped[:, :, head_idx, :] = 0.0
            elif ablation_method == "mean":
                reshaped[:, :, head_idx, :] = reshaped[:, :, head_idx, :].mean(dim=-1, keepdim=True)
            else:
                raise ValueError(f"Invalid ablation method {ablation_method}")

        modified = reshaped.view(batch, seq_len, -1)
        return module.forward(modified)

    def _ablate_mlp_hook_fn(
        self,
        module: torch.nn.Module,
        _input: Any,
        output: Any,
        ablation_method: Literal["zero", "mean"],
    ) -> Any:
        out = output[0] if isinstance(output, tuple) else output
        if ablation_method == "zero":
            return torch.zeros_like(out)
        return out.mean(dim=-1, keepdim=True)

    def _read_stream_hook_fn(self, module: torch.nn.Module, inputs: Any, outputs: Any, layer_idx: int) -> None:
        x_in = inputs[0] if isinstance(inputs, tuple) else inputs
        x_out = outputs[0] if isinstance(outputs, tuple) else outputs
        self.read_stream_inputs.setdefault(layer_idx, []).append(x_in)
        self.read_stream_outputs.setdefault(layer_idx, []).append(x_out)

    def _capture_hook_fn(self, module: torch.nn.Module, _input: Any, output: Any, key: str) -> None:
        self.custom_outputs[key] = output[0] if isinstance(output, tuple) else output

    def _attribute_head_hook_fn(
        self,
        module: torch.nn.Module,
        inputs: Any,
        _outputs: Any,
        head_indices: list[int],
        layer_idx: int,
    ) -> None:
        if not isinstance(inputs, tuple) or len(inputs) != 1:
            raise ValueError("Expected attention projection input as a single-element tuple.")

        x = inputs[0]
        if x.dim() != 3:
            raise ValueError(f"Unexpected attention projection input shape {tuple(x.shape)}")

        batch, seq_len, d_model = x.shape
        if d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({self.num_heads})")
        head_dim = d_model // self.num_heads

        weight_t = module.weight.T.to(x.dtype)
        layer_store = self.head_attribution_outputs.setdefault(layer_idx, {})

        for head_idx in head_indices:
            head_slice = slice(head_idx * head_dim, (head_idx + 1) * head_dim)
            x_h = x[:, :, head_slice]
            w_h_t = weight_t[head_slice, :]
            head_output = torch.einsum("bsh,hd->bsd", x_h, w_h_t)
            layer_store.setdefault(head_idx, []).append(head_output)

    def _attribute_mlp_hook_fn(
        self,
        module: torch.nn.Module,
        inputs: Any,
        outputs: Any,
        layer_idx: int,
    ) -> None:
        x_in = inputs[0] if isinstance(inputs, tuple) else inputs
        x_out = outputs[0] if isinstance(outputs, tuple) else outputs
        self.mlp_attribution_inputs.setdefault(layer_idx, []).append(x_in)
        self.mlp_attribution_outputs.setdefault(layer_idx, []).append(x_out)

    # ---------- Public API ----------
    def set_to_eval_mode(self) -> None:
        self.model.eval()

    def add_ablation_hooks_explicit(
        self,
        ablations_types: list[Literal["head", "mlp"]],
        layers_to_ablate_mlp: list[int],
        heads_to_ablate: list[tuple[int, int]],
    ) -> None:
        self.remove_all_hooks()
        self.reset_attribution_inputs_and_outputs()

        if "head" in ablations_types:
            self.add_head_ablation_hooks(heads_to_ablate, ablation_method="zero")
        if "mlp" in ablations_types:
            self.add_mlp_ablation_hooks(layers_to_ablate_mlp)

    def add_ablation_hooks(
        self,
        ablations_types: list[Literal["head", "mlp"]],
        layers_to_ablate: list[int],
        ablate_n_heads_per_layer: int | None,
        rng: np.random.Generator | None = None,
        verbose: bool = False,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        if verbose:
            print(f"Ablations: {ablations_types} on layers {layers_to_ablate}")

        self.remove_all_hooks()
        self.reset_attribution_inputs_and_outputs()

        if "head" in ablations_types:
            if ablate_n_heads_per_layer is None:
                heads_to_ablate = [(layer, head) for layer in layers_to_ablate for head in range(self.num_heads)]
            else:
                heads_to_ablate = []
                for layer in layers_to_ablate:
                    heads = rng.choice(list(range(self.num_heads)), size=ablate_n_heads_per_layer, replace=False)
                    heads_to_ablate.extend([(layer, h) for h in heads])
            self.add_head_ablation_hooks(heads_to_ablate, ablation_method="zero")

        if "mlp" in ablations_types:
            self.add_mlp_ablation_hooks(layers_to_ablate)

    def add_head_ablation_hooks(
        self,
        heads_to_ablate: list[tuple[int, int]],
        ablation_method: Literal["zero", "mean"] = "zero",
    ) -> dict[int, RemovableHandle]:
        heads_by_layer: dict[int, list[int]] = {}
        for layer_idx, head_idx in heads_to_ablate:
            heads_by_layer.setdefault(layer_idx, []).append(head_idx)
            self.head_ablation_positions.setdefault(layer_idx, {})[head_idx] = ablation_method

        for layer_idx, head_indices in heads_by_layer.items():
            target_layer = self.model.module.encoder.layers[layer_idx].self_attn.out_proj
            hook = target_layer.register_forward_hook(
                partial(self._ablate_head_hook_fn, head_indices=head_indices, ablation_method=ablation_method)
            )
            self.head_ablation_handles[layer_idx] = hook

        return self.head_ablation_handles

    def add_mlp_ablation_hooks(
        self,
        layers_to_ablate: list[int],
        ablation_method: Literal["zero", "mean"] = "zero",
    ) -> dict[int, RemovableHandle]:
        handles: dict[int, RemovableHandle] = {}
        for layer_idx in layers_to_ablate:
            target = self.model.module.encoder.layers[layer_idx].ffn
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
            target_layer = self.model.module.encoder.layers[layer_idx].self_attn.out_proj
            hook = target_layer.register_forward_hook(
                partial(self._attribute_head_hook_fn, head_indices=head_indices, layer_idx=layer_idx)
            )
            handles[layer_idx] = hook
        return handles

    def add_mlp_attribution_hooks(self, layers_to_attribute: list[int]) -> dict[int, RemovableHandle]:
        handles = self.mlp_attribution_handles
        for layer_idx in layers_to_attribute:
            target = self.model.module.encoder.layers[layer_idx].ffn
            hook = target.register_forward_hook(partial(self._attribute_mlp_hook_fn, layer_idx=layer_idx))
            handles[layer_idx] = hook
            if layer_idx not in self.mlp_attribution_positions:
                self.mlp_attribution_positions.append(layer_idx)
        return handles

    def add_read_stream_hooks(self, layer_idxs: list[int]) -> dict[int, RemovableHandle]:
        handles = {}
        for layer_idx in layer_idxs:
            layer = self.model.module.encoder.layers[layer_idx]
            hook = layer.register_forward_hook(partial(self._read_stream_hook_fn, layer_idx=layer_idx))
            self.read_stream_handles[layer_idx] = hook
            handles[layer_idx] = hook
        return handles

    def add_custom_capture_hook(self, module: torch.nn.Module, key: str) -> dict[str, RemovableHandle]:
        hook = module.register_forward_hook(partial(self._capture_hook_fn, key=key))
        self.custom_handles[key] = hook
        return self.custom_handles

    # ---------- Cleanup ----------
    def reset_attribution_inputs_and_outputs(self) -> None:
        self.head_attribution_outputs.clear()
        self.mlp_attribution_inputs.clear()
        self.mlp_attribution_outputs.clear()
        self.read_stream_inputs.clear()
        self.read_stream_outputs.clear()
        self.custom_outputs.clear()

    def remove_head_ablation_hooks(self) -> None:
        for hook in self.head_ablation_handles.values():
            hook.remove()
        self.head_ablation_handles.clear()
        self.head_ablation_positions.clear()

    def remove_mlp_ablation_hooks(self) -> None:
        for hook in self.mlp_ablation_handles.values():
            hook.remove()
        self.mlp_ablation_handles.clear()
        self.mlp_ablation_positions.clear()

    def remove_head_attribution_hooks(self) -> None:
        for hook in self.head_attribution_handles.values():
            hook.remove()
        self.head_attribution_handles.clear()
        self.head_attribution_positions.clear()
        self.head_attribution_outputs.clear()

    def remove_mlp_attribution_hooks(self) -> None:
        for hook in self.mlp_attribution_handles.values():
            hook.remove()
        self.mlp_attribution_handles.clear()
        self.mlp_attribution_positions.clear()
        self.mlp_attribution_inputs.clear()
        self.mlp_attribution_outputs.clear()

    def remove_read_stream_hooks(self) -> None:
        for hook in self.read_stream_handles.values():
            hook.remove()
        self.read_stream_handles.clear()
        self.read_stream_inputs.clear()
        self.read_stream_outputs.clear()

    def remove_all_hooks(self) -> None:
        for hook in self.head_ablation_handles.values():
            hook.remove()
        for hook in self.mlp_ablation_handles.values():
            hook.remove()
        for hook in self.head_attribution_handles.values():
            hook.remove()
        for hook in self.mlp_attribution_handles.values():
            hook.remove()
        for hook in self.read_stream_handles.values():
            hook.remove()
        for hook in self.custom_handles.values():
            hook.remove()

        self.head_ablation_handles.clear()
        self.mlp_ablation_handles.clear()
        self.head_attribution_handles.clear()
        self.mlp_attribution_handles.clear()
        self.read_stream_handles.clear()
        self.custom_handles.clear()

        self.head_ablation_positions.clear()
        self.mlp_ablation_positions.clear()
        self.head_attribution_positions.clear()
        self.head_attribution_outputs.clear()
        self.mlp_attribution_inputs.clear()
        self.mlp_attribution_outputs.clear()
        self.read_stream_inputs.clear()
        self.read_stream_outputs.clear()
        self.custom_outputs.clear()

    # ---------- Unsupported ops ----------
    def unembed_residual(self, residual: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Moirai does not expose a token unembedding layer.")

    def sample_tokens(self, residual: torch.Tensor, do_sample: bool = False) -> torch.Tensor:
        raise NotImplementedError("Moirai is not a token-generative model; call `predict` instead.")
