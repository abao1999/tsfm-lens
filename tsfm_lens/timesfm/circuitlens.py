from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.hooks import RemovableHandle

from tsfm_lens.circuitlens import BaseCircuitLens

from .pipeline import TimesFMPipelineCustom


@dataclass
class CircuitLensTimesFM(TimesFMPipelineCustom, BaseCircuitLens):
    """
    Hook-based inspection utilities for TimesFM modules.

    Supports:
    - Attention head ablation on Transformer blocks
    - MLP (feedforward) ablation on Transformer blocks
    - Residual stream reading per layer
    - Capturing intermediate outputs from tokenizer and output projections

    Note: The TimesFM reference implements a custom Transformer stack in
    `timesfm_references.torch.transformer.Transformer`. We attach hooks at
    the top-level per-layer module, and also support optional handles to
    the internal attention/MLP submodules when present.

    TODO: add attributions
    """

    # Ablation positions and handles
    head_ablation_positions: dict[int, dict[int, str]] = field(default_factory=dict)
    head_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    attn_ablation_positions: dict[int, str] = field(default_factory=dict)
    attn_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    mlp_ablation_positions: list[int] = field(default_factory=list)
    mlp_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # Attribution/read handles and outputs
    # individual cross-attention heads
    head_attribution_positions: dict[int, list[int]] = field(default_factory=dict)
    head_attribution_outputs: dict[int, dict[int, list[torch.Tensor]]] = field(default_factory=dict)
    head_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # residual stream reading
    read_stream_handles: dict[int, RemovableHandle] = field(default_factory=dict)
    read_stream_outputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    read_stream_inputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)

    # custom logit attribution
    custom_handles: dict[str, RemovableHandle] = field(default_factory=dict)
    custom_outputs: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        TimesFMPipelineCustom.__post_init__(self)

        # Initialize attributes required by BaseCircuitLens
        # NOTE: this is weird because timesfm TimesFM_2p5_200M_torch is itself a wrapper around the model
        self.num_heads = self.model.model.config.stacked_transformers.transformer.num_heads
        self.num_layers = self.model.model.config.stacked_transformers.num_layers
        self.quantiles = self.model.model.config.quantiles
        self.is_decoder_only = True

    def set_to_eval_mode(self) -> None:
        """Set the model to evaluation mode."""
        self.model.model.eval()

    def add_ablation_hooks_explicit(
        self,
        ablations_types: list[Literal["head", "mlp"]],
        layers_to_ablate_mlp: list[int],
        heads_to_ablate: list[tuple[int, int]],
    ) -> None:
        """
        Add ablation hooks to the pipeline.
        Args:
            ablations_types: The types of ablations to add (e.g. "head", "mlp")
            layers_to_ablate_mlp: The layers to ablate the MLP of
            heads_to_ablate: The heads to ablate
        """
        self.remove_all_hooks()
        self.reset_attribution_inputs_and_outputs()
        if "head" in ablations_types:
            # Ablate all heads per layer
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
    ):
        """
        Add ablation hooks to the pipeline.
        Args:
            ablations_types: The types of ablations to add (e.g. "head", "mlp")
            layers_to_ablate: The layers to ablate
            ablate_n_heads_per_layer: The number of heads to ablate per layer
            rng: The random number generator to use
        """
        if rng is None:
            rng = np.random.default_rng()
        if verbose:
            print(f"Ablations: {ablations_types} on layers {layers_to_ablate}")

        self.remove_all_hooks()
        self.reset_attribution_inputs_and_outputs()
        if "head" in ablations_types:
            # Ablate all heads per layer
            if ablate_n_heads_per_layer is None:
                # List of tuples of (layer_idx, head_idx) to ablate
                heads_to_ablate = [(layer, head) for layer in layers_to_ablate for head in range(self.num_heads)]
            # Ablate a fixed number of heads (random subset of heads) per layer
            else:
                heads_to_ablate = []
                for layer in layers_to_ablate:
                    heads_to_ablate_for_layer = rng.choice(
                        list(range(self.num_heads)), size=ablate_n_heads_per_layer, replace=False
                    )
                    heads_to_ablate.extend([(layer, head) for head in heads_to_ablate_for_layer])  # type: ignore
            self.add_head_ablation_hooks(
                heads_to_ablate,
                ablation_method="zero",
            )
        if "mlp" in ablations_types:
            self.add_mlp_ablation_hooks(layers_to_ablate)

    # ---------- Hook functions ----------
    def _ablate_head_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        head_indices: list[int],
        ablation_method: Literal["zero", "mean"],
    ) -> Any:
        """Hook function to ablate attention head outputs."""
        if not isinstance(input, tuple):
            raise ValueError("Expected input to be a tuple")
        if len(input) != 1:
            raise ValueError("Expected input to be a tuple of length 1")

        x = input[0]  # shape: [batch, patches, d]
        h = self.model.model.config.stacked_transformers.transformer.num_heads
        b, p, d = x.shape
        d = d // h

        # Reshape to (batch, seq_len, n_heads, head_size)
        reshaped = x.reshape(b, p, h, d)

        for head_idx in head_indices:
            # reshaped[:, :, head_idx, :] shape: (batch, seq_len, head_size)
            if ablation_method == "zero":
                reshaped[:, :, head_idx, :] = 0.0
            elif ablation_method == "mean":
                # reshaped[:, :, head_idx, :].mean(dim=-1, keepdim=True) shape: (batch, seq_len, 1)
                reshaped[:, :, head_idx, :] = reshaped[:, :, head_idx, :].mean(dim=-1, keepdim=True)
            else:
                raise ValueError(f"Invalid ablation method: {ablation_method}")

        modified_input = x.reshape(b, p, -1)
        return module.forward(modified_input)

    def _ablate_attn_hook_fn(
        self, module: torch.nn.Module, _in: Any, out: Any, ablation_method: Literal["zero", "mean"]
    ) -> Any:
        if ablation_method == "zero":
            return torch.zeros_like(out[0]), out[1]
        elif ablation_method == "mean":
            return out[0].mean(dim=-1, keepdim=True), out[1]

    def _ablate_mlp_hook_fn(
        self, module: torch.nn.Module, _in: Any, out: Any, ablation_method: Literal["zero", "mean"]
    ) -> Any:
        if ablation_method == "zero":
            return torch.zeros_like(out)
        return out.mean(dim=-1, keepdim=True)

    def _read_stream_hook_fn(self, module: torch.nn.Module, inputs: Any, outputs: Any, layer_idx: int) -> None:
        x_in = inputs[0] if isinstance(inputs, tuple) else inputs
        x_out = outputs[0] if isinstance(outputs, tuple) else outputs
        self.read_stream_inputs.setdefault(layer_idx, []).append(x_in)
        self.read_stream_outputs.setdefault(layer_idx, []).append(x_out)

    def _capture_hook_fn(self, module: torch.nn.Module, _in: Any, out: Any, key: str) -> None:
        self.custom_outputs[key] = out[0] if isinstance(out, tuple) else out

    def _attribute_head_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        head_indices: list[int],
        layer_idx: int,
    ) -> None:
        """Hook function to attribute individual attention head contributions at the attention
        output projection (attn.out).

        Notes
        -----
        - Given y = x @ W.T (+ b), the per-head contribution is x_h @ W[:, slice_h].T
        """
        if not isinstance(input, tuple):
            raise ValueError("Expected input to be a tuple")
        if len(input) != 1:
            raise ValueError("Expected input to be a tuple of length 1")

        x = input[0]  # shape: [batch, patches, d_model]
        bsz, seq_len, d_model = x.shape
        num_heads = self.model.model.config.stacked_transformers.transformer.num_heads
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")
        head_dim = d_model // num_heads

        reshaped = x.reshape(bsz, seq_len, num_heads, -1)

        # Prepare storage
        if layer_idx not in self.head_attribution_outputs:
            self.head_attribution_outputs[layer_idx] = {}

        # Linear weight: [out_features, in_features]; want [in_features, out_features]
        weight_t = module.weight.T  # [d_model, d_model]

        for head_idx in head_indices:
            # Slice input for this head and corresponding weight columns
            x_h = reshaped[:, :, head_idx, :]  # [bsz, seq_len, head_dim]
            w_h_t = weight_t[head_idx * head_dim : (head_idx + 1) * head_dim, :]  # [head_dim, d_model]

            head_out = x_h @ w_h_t  # [bsz, seq_len, d_model]

            if head_idx not in self.head_attribution_outputs[layer_idx]:
                self.head_attribution_outputs[layer_idx][head_idx] = []
            self.head_attribution_outputs[layer_idx][head_idx].append(head_out)

    # ---------- Public API ----------
    def add_head_ablation_hooks(
        self,
        heads_to_ablate: list[tuple[int, int]],
        ablation_method: Literal["zero", "mean"] = "zero",
    ) -> dict[int, RemovableHandle]:
        """Add hooks to ablate attention heads during model generation."""
        if ablation_method not in ["zero", "mean"]:
            raise ValueError(f"Invalid ablation method: {ablation_method}")

        # Group heads by layer
        heads_by_layer: dict[int, list[int]] = {}
        for layer_idx, head_idx in heads_to_ablate:
            heads_by_layer.setdefault(layer_idx, []).append(head_idx)
            self.head_ablation_positions.setdefault(layer_idx, {})[head_idx] = ablation_method

        # Create hooks for each layer
        handles = self.head_ablation_handles
        for layer_idx, head_indices in heads_by_layer.items():
            target_layer = self.model.model.stacked_xf[layer_idx].attn.out
            hook = target_layer.register_forward_hook(
                partial(
                    self._ablate_head_hook_fn,
                    head_indices=head_indices,
                    ablation_method=ablation_method,
                )
            )
            handles[layer_idx] = hook

        return handles

    def add_head_attribution_hooks(
        self,
        heads_to_attribute: list[tuple[int, int]],
    ) -> dict[int, RemovableHandle]:
        """Add hooks to attribute individual attention head outputs at attn.out.

        Parameters
        ----------
        heads_to_attribute : list of (layer_idx, head_idx)
            The specific heads to capture contributions for.
        """
        # Group heads by layer
        heads_by_layer: dict[int, list[int]] = {}
        for layer_idx, head_idx in heads_to_attribute:
            heads_by_layer.setdefault(layer_idx, []).append(head_idx)
            self.head_attribution_positions.setdefault(layer_idx, []).append(head_idx)

        # Create hooks per layer on the attention output projection
        handles = self.head_attribution_handles
        for layer_idx, head_indices in heads_by_layer.items():
            target_layer = self.model.model.stacked_xf[layer_idx].attn.out
            hook = target_layer.register_forward_hook(
                partial(
                    self._attribute_head_hook_fn,
                    head_indices=head_indices,
                    layer_idx=layer_idx,
                )
            )
            handles[layer_idx] = hook

        return handles

    def add_attn_ablation_hooks(
        self,
        layers_to_ablate: list[int],
        ablation_method: Literal["zero", "mean"] = "zero",
    ) -> dict[int, RemovableHandle]:
        handles = {}
        for layer_idx in layers_to_ablate:
            layer = self.model.model.stacked_xf[layer_idx].attn
            hook = layer.register_forward_hook(partial(self._ablate_attn_hook_fn, ablation_method=ablation_method))
            self.attn_ablation_handles[layer_idx] = hook
            self.attn_ablation_positions[layer_idx] = ablation_method
            handles[layer_idx] = hook
        return handles

    def add_mlp_ablation_hooks(
        self,
        layers_to_ablate: list[int],
        ablation_method: Literal["zero", "mean"] = "zero",
    ) -> dict[int, RemovableHandle]:
        handles = {}
        for layer_idx in layers_to_ablate:
            # Many Transformer blocks implement FFN as attribute `ff` or similar.
            # Fall back to hooking the layer itself if FFN attribute is absent.
            layer = self.model.model.stacked_xf[layer_idx]
            target = layer.ff1
            hook = target.register_forward_hook(partial(self._ablate_mlp_hook_fn, ablation_method=ablation_method))
            self.mlp_ablation_handles[layer_idx] = hook
            self.mlp_ablation_positions.append(layer_idx)
            handles[layer_idx] = hook
        return handles

    def add_read_stream_hooks(self, layer_idxs: list[int]) -> dict[int, RemovableHandle]:
        handles = {}
        for layer_idx in layer_idxs:
            layer = self.model.model.stacked_xf[layer_idx]
            hook = layer.register_forward_hook(partial(self._read_stream_hook_fn, layer_idx=layer_idx))
            self.read_stream_handles[layer_idx] = hook
            handles[layer_idx] = hook
        return handles

    def add_custom_capture_hook(self, module: torch.nn.Module, key: str) -> dict[str, RemovableHandle]:
        hook = module.register_forward_hook(partial(self._capture_hook_fn, key=key))
        self.custom_handles[key] = hook
        return self.custom_handles

    # ---------- Attribution utilities ----------
    def reset_attribution_inputs_and_outputs(self) -> None:
        """Reset stored attribution outputs for heads and any custom captures."""
        self.head_attribution_outputs.clear()
        # Keep positions to allow re-use across runs if desired
        # Do not remove hooks here; this only clears stored tensors

    def remove_head_attribution_hooks(self) -> None:
        """Remove head attribution hooks and clear related state."""
        for hook in self.head_attribution_handles.values():
            hook.remove()
        self.head_attribution_handles.clear()
        self.head_attribution_positions.clear()
        self.head_attribution_outputs.clear()

    # ---------- Cleanup ----------
    def remove_all_hooks(self) -> None:
        for hook in self.head_ablation_handles.values():
            hook.remove()
        for h in self.attn_ablation_handles.values():
            h.remove()
        for h in self.mlp_ablation_handles.values():
            h.remove()
        for h in self.read_stream_handles.values():
            h.remove()
        for h in self.custom_handles.values():
            h.remove()
        for h in self.head_attribution_handles.values():
            h.remove()

        self.head_ablation_handles.clear()
        self.attn_ablation_handles.clear()
        self.mlp_ablation_handles.clear()
        self.read_stream_handles.clear()
        self.custom_handles.clear()
        self.head_attribution_handles.clear()

        self.attn_ablation_positions.clear()
        self.mlp_ablation_positions.clear()
        self.read_stream_inputs.clear()
        self.read_stream_outputs.clear()
        self.custom_outputs.clear()
        self.head_attribution_positions.clear()
        self.head_attribution_outputs.clear()
