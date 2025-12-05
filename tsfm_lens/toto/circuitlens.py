from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.hooks import RemovableHandle

from tsfm_lens.circuitlens import BaseCircuitLens
from tsfm_lens.toto.pipeline import TotoForecasterCustom


@dataclass
class CircuitLensToto(TotoForecasterCustom, BaseCircuitLens):
    """
    Hook-based inspection utilities for Toto modules.

    Supports:
    - Attention head ablation on Transformer blocks
    - MLP (feedforward) ablation on Transformer blocks
    - Residual stream reading per layer
    - Capturing intermediate outputs from tokenizer and output projections

    TODO: add attributions
    """

    # Ablation positions and handles
    head_ablation_positions: dict[int, dict[int, str]] = field(default_factory=dict)
    head_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    attn_ablation_positions: dict[int, str] = field(default_factory=dict)
    attn_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    mlp_ablation_positions: list[int] = field(default_factory=list)
    mlp_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # Attribution tracking
    head_attribution_positions: dict[int, list[int]] = field(default_factory=dict)
    head_attribution_outputs: dict[int, dict[int, list[torch.Tensor]]] = field(default_factory=dict)
    head_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    mlp_attribution_positions: list[int] = field(default_factory=list)
    mlp_attribution_inputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    mlp_attribution_outputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    mlp_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # Attribution/read handles and outputs
    read_stream_handles: dict[int, RemovableHandle] = field(default_factory=dict)
    read_stream_outputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    read_stream_inputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)

    custom_handles: dict[str, RemovableHandle] = field(default_factory=dict)
    custom_outputs: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        self.num_layers = self.model.num_layers
        self.num_heads = self.model.transformer.layers[0].num_heads
        self.is_decoder_only = True

    def set_to_eval_mode(self) -> None:
        """Set the model to evaluation mode."""
        self.model.eval()

    def add_ablation_hooks_v2(
        self,
        ablations_types: list[Literal["head", "mlp"]],
        heads_to_ablate: list[tuple[int, int]],
    ):
        """
        Add ablation hooks to the pipeline.
        Args:
            ablations_types: The types of ablations to add (e.g. "head", "mlp")
            heads_to_ablate: The heads to ablate
        """
        self.remove_all_hooks()
        self.reset_attribution_inputs_and_outputs()
        if "head" in ablations_types:
            self.add_head_ablation_hooks(heads_to_ablate, ablation_method="zero")

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
    ) -> None:
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

        # NOTE: Toto patch length is 64 = self.model.patch_embed.projection.in_features
        x = input[0]  # shape: [num_samples, batch, num_patches, d_model]
        h = self.num_heads
        num_samples, b, p, d = x.shape
        d = d // h

        # Reshape to (num_samples, batch, seq_len, n_heads, head_size)
        reshaped = x.reshape(num_samples, b, p, h, d)

        for head_idx in head_indices:
            # reshaped[:, :, :, head_idx, :] shape: (num_samples, batch, seq_len, head_size)
            if ablation_method == "zero":
                reshaped[:, :, :, head_idx, :] = 0.0
            elif ablation_method == "mean":
                # reshaped[:, :, :, head_idx, :].mean(dim=-1, keepdim=True) shape: (num_samples, batch, seq_len, 1)
                reshaped[:, :, :, head_idx, :] = reshaped[:, :, :, head_idx, :].mean(dim=-1, keepdim=True)
            else:
                raise ValueError(f"Invalid ablation method: {ablation_method}")

        modified_input = reshaped.reshape(num_samples, b, p, -1)
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
        inputs: Any,
        _outputs: Any,
        head_indices: list[int],
        layer_idx: int,
    ) -> None:
        """Attribute per-head contributions at the attention output projection."""
        if not isinstance(inputs, tuple) or len(inputs) != 1:
            raise ValueError("Expected input to be a tuple of length 1 for attention output projection hooks")

        x = inputs[0]
        if x.dim() == 3:
            # No sampling dimension; promote for shared logic
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"Unexpected attention output shape {tuple(x.shape)}")

        num_samples, batch_size, seq_len, d_model = x.shape
        if d_model % self.num_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {self.num_heads}")
        head_dim = d_model // self.num_heads

        outputs_dict = self.head_attribution_outputs
        if layer_idx not in outputs_dict:
            outputs_dict[layer_idx] = {}

        weight_t = module.weight.T.to(x.dtype)

        for head_idx in head_indices:
            head_slice = slice(head_idx * head_dim, (head_idx + 1) * head_dim)
            x_h = x[..., head_slice]  # [num_samples, batch, seq_len, head_dim]
            w_h_t = weight_t[head_slice, :]  # [head_dim, d_model]

            head_output = torch.einsum("nbph,hd->nbpd", x_h, w_h_t)
            head_output = head_output.reshape(num_samples * batch_size, seq_len, d_model)

            if head_idx not in outputs_dict[layer_idx]:
                outputs_dict[layer_idx][head_idx] = []
            outputs_dict[layer_idx][head_idx].append(head_output)

    def _attribute_mlp_hook_fn(
        self,
        module: torch.nn.Module,
        inputs: Any,
        outputs: Any,
        layer_idx: int,
    ) -> None:
        """Store inputs and outputs for MLP attribution analysis."""
        x_in = inputs[0] if isinstance(inputs, tuple) else inputs
        x_out = outputs[0] if isinstance(outputs, tuple) else outputs

        self.mlp_attribution_inputs.setdefault(layer_idx, []).append(x_in)
        self.mlp_attribution_outputs.setdefault(layer_idx, []).append(x_out)

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
        # handles = self.head_ablation_handles
        for layer_idx, head_indices in heads_by_layer.items():
            target_layer = self.model.transformer.layers[layer_idx].attention.wO
            hook = target_layer.register_forward_hook(
                partial(
                    self._ablate_head_hook_fn,
                    head_indices=head_indices,
                    ablation_method=ablation_method,
                )
            )
            # handles[layer_idx] = hook
            self.head_ablation_handles[layer_idx] = hook

        return self.head_ablation_handles

    def add_head_attribution_hooks(
        self,
        heads_to_attribute: list[tuple[int, int]],
    ) -> dict[int, RemovableHandle]:
        """Register hooks to capture per-head attention outputs."""
        heads_by_layer: dict[int, list[int]] = {}
        for layer_idx, head_idx in heads_to_attribute:
            heads_by_layer.setdefault(layer_idx, []).append(head_idx)
            self.head_attribution_positions.setdefault(layer_idx, []).append(head_idx)

        handles = self.head_attribution_handles
        for layer_idx, head_indices in heads_by_layer.items():
            target_layer = self.model.transformer.layers[layer_idx].attention.wO
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
            layer = self.model.transformer.layers[layer_idx].attention
            hook = layer.register_forward_hook(partial(self._ablate_attn_hook_fn, ablation_method=ablation_method))
            self.attn_ablation_handles[layer_idx] = hook
            self.attn_ablation_positions[layer_idx] = ablation_method
            handles[layer_idx] = hook
        return handles

    def add_mlp_attribution_hooks(self, layers_to_attribute: list[int]) -> dict[int, RemovableHandle]:
        """Register hooks to capture MLP inputs/outputs per layer."""
        handles = self.mlp_attribution_handles
        for layer_idx in layers_to_attribute:
            target = self.model.transformer.layers[layer_idx].mlp
            hook = target.register_forward_hook(partial(self._attribute_mlp_hook_fn, layer_idx=layer_idx))
            handles[layer_idx] = hook
            if layer_idx not in self.mlp_attribution_positions:
                self.mlp_attribution_positions.append(layer_idx)
        return handles

    def add_mlp_ablation_hooks(
        self,
        layers_to_ablate: list[int],
        ablation_method: Literal["zero", "mean"] = "zero",
    ) -> dict[int, RemovableHandle]:
        handles = {}
        for layer_idx in layers_to_ablate:
            # TotoBackbone implements MLP as Sequential with Linear layers
            layer = self.model.transformer.layers[layer_idx]
            target = layer.mlp
            hook = target.register_forward_hook(partial(self._ablate_mlp_hook_fn, ablation_method=ablation_method))
            self.mlp_ablation_handles[layer_idx] = hook
            self.mlp_ablation_positions.append(layer_idx)
            handles[layer_idx] = hook
        return handles

    def add_read_stream_hooks(self, layer_idxs: list[int]) -> dict[int, RemovableHandle]:
        handles = {}
        for layer_idx in layer_idxs:
            layer = self.model.transformer.layers[layer_idx]
            hook = layer.register_forward_hook(partial(self._read_stream_hook_fn, layer_idx=layer_idx))
            self.read_stream_handles[layer_idx] = hook
            handles[layer_idx] = hook
        return handles

    def add_custom_capture_hook(self, module: torch.nn.Module, key: str) -> dict[str, RemovableHandle]:
        hook = module.register_forward_hook(partial(self._capture_hook_fn, key=key))
        self.custom_handles[key] = hook
        return self.custom_handles

    def reset_attribution_inputs_and_outputs(self) -> None:
        """Clear cached tensors collected by attribution hooks."""
        self.head_attribution_outputs.clear()
        self.mlp_attribution_inputs.clear()
        self.mlp_attribution_outputs.clear()
        self.read_stream_inputs.clear()
        self.read_stream_outputs.clear()
        self.custom_outputs.clear()

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

    # ---------- Cleanup ----------
    def remove_all_hooks(self) -> None:
        for hook in self.head_ablation_handles.values():
            hook.remove()
        for h in self.attn_ablation_handles.values():
            h.remove()
        for h in self.mlp_ablation_handles.values():
            h.remove()
        for h in self.head_attribution_handles.values():
            h.remove()
        for h in self.mlp_attribution_handles.values():
            h.remove()
        for h in self.read_stream_handles.values():
            h.remove()
        for h in self.custom_handles.values():
            h.remove()

        self.head_ablation_handles.clear()
        self.attn_ablation_handles.clear()
        self.mlp_ablation_handles.clear()
        self.head_attribution_handles.clear()
        self.mlp_attribution_handles.clear()
        self.read_stream_handles.clear()
        self.custom_handles.clear()

        self.attn_ablation_positions.clear()
        self.mlp_ablation_positions.clear()
        self.head_attribution_positions.clear()
        self.head_attribution_outputs.clear()
        self.mlp_attribution_positions.clear()
        self.mlp_attribution_inputs.clear()
        self.mlp_attribution_outputs.clear()
        self.read_stream_inputs.clear()
        self.read_stream_outputs.clear()
        self.custom_outputs.clear()
