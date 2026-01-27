from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.hooks import RemovableHandle

from tsfm_lens.chronos.pipeline import ChronosPipelineCustom
from tsfm_lens.circuitlens import BaseCircuitLens

@dataclass
class CircuitLensChronos(ChronosPipelineCustom, BaseCircuitLens):
    """
    A class for performing circuit lens analysis on Chronos (wrapper of T5ForConditionalGeneration).

    CircuitLens supports attention head ablation, residual stream reading, and logit attribution analysis
        to understand model behavior and identify important computational sub-paths.

    Key Features:
        - Attention Head Ablation: Zero out or replace attention head outputs with their mean, to study their impact
        - Residual Stream Reading: Extract hidden states at any layer for analysis
        - Logit Attribution: Track how specific components contribute to final predictions
        - MLP Ablation: Ablate MLP layers to understand their role in computation
        - Hook Management: Automatic cleanup of all registered hooks

    Attributes:
        sa_head_ablation_positions: Tracks self-attention head ablation positions and methods
        ca_head_ablation_positions: Tracks cross-attention head ablation positions and methods
        mlp_ablation_positions: Tracks MLP layer ablation positions
        sa_head_attribution_positions: Tracks self-attention head attribution positions
        sa_attribution_inputs: Tracks self-attention output attribution inputs
        sa_attribution_outputs: Tracks self-attention output attribution outputs
        ca_head_attribution_positions: Tracks cross-attention head attribution positions
        ca_attribution_inputs: Tracks cross-attention output attribution inputs
        ca_attribution_outputs: Tracks cross-attention output attribution outputs
        mlp_attribution_positions: Tracks MLP layer attribution positions
        mlp_attribution_inputs: Tracks MLP layer attribution inputs
        mlp_attribution_outputs: Tracks MLP layer attribution outputs
        read_stream_outputs: Stores residual stream outputs from read hooks
        custom_attribution_outputs: Stores custom attribution outputs

    Note:
        This class is designed specifically for T5 models and assumes the standard T5 architecture
        with decoder blocks containing self-attention, cross-attention, and MLP layers.
    """

    # Ablation tracking: {layer_idx: {head_idx: ablation_method}}
    sa_head_ablation_positions: dict[int, dict[int, str]] = field(default_factory=dict)
    sa_head_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)
    # sa_output_ablation_positions: Dict[int, Dict[int, str]] = field(default_factory=dict)
    # sa_output_ablation_handles: Dict[int, RemovableHandle] = field(default_factory=dict)

    ca_head_ablation_positions: dict[int, dict[int, str]] = field(default_factory=dict)
    ca_head_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)
    # ca_output_ablation_positions: Dict[int, Dict[int, str]] = field(default_factory=dict)
    # ca_output_ablation_handles: Dict[int, RemovableHandle] = field(default_factory=dict)

    mlp_ablation_positions: list[int] = field(default_factory=list)
    mlp_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # Attribution tracking: {layer_idx: [head_idxs]}
    # individual self-attention heads
    sa_head_attribution_positions: dict[int, list[int]] = field(default_factory=dict)
    sa_head_attribution_outputs: dict[int, dict[int, list[torch.Tensor]]] = field(default_factory=dict)
    sa_head_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # output of the whole self-attention layer
    sa_attribution_positions: list[int] = field(default_factory=list)
    sa_attribution_inputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    sa_attribution_outputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    sa_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # individual cross-attention heads
    ca_head_attribution_positions: dict[int, list[int]] = field(default_factory=dict)
    ca_head_attribution_outputs: dict[int, dict[int, list[torch.Tensor]]] = field(default_factory=dict)
    ca_head_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # idividual value states
    ca_v_attribution_positions: dict[int, list[int]] = field(default_factory=dict)
    ca_v_attribution_outputs: dict[int, dict[int, list[torch.Tensor]]] = field(default_factory=dict)
    ca_v_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # output of the whole cross-attention layer
    ca_attribution_positions: list[int] = field(default_factory=list)
    ca_attribution_inputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    ca_attribution_outputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    ca_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # MLP attribution
    mlp_attribution_positions: list[int] = field(default_factory=list)
    mlp_attribution_inputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    mlp_attribution_outputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    mlp_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # Pivot head manipulation (custom ablate + boost): {layer_idx: {head_idx: multiplier}}
    sa_head_pivot_manipulation_positions: dict[int, dict[int, float]] = field(default_factory=dict)
    sa_head_pivot_manipulation_handles: dict[int, RemovableHandle] = field(default_factory=dict)
    ca_head_pivot_manipulation_positions: dict[int, dict[int, float]] = field(default_factory=dict)
    ca_head_pivot_manipulation_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # Residual stream reading
    read_stream_inputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    read_stream_outputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    read_stream_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # Custom logit attribution
    custom_attribution_outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    custom_attribution_handles: dict[str, RemovableHandle] = field(default_factory=dict)

    def __post_init__(self):
        self.num_heads = self.model.model.config.num_heads
        self.num_layers = self.model.model.config.num_decoder_layers
        self.is_decoder_only = False

    def set_to_eval_mode(self) -> None:
        """Set the model to evaluation mode."""
        self.model.eval()

    def reset_attribution_inputs_and_outputs(self) -> None:
        """Reset attribution outputs."""
        self.sa_attribution_inputs.clear()
        self.ca_attribution_inputs.clear()
        self.mlp_attribution_inputs.clear()
        self.read_stream_inputs.clear()
        if hasattr(self, "custom_attribution_inputs"):
            self.custom_attribution_inputs.clear()
        self.sa_head_attribution_outputs.clear()
        self.sa_attribution_outputs.clear()
        self.ca_head_attribution_outputs.clear()
        self.ca_attribution_outputs.clear()
        self.mlp_attribution_outputs.clear()
        self.read_stream_outputs.clear()
        if hasattr(self, "custom_attribution_outputs"):
            self.custom_attribution_outputs.clear()

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
            for attention_type in ["ca", "sa"]:
                self.add_head_ablation_hooks(heads_to_ablate, attention_type=attention_type, ablation_method="zero")  # type: ignore
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
            for attention_type in ["ca", "sa"]:
                self.add_head_ablation_hooks(
                    heads_to_ablate,
                    attention_type=attention_type,
                    ablation_method="zero",
                )
        if "mlp" in ablations_types:
            self.add_mlp_ablation_hooks(layers_to_ablate)

    def _ablate_head_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        head_indices: list[int],
        ablation_method: Literal["zero", "mean"],
    ) -> Any:
        """Hook function to ablate attention head outputs."""
        if not isinstance(input, tuple) and len(input) == 1:
            raise ValueError("Expected output to be a tuple of length 1 for T5 models")

        # (batch, seq_len, d_model)
        b, _, d = input[0].shape
        n = self.model.model.config.num_heads
        d_kv = self.model.model.config.d_kv

        # Reshape to (batch, n_heads, seq_len, d_kv)
        reshaped = input[0].view(b, -1, n, d_kv).transpose(1, 2).contiguous()

        for head_idx in head_indices:
            if ablation_method == "zero":
                reshaped[:, head_idx, :, :] = 0.0
            elif ablation_method == "mean":
                reshaped[:, head_idx, :, :] = reshaped[:, head_idx, :, :].mean(dim=-1, keepdim=True)
            else:
                raise ValueError(f"Invalid ablation method: {ablation_method}")

        # Reshape to (batch, seq_len, d_model)
        modified_output = module.forward(reshaped.reshape_as(input[0]))
        return modified_output

    def _ablate_mlp_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        ablation_method: Literal["zero", "mean"],
    ) -> Any:
        """Hook function to ablate MLP outputs."""
        if ablation_method == "zero":
            return torch.zeros_like(output)
        return output.mean(dim=-1, keepdim=True)

    def _ablate_and_scale_pivot_heads_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        pivot_multipliers: dict[int, float],
    ) -> Any:
        """Hook that zeroes all non-pivot heads and scales pivot heads by multipliers.

        This hook is attached to the attention output projection `o` module. The
        input[0] to this module has shape (batch, seq_len, d_model). We reshape to
        (batch, n_heads, seq_len, d_kv), manipulate per-head activations, then pass
        the modified tensor back through the `o` projection.
        """
        if not isinstance(input, tuple):
            raise ValueError("Expected input to be a tuple for T5 models")

        hidden_states = input[0]
        if not isinstance(hidden_states, torch.Tensor):
            raise ValueError("Expected input[0] to be a tensor for T5 models")

        bsz, _, _ = hidden_states.shape
        num_heads = self.model.model.config.num_heads  # type: ignore
        d_kv = self.model.model.config.d_kv  # type: ignore

        # (b, s, d_model) -> (b, n, s, d_kv)
        reshaped = hidden_states.view(bsz, -1, num_heads, d_kv).transpose(1, 2).contiguous()

        # Apply zeroing for non-pivot heads and scaling for pivot heads
        for head_idx in range(num_heads):
            if head_idx in pivot_multipliers:
                multiplier = pivot_multipliers[head_idx]
                reshaped[:, head_idx, :, :] = reshaped[:, head_idx, :, :] * multiplier
            else:
                reshaped[:, head_idx, :, :] = 0.0

        # Back to (b, s, d_model), then apply the output projection
        modified_input = reshaped.transpose(1, 2).contiguous().view_as(hidden_states)
        modified_output = module.forward(modified_input)
        return modified_output

    def _attribute_head_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        head_indices: list[int],
        layer_idx: int,
        attention_type: Literal["sa", "ca"],
    ) -> None:
        """Hook function to attribute attention head outputs."""
        if not isinstance(input, tuple):
            raise ValueError("Expected input to be a tuple for T5 models")

        # (batch, seq_len, d_model)
        b, _, d = input[0].shape
        n = self.model.model.config.num_heads
        d_kv = self.model.model.config.d_kv

        # Reshape to (batch, n_heads, seq_len, d_kv)
        reshaped = input[0].view(b, -1, n, d_kv).transpose(1, 2).contiguous()

        outputs_dict = getattr(self, f"{attention_type}_head_attribution_outputs")
        if layer_idx not in outputs_dict:
            outputs_dict[layer_idx] = {}

        for head_idx in head_indices:
            if head_idx not in outputs_dict[layer_idx]:
                outputs_dict[layer_idx][head_idx] = []

            # shape: (batch, seq_len, d_kv)
            head_hidden_state = reshaped[:, head_idx, :, :]

            # module.weight: shape (d_model, n_heads*d_kv)
            # module.weight.T[head_idx * d_kv : (head_idx + 1) * d_kv, :] shape: (d_kv, d_model)
            head_output = (
                head_hidden_state @ module.weight.T[head_idx * d_kv : (head_idx + 1) * d_kv, :]  # type: ignore
            )  # shape: (batch, seq_len, d_model)
            outputs_dict[layer_idx][head_idx].append(head_output)

    def _attribute_v_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        head_indices: list[int],
        layer_idx: int,
        o_module: torch.nn.Module,
        attention_type: Literal["sa", "ca"],
    ) -> None:
        """Hook function to attribute attention head outputs."""
        if not isinstance(output, torch.Tensor):
            raise ValueError("Expected input to be a tensor for T5 models")

        # (batch, seq_len, d_model)
        b, s, d = output.shape
        n = self.model.model.config.num_heads
        d_kv = self.model.model.config.d_kv

        # Reshape to (batch, n_heads, seq_len, d_kv)
        reshaped = output.view(b, -1, n, d_kv).transpose(1, 2).contiguous()

        outputs_dict = getattr(self, f"{attention_type}_v_attribution_outputs")
        if layer_idx not in outputs_dict:
            outputs_dict[layer_idx] = {}

        for head_idx in head_indices:
            if head_idx not in outputs_dict[layer_idx]:
                outputs_dict[layer_idx][head_idx] = []

            # shape: (batch, seq_len, d_kv)
            head_hidden_state = reshaped[:, head_idx, :, :]

            # module.weight: shape (d_model, n_heads*d_kv)
            # module.weight.T[head_idx * d_kv : (head_idx + 1) * d_kv, :] shape: (d_kv, d_model)
            head_output = (
                head_hidden_state @ o_module.weight.T[head_idx * d_kv : (head_idx + 1) * d_kv, :]  # type: ignore
            )  # shape: (batch, seq_len, d_model)
            outputs_dict[layer_idx][head_idx].append(head_output)

    def _attribute_attn_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        layer_idx: int,
        attention_type: Literal["sa", "ca"],
    ) -> None:
        """Hook function to attribute attention head outputs, which are the outputs of the attention layer right before they get added to the residual stream"""
        if not isinstance(input, tuple):
            raise ValueError("Expected input to be a tuple for T5 models")
        if not isinstance(output, tuple):
            raise ValueError("Expected output to be a tuple for T5 models")

        output_dict = getattr(self, f"{attention_type}_attribution_outputs")
        input_dict = getattr(self, f"{attention_type}_attribution_inputs")
        if layer_idx not in output_dict:
            output_dict[layer_idx] = []

        if layer_idx not in input_dict:
            input_dict[layer_idx] = []

        input_dict[layer_idx].append(input[0])
        output_dict[layer_idx].append(output[0])

    def _attribute_mlp_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        layer_idx: int,
    ) -> None:
        """Hook function to attribute MLP outputs."""
        if not isinstance(input, tuple):
            raise ValueError("Expected input to be a tuple for T5 models")

        # input[0]: (batch, seq_len, d_model)
        # output: (batch, seq_len, d_model)
        input_dict = getattr(self, "mlp_attribution_inputs")
        output_dict = getattr(self, "mlp_attribution_outputs")
        if layer_idx not in output_dict:
            output_dict[layer_idx] = []

        if layer_idx not in input_dict:
            input_dict[layer_idx] = []

        input_dict[layer_idx].append(input[0])  # (batch, seq_len, d_model)
        output_dict[layer_idx].append(output)  # (batch, seq_len, d_model)

    def _read_stream_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        layer_idx: int,
    ) -> None:
        """Read and store the residual stream at the specified layer."""
        if not isinstance(input, tuple):
            raise ValueError("Expected input to be a tuple for T5 models")
        if not isinstance(output, tuple):
            raise ValueError("Expected output to be a tuple for T5 models")

        # input[0]: (batch, seq_len, d_model)
        # output[0]: (batch, seq_len, d_model)
        if layer_idx < 0:
            layer_idx = len(self.model.model.decoder.block) + layer_idx  # type: ignore

        if layer_idx not in self.read_stream_inputs:
            self.read_stream_inputs[layer_idx] = []
        # input[0] shape: (batch, seq_len, d_model)
        self.read_stream_inputs[layer_idx].append(input[0])

        if layer_idx not in self.read_stream_outputs:
            self.read_stream_outputs[layer_idx] = []
        # output[0] shape: (batch, seq_len, d_model)
        self.read_stream_outputs[layer_idx].append(output[0])

    def _logit_attribution_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        key: str,
    ) -> None:
        """Extract and store residual stream for logit attribution analysis."""
        resid = output[0] if isinstance(output, tuple) else output
        expected_d_model = self.model.model.config.d_model

        if resid.shape[-1] != expected_d_model:
            raise ValueError(f"Expected d_model dimension {expected_d_model}, got {resid.shape[-1]}")

        self.custom_attribution_outputs[key] = resid

    def _get_attention_layer(self, layer_idx: int, attention_type: Literal["sa", "ca"]) -> torch.nn.Module:
        """
        NOTE: Specific to Chronos and Chronos-Bolt CircuitLens
        Get the appropriate attention layer based on type.

        Args:
            layer_idx: Index of the layer to get
            attention_type: Type of attention layer ("sa" for self-attention or "ca" for cross-attention)
        """
        t5_model = self.model.model  # type: ignore
        if attention_type == "sa":
            return t5_model.decoder.block[layer_idx].layer[0].SelfAttention  # type: ignore
        return t5_model.decoder.block[layer_idx].layer[1].EncDecAttention  # type: ignore

    def _get_attention_output_layer(self, layer_idx: int, attention_type: Literal["sa", "ca"]) -> torch.nn.Module:
        """
        NOTE: Specific to Chronos and Chronos-Bolt CircuitLens
        Get the appropriate attention output layer based on type.

        Args:
            layer_idx: Index of the layer to get
            attention_type: Type of attention layer ("sa" for self-attention or "ca" for cross-attention)
        """
        t5_model = self.model.model  # type: ignore
        if attention_type == "sa":
            return t5_model.decoder.block[layer_idx].layer[0].SelfAttention.o  # type: ignore
        return t5_model.decoder.block[layer_idx].layer[1].EncDecAttention.o  # type: ignore

    def _get_attention_value_layer(self, layer_idx: int, attention_type: Literal["sa", "ca"]) -> torch.nn.Module:
        """
        NOTE: Specific to Chronos and Chronos-Bolt CircuitLens
        Get the appropriate attention output layer based on type.

        Args:
            layer_idx: Index of the layer to get
            attention_type: Type of attention layer ("sa" for self-attention or "ca" for cross-attention)
        """
        t5_model = self.model.model  # type: ignore
        if attention_type == "sa":
            return t5_model.decoder.block[layer_idx].layer[0].SelfAttention.v  # type: ignore
        return t5_model.decoder.block[layer_idx].layer[1].EncDecAttention.v  # type: ignore

    def _get_attention_module(self, layer_idx: int, attention_type: Literal["sa", "ca"]) -> torch.nn.Module:
        """
        NOTE: Specific to Chronos and Chronos-Bolt CircuitLens
        Get the appropriate attention module based on type. Module is one abstraction above layer

        Args:
            layer_idx: Index of the layer to get
            attention_type: Type of attention layer ("sa" for self-attention or "ca" for cross-attention)
        """
        t5_model = self.model.model  # type: ignore
        if attention_type == "sa":
            return t5_model.decoder.block[layer_idx].layer[0]  # type: ignore
        return t5_model.decoder.block[layer_idx].layer[1]  # type: ignore

    def add_head_ablation_hooks(
        self,
        heads_to_ablate: list[tuple[int, int]],
        ablation_method: Literal["zero", "mean"] = "zero",
        attention_type: Literal["sa", "ca"] = "ca",
        stack_type: Literal["encoder", "decoder"] = "decoder",
    ) -> dict[int, RemovableHandle]:
        """Add hooks to ablate attention heads during model generation."""
        if ablation_method not in ["zero", "mean"]:
            raise ValueError(f"Invalid ablation method: {ablation_method}")

        # Group heads by layer
        heads_by_layer: dict[int, list[int]] = {}
        for layer_idx, head_idx in heads_to_ablate:
            heads_by_layer.setdefault(layer_idx, []).append(head_idx)
            getattr(self, f"{attention_type}_head_ablation_positions").setdefault(layer_idx, {})[head_idx] = (
                ablation_method
            )

        # Create hooks for each layer
        handles = getattr(self, f"{attention_type}_head_ablation_handles")
        for layer_idx, head_indices in heads_by_layer.items():
            target_layer = self._get_attention_output_layer(layer_idx, attention_type)
            hook = target_layer.register_forward_hook(
                partial(
                    self._ablate_head_hook_fn,
                    head_indices=head_indices,
                    ablation_method=ablation_method,
                )
            )
            handles[layer_idx] = hook

        return handles

    def add_pivot_head_manipulation_hook(
        self,
        layer_to_pivots: dict[int, list[tuple[int, float]]],
        attention_type: Literal["sa", "ca"],
    ) -> dict[int, RemovableHandle]:
        """Add hooks to ablate all non-pivot heads and scale specified pivot heads.

        Pass a mapping of decoder layer indices to their pivot head multipliers, e.g.:
            {layer_idx: [(head_idx, multiplier), ...], ...}

        Args:
            layer_to_pivots: Mapping of layer idx to list of (head_idx, multiplier). Must be non-empty and each list non-empty.
            attention_type: "sa" for self-attention, "ca" for cross-attention.

        Returns:
            The dict of removable handles for this attention type keyed by layer_idx.
        """
        if not isinstance(layer_to_pivots, dict) or not layer_to_pivots:
            raise ValueError("layer_to_pivots must be a non-empty dict of {layer_idx: [(head_idx, multiplier), ...]}")

        # Validate and register per-layer hooks
        num_heads = self.model.model.config.num_heads  # type: ignore
        pos_attr = f"{attention_type}_head_pivot_manipulation_positions"
        handle_attr = f"{attention_type}_head_pivot_manipulation_handles"
        handles = getattr(self, handle_attr)

        for layer_idx, pivots in layer_to_pivots.items():
            if not pivots:
                raise ValueError(f"Pivot list for layer {layer_idx} must be non-empty")

            # Validate heads are in range
            for head_idx, _ in pivots:
                if head_idx < 0 or head_idx >= num_heads:
                    raise ValueError(f"Head index {head_idx} out of range [0, {num_heads}) for layer {layer_idx}")

            # Build mapping and store positions
            pivot_multipliers: dict[int, float] = {int(h): float(m) for h, m in pivots}
            getattr(self, pos_attr)[layer_idx] = pivot_multipliers

            # Register/replace hook on the attention output projection `o`
            target_layer = self._get_attention_output_layer(layer_idx, attention_type)
            hook = target_layer.register_forward_hook(
                partial(
                    self._ablate_and_scale_pivot_heads_hook_fn,
                    pivot_multipliers=pivot_multipliers,
                )
            )
            handles[layer_idx] = hook

        return handles

    def compute_groups_pivots_and_scalings(
        self,
        similarity: torch.Tensor,
        magnitudes: torch.Tensor,
        epsilon: float,
    ) -> tuple[list[list[int]], list[tuple[int, float]]]:
        """Compute groups, pivot heads, and their scalings from similarity and magnitudes.

        Grouping:
        - Start from the highest-magnitude ungrouped head.
        - Add any remaining heads whose absolute similarity to the seed head is >= epsilon.
        - Repeat until all heads are grouped.

        Pivot per group: head with highest sum of absolute similarities to other heads in the group.
        Scaling for pivot p: sum_{h in group} (mag[h]/mag[p]) * sign(sim[p,h]). If mag[p]==0, scaling=0.
        """
        if similarity.dim() != 2 or similarity.size(0) != similarity.size(1):
            raise ValueError("similarity must be square (n_heads x n_heads)")
        if magnitudes.dim() != 1 or magnitudes.size(0) != similarity.size(0):
            raise ValueError("magnitudes must be 1D with length n_heads matching similarity")

        n_heads = similarity.size(0)
        sim = similarity.detach().to("cpu")
        mags = magnitudes.detach().to("cpu")

        # Indices sorted by magnitude, descending
        sorted_heads = torch.argsort(mags, descending=True).tolist()
        ungrouped: set[int] = set(range(n_heads))
        groups: list[list[int]] = []

        for seed in sorted_heads:
            if seed not in ungrouped:
                continue
            group: list[int] = []
            for h in list(ungrouped):
                if torch.abs(sim[seed, h]).item() >= float(epsilon):
                    group.append(h)
            if seed not in group:
                group.append(seed)
            for h in group:
                ungrouped.discard(h)
            groups.append(sorted(group))

        pivot_scalings: list[tuple[int, float]] = []
        tiny = 1e-12
        for group in groups:
            # Choose pivot by maximum sum of absolute similarities within group
            best_pivot = None
            best_score = float("-inf")
            for cand in group:
                score = float(torch.sum(torch.abs(sim[cand, group])))
                if score > best_score:
                    best_score = score
                    best_pivot = cand
            assert best_pivot is not None

            p = int(best_pivot)
            p_mag = float(mags[p].item())
            if abs(p_mag) <= tiny:
                scaling = 0.0
            else:
                total = 0.0
                for h in group:
                    sgn = 1.0 if float(sim[p, h].item()) >= 0.0 else -1.0
                    total += float(mags[h].item()) / p_mag * sgn
                scaling = float(total)

            pivot_scalings.append((p, scaling))

        return groups, pivot_scalings

    def apply_pivot_head_manipulation_from_similarity(
        self,
        layer_idx: int,
        attention_type: Literal["sa", "ca"],
        similarity: torch.Tensor,
        magnitudes: torch.Tensor,
        epsilon: float,
    ) -> dict[int, RemovableHandle]:
        """Compute groups/pivots/scalings and register pivot manipulation for a layer.

        Validates input shapes against the model's configured number of heads and then
        uses `add_pivot_head_manipulation_hook` to install the hook for the layer.
        """
        num_heads = int(self.model.model.config.num_heads)  # type: ignore
        if similarity.shape != (num_heads, num_heads):
            raise ValueError(f"similarity must be of shape ({num_heads}, {num_heads})")
        if magnitudes.numel() != num_heads:
            raise ValueError(f"magnitudes must have length {num_heads}")

        _, pivot_scalings = self.compute_groups_pivots_and_scalings(similarity, magnitudes, epsilon)
        return self.add_pivot_head_manipulation_hook({layer_idx: pivot_scalings}, attention_type)

    def add_mlp_ablation_hooks(
        self,
        mlp_to_ablate: list[int],
        ablation_method: Literal["zero", "mean"] = "zero",
        stack_type: Literal["encoder", "decoder"] = "decoder",
    ) -> dict[int, RemovableHandle]:
        """Add hooks to ablate MLP outputs."""
        if ablation_method not in ["zero", "mean"]:
            raise ValueError(f"Invalid ablation method: {ablation_method}")

        t5_model = self.model.model  # type: ignore
        for layer_idx in mlp_to_ablate:
            target_layer = t5_model.decoder.block[layer_idx].layer[2].DenseReluDense  # type: ignore
            hook_fn = partial(self._ablate_mlp_hook_fn, ablation_method=ablation_method)
            hook = target_layer.register_forward_hook(hook_fn)
            self.mlp_ablation_handles[layer_idx] = hook
            self.mlp_ablation_positions.append(layer_idx)

        return self.mlp_ablation_handles

    def add_head_attribution_hooks(
        self,
        heads_to_attribute: list[tuple[int, int]],
        attention_type: Literal["sa", "ca"] = "ca",
        stack_type: Literal["encoder", "decoder"] = "decoder",
    ) -> dict[int, RemovableHandle]:
        """Add hooks to attribute attention head outputs."""
        # Group heads by layer
        heads_by_layer: dict[int, list[int]] = {}
        for layer_idx, head_idx in heads_to_attribute:
            heads_by_layer.setdefault(layer_idx, []).append(head_idx)
            getattr(self, f"{attention_type}_head_attribution_positions").setdefault(layer_idx, []).append(head_idx)

        # Create hooks for each layer
        handles = getattr(self, f"{attention_type}_head_attribution_handles")
        for layer_idx, head_indices in heads_by_layer.items():
            target_layer = self._get_attention_output_layer(layer_idx, attention_type)
            hook = target_layer.register_forward_hook(
                partial(
                    self._attribute_head_hook_fn,
                    head_indices=head_indices,
                    layer_idx=layer_idx,
                    attention_type=attention_type,
                )
            )
            handles[layer_idx] = hook

        return handles

    def add_v_attribution_hooks(
        self,
        heads_to_attribute: list[tuple[int, int]],
        attention_type: Literal["sa", "ca"] = "ca",
        stack_type: Literal["encoder", "decoder"] = "decoder",
    ) -> dict[int, RemovableHandle]:
        """Add hooks to attribute attention head outputs."""
        # Group heads by layer
        heads_by_layer: dict[int, list[int]] = {}
        for layer_idx, head_idx in heads_to_attribute:
            heads_by_layer.setdefault(layer_idx, []).append(head_idx)
            getattr(self, f"{attention_type}_v_attribution_positions").setdefault(layer_idx, []).append(head_idx)

        # Create hooks for each layer
        handles = getattr(self, f"{attention_type}_v_attribution_handles")
        for layer_idx, head_indices in heads_by_layer.items():
            target_layer = self._get_attention_value_layer(layer_idx, attention_type)
            o_layer = self._get_attention_output_layer(layer_idx, attention_type)
            hook = target_layer.register_forward_hook(
                partial(
                    self._attribute_v_hook_fn,
                    head_indices=head_indices,
                    layer_idx=layer_idx,
                    o_module=o_layer,
                    attention_type=attention_type,
                )
            )
            handles[layer_idx] = hook

        return handles

    def add_attn_attribution_hooks(
        self,
        layer_idxs: list[int],
        attention_type: Literal["sa", "ca"] = "ca",
        stack_type: Literal["encoder", "decoder"] = "decoder",
    ) -> dict[int, RemovableHandle]:
        """Add hooks to attribute attention head outputs for a given layer."""
        if attention_type not in ["sa", "ca"]:
            raise ValueError(f"Invalid attention type: {attention_type}")

        getattr(self, f"{attention_type}_attribution_positions").extend(layer_idxs)

        handles = getattr(self, f"{attention_type}_attribution_handles")
        for layer_idx in layer_idxs:
            target_layer = self._get_attention_module(layer_idx, attention_type)
            hook = target_layer.register_forward_hook(
                partial(
                    self._attribute_attn_hook_fn,
                    layer_idx=layer_idx,
                    attention_type=attention_type,
                )
            )
            handles[layer_idx] = hook

        return handles

    def add_mlp_attribution_hooks(
        self,
        mlp_to_attribute: list[int],
        stack_type: Literal["encoder", "decoder"] = "decoder",
    ) -> dict[int, RemovableHandle]:
        """Add hooks to attribute MLP outputs."""
        t5_model = self.model.model  # type: ignore
        for layer_idx in mlp_to_attribute:
            target_layer = t5_model.decoder.block[layer_idx].layer[2]  # type: ignore
            hook = target_layer.register_forward_hook(partial(self._attribute_mlp_hook_fn, layer_idx=layer_idx))
            self.mlp_attribution_positions.append(layer_idx)
            self.mlp_attribution_handles[layer_idx] = hook

        return self.mlp_attribution_handles

    def add_read_stream_hooks(
        self,
        layer_idxs: list[int],
        stack_type: Literal["encoder", "decoder"] = "decoder",
    ) -> dict[int, RemovableHandle]:
        """Add hooks to read residual streams at specified layers."""
        t5_model = self.model.model  # type: ignore
        for layer_idx in layer_idxs:
            # NOTE: we are explicit about typing this target_layer to avoid type errors
            target_layer: torch.nn.Module = t5_model.decoder.block[layer_idx]  # type: ignore
            hook = target_layer.register_forward_hook(
                partial(
                    self._read_stream_hook_fn,
                    layer_idx=layer_idx,
                )
            )
            self.read_stream_handles[layer_idx] = hook

        return self.read_stream_handles

    def add_logit_attribution_hooks(self, target_layer: torch.nn.Module, key: str) -> dict[str, RemovableHandle]:
        """Add a hook for logit attribution analysis."""
        hook = target_layer.register_forward_hook(
            partial(
                self._logit_attribution_hook_fn,
                key=key,
            )
        )
        self.custom_attribution_handles[key] = hook
        return self.custom_attribution_handles

    def remove_all_hooks(self) -> None:
        """Remove all hooks and clear related state."""
        attention_types = ["sa", "ca"]

        # Remove all hook types
        for hook_type in ["ablation", "attribution"]:
            for attn_type in attention_types:
                handles = getattr(self, f"{attn_type}_head_{hook_type}_handles")
                for hook in handles.values():
                    hook.remove()
                handles.clear()
                getattr(self, f"{attn_type}_head_{hook_type}_positions").clear()

                if hook_type == "attribution":
                    handles = getattr(self, f"{attn_type}_{hook_type}_handles")
                    for hook in handles.values():
                        hook.remove()
                    handles.clear()
                    getattr(self, f"{attn_type}_{hook_type}_positions").clear()

        # Remove pivot head manipulation hooks
        for attn_type in attention_types:
            handles = getattr(self, f"{attn_type}_head_pivot_manipulation_handles")
            for hook in handles.values():
                hook.remove()
            handles.clear()
            getattr(self, f"{attn_type}_head_pivot_manipulation_positions").clear()

        # Remove MLP hooks
        for hook in self.mlp_ablation_handles.values():
            hook.remove()
        for hook in self.mlp_attribution_handles.values():
            hook.remove()
        self.mlp_ablation_handles.clear()
        self.mlp_attribution_handles.clear()
        self.mlp_ablation_positions.clear()
        self.mlp_attribution_positions.clear()

        # Remove stream and custom hooks
        for hook in self.read_stream_handles.values():
            hook.remove()
        for hook in self.custom_attribution_handles.values():
            hook.remove()

        # Clear outputs
        self.read_stream_handles.clear()
        self.custom_attribution_handles.clear()
        self.read_stream_outputs.clear()
        self.custom_attribution_outputs.clear()
        self.mlp_attribution_outputs.clear()
        for attn_type in attention_types:
            getattr(self, f"{attn_type}_head_attribution_outputs").clear()

    def remove_pivot_head_manipulation_hooks(self, attention_type: Literal["sa", "ca"] = "ca") -> None:
        """Remove pivot head manipulation hooks and clear related state for the given attention type."""
        handles = getattr(self, f"{attention_type}_head_pivot_manipulation_handles")
        for hook in handles.values():
            hook.remove()
        handles.clear()
        getattr(self, f"{attention_type}_head_pivot_manipulation_positions").clear()

    def remove_head_ablation_hooks(self, attention_type: Literal["sa", "ca"] = "ca") -> None:
        """Remove head ablation hooks and clear related state."""
        handles = getattr(self, f"{attention_type}_head_ablation_handles")
        for hook in handles.values():
            hook.remove()
        handles.clear()
        getattr(self, f"{attention_type}_head_ablation_positions").clear()

    def remove_head_attribution_hooks(self, attention_type: Literal["sa", "ca"] = "ca") -> None:
        """Remove head attribution hooks and clear related state."""
        handles = getattr(self, f"{attention_type}_head_attribution_handles")
        for hook in handles.values():
            hook.remove()
        handles.clear()
        getattr(self, f"{attention_type}_head_attribution_positions").clear()
        getattr(self, f"{attention_type}_head_attribution_outputs").clear()

    def remove_attn_attribution_hooks(self, attention_type: Literal["sa", "ca"] = "ca") -> None:
        """Remove output head attribution hooks and clear related state."""
        handles = getattr(self, f"{attention_type}_attribution_handles")
        for hook in handles.values():
            hook.remove()
        handles.clear()
        getattr(self, f"{attention_type}_attribution_positions").clear()
        getattr(self, f"{attention_type}_attribution_inputs").clear()
        getattr(self, f"{attention_type}_attribution_outputs").clear()

    def remove_mlp_attribution_hooks(self) -> None:
        """Remove MLP attribution hooks and clear related state."""
        for hook in self.mlp_attribution_handles.values():
            hook.remove()
        self.mlp_attribution_handles.clear()
        self.mlp_attribution_inputs.clear()
        self.mlp_attribution_outputs.clear()
        self.mlp_attribution_positions.clear()

    def remove_read_stream_hooks(self) -> None:
        """Remove read stream hooks and clear related state."""
        for hook in self.read_stream_handles.values():
            hook.remove()
        self.read_stream_handles.clear()
        self.read_stream_inputs.clear()
        self.read_stream_outputs.clear()

    def remove_custom_attribution_hooks(self) -> None:
        """Remove custom attribution hooks and clear related state."""
        for hook in self.custom_attribution_handles.values():
            hook.remove()
        self.custom_attribution_handles.clear()
        self.custom_attribution_inputs.clear()
        self.custom_attribution_outputs.clear()

    def unembed_residual(self, residual: torch.Tensor) -> torch.Tensor:
        """Unembed residual stream by applying final layer norm and language model head."""
        t5_model: torch.nn.Module = self.model.model  # type: ignore

        if (
            not hasattr(t5_model, "decoder")
            or not hasattr(t5_model.decoder, "final_layer_norm")
            or not callable(getattr(t5_model.decoder, "final_layer_norm", None))
        ):
            raise AttributeError("Model does not have expected T5 decoder structure")
        if not hasattr(t5_model, "lm_head") or not callable(getattr(t5_model, "lm_head", None)):
            raise AttributeError("Model does not have expected lm_head")

        residual = t5_model.decoder.final_layer_norm(residual)  # type: ignore
        return t5_model.lm_head(residual * t5_model.config.d_model**-0.5)  # type: ignore

    def sample_tokens(self, residual: torch.Tensor, do_sample: bool = False) -> torch.Tensor:
        """Sample tokens from residual stream using greedy decoding."""
        if do_sample:
            raise NotImplementedError("Sampling not yet implemented")

        logits = self.unembed_residual(residual)
        return torch.argmax(logits, dim=-1)
