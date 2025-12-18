from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.hooks import RemovableHandle

from tsfm_lens.chronos_bolt.pipeline import ChronosBoltPipelineCustom
from tsfm_lens.circuitlens import BaseCircuitLens


@dataclass
class CircuitLensBolt(ChronosBoltPipelineCustom, BaseCircuitLens):
    """
    A class for performing circuit lens analysis on ChronosBolt models.

    CircuitLensBolt supports attention head ablation, residual stream reading, and logit attribution analysis
    to understand model behavior and identify important computational sub-paths in ChronosBolt architecture.

    Key Features:
        - Attention Head Ablation: Zero out or replace attention head outputs with their mean
        - Residual Stream Reading: Extract hidden states at any layer for analysis
        - Logit Attribution: Track how specific components contribute to final predictions
        - MLP Ablation: Ablate MLP layers to understand their role in computation
        - Hook Management: Automatic cleanup of all registered hooks
        - Encoder-Decoder Support: Works with both encoder and decoder layers

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

    Example Usage:
        TODO

    Note:
        This class is designed specifically for ChronosBolt models and assumes the standard T5 architecture
        with encoder and decoder blocks containing self-attention, cross-attention, and MLP layers.
        The model uses patching with patch size and stride of 16.
    """

    # Ablation tracking: {layer_idx: {head_idx: ablation_method}}
    sa_head_ablation_positions: dict[int, dict[int, str]] = field(default_factory=dict)
    sa_head_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    ca_head_ablation_positions: dict[int, dict[int, str]] = field(default_factory=dict)
    ca_head_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    mlp_ablation_positions: list[int] = field(default_factory=list)
    mlp_ablation_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # Attribution tracking: {layer_idx: [head_idxs]}
    # individual self-attention heads
    sa_head_attribution_positions: dict[int, list[int]] = field(default_factory=dict)
    sa_head_attribution_outputs: dict[int, dict[int, list[torch.Tensor]]] = field(default_factory=dict)
    sa_head_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # output of the whole self-attention layer
    sa_attribution_inputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    sa_attribution_outputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    sa_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # individual cross-attention heads
    ca_head_attribution_positions: dict[int, list[int]] = field(default_factory=dict)
    ca_head_attribution_outputs: dict[int, dict[int, list[torch.Tensor]]] = field(default_factory=dict)
    ca_head_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # output of the whole cross-attention layer
    ca_attribution_inputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    ca_attribution_outputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    ca_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # MLP attribution
    mlp_attribution_positions: list[int] = field(default_factory=list)
    mlp_attribution_inputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    mlp_attribution_outputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    mlp_attribution_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # Residual stream reading
    read_stream_inputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    read_stream_outputs: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    read_stream_handles: dict[int, RemovableHandle] = field(default_factory=dict)

    # Custom logit attribution
    custom_attribution_outputs: dict[str, torch.Tensor] = field(default_factory=dict)
    custom_attribution_handles: dict[str, RemovableHandle] = field(default_factory=dict)

    def __post_init__(self):
        self.num_heads = self.model.config.num_heads
        self.num_layers = self.model.config.num_decoder_layers
        # NOTE: don't set quantiles here because it is already set in the pipeline
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
        # NOTE: there are no custom attribution inputs yet
        if hasattr(self, "custom_attribution_inputs"):
            self.custom_attribution_inputs.clear()  # type: ignore[attr-defined]
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
                    attention_type=attention_type,  # type: ignore[arg-type]
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

        b, _, d = input[0].shape
        n = self.model.config.num_heads
        d_kv = self.model.config.d_kv

        hidden_states = output[0]

        # Reshape to (batch, seq_len, n_heads, head_size)
        # reshaped = einops.rearrange(hidden_states, "b s (n h) -> b s n h", n=module.n_heads)
        reshaped = input[0].view(b, -1, n, d_kv).transpose(1, 2).contiguous()

        for head_idx in head_indices:
            if ablation_method == "zero":
                reshaped[:, head_idx, :, :] = 0.0
            elif ablation_method == "mean":
                reshaped[:, head_idx, :, :] = reshaped[:, head_idx, :, :].mean(dim=-1, keepdim=True)
            else:
                raise ValueError(f"Invalid ablation method: {ablation_method}")

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

        hidden_states = input[0]
        b, _, d = input[0].shape
        n = self.model.config.num_heads
        d_kv = self.model.config.d_kv

        reshaped = input[0].view(b, -1, n, d_kv).transpose(1, 2).contiguous()

        outputs_dict = getattr(self, f"{attention_type}_head_attribution_outputs")
        if layer_idx not in outputs_dict:
            outputs_dict[layer_idx] = {}

        for head_idx in head_indices:
            if head_idx not in outputs_dict[layer_idx]:
                outputs_dict[layer_idx][head_idx] = []

            head_hidden_state = reshaped[:, head_idx, :, :]
            head_output = (
                head_hidden_state @ module.weight.T[head_idx * d_kv : (head_idx + 1) * d_kv, :]  # type: ignore
            )
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

        input_dict = getattr(self, "mlp_attribution_inputs")
        output_dict = getattr(self, "mlp_attribution_outputs")
        if layer_idx not in output_dict:
            output_dict[layer_idx] = []

        if layer_idx not in input_dict:
            input_dict[layer_idx] = []

        input_dict[layer_idx].append(input[0])
        output_dict[layer_idx].append(output)

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

        if layer_idx < 0:
            # For ChronosBolt, we need to handle both encoder and decoder layers
            if hasattr(module, "encoder") and hasattr(module, "decoder"):
                # This is the full model, determine which stack
                total_encoder_layers = len(self.model.encoder.block)
                total_decoder_layers = len(self.model.decoder.block)
                if abs(layer_idx) <= total_encoder_layers:
                    layer_idx = total_encoder_layers + layer_idx
                else:
                    layer_idx = total_decoder_layers + layer_idx
            else:
                # This is a specific stack (encoder or decoder)
                total_layers = len(module.block)  # type: ignore[attr-defined]
                layer_idx = total_layers + layer_idx

        if layer_idx not in self.read_stream_inputs:
            self.read_stream_inputs[layer_idx] = []
        self.read_stream_inputs[layer_idx].append(input[0])

        if layer_idx not in self.read_stream_outputs:
            self.read_stream_outputs[layer_idx] = []
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
        expected_d_model = self.model.config.d_model

        if resid.shape[-1] != expected_d_model:
            raise ValueError(f"Expected d_model dimension {expected_d_model}, got {resid.shape[-1]}")

        self.custom_attribution_outputs[key] = resid

    def _get_attention_layer(
        self,
        layer_idx: int,
        attention_type: Literal["sa", "ca"],
        stack_type: Literal["encoder", "decoder"] = "decoder",
    ) -> torch.nn.Module:
        """
        NOTE: Specific to Chronos and Chronos-Bolt CircuitLens
        Get the appropriate attention layer based on type and stack.

        Args:
            layer_idx: Index of the layer to get
            attention_type: Type of attention layer ("sa" for self-attention or "ca" for cross-attention)
            stack_type: Which stack to get the layer from ("encoder" or "decoder")

        Returns:
            The appropriate attention layer based on type and stack

        Raises:
            ValueError: If trying to get cross-attention from encoder
        """
        if stack_type == "encoder":
            t5_model = self.model.encoder
            if attention_type == "sa":
                return t5_model.block[layer_idx].layer[0].SelfAttention  # type: ignore[attr-defined]
            else:
                raise ValueError("Encoder only has self-attention, not cross-attention")
        else:  # decoder
            t5_model = self.model.decoder
            if attention_type == "sa":
                return t5_model.block[layer_idx].layer[0].SelfAttention  # type: ignore[attr-defined]
            else:  # ca
                return t5_model.block[layer_idx].layer[1].EncDecAttention  # type: ignore[attr-defined]

    def _get_attention_output_layer(
        self,
        layer_idx: int,
        attention_type: Literal["sa", "ca"],
        stack_type: Literal["encoder", "decoder"] = "decoder",
    ) -> torch.nn.Module:
        """
        NOTE: Specific to Chronos and Chronos-Bolt CircuitLens
        Get the appropriate attention output layer based on type and stack.

        Args:
            layer_idx: Index of the layer to get
            attention_type: Type of attention layer ("sa" for self-attention or "ca" for cross-attention)
            stack_type: Which stack to get the layer from ("encoder" or "decoder")

        Returns:
            The output projection layer of the specified attention layer

        Raises:
            ValueError: If trying to get cross-attention from encoder
        """
        if stack_type == "encoder":
            t5_model = self.model.encoder
            if attention_type == "sa":
                return t5_model.block[layer_idx].layer[0].SelfAttention.o  # type: ignore[attr-defined]
            else:
                raise ValueError("Encoder only has self-attention, not cross-attention")
        else:  # decoder
            t5_model = self.model.decoder
            if attention_type == "sa":
                return t5_model.block[layer_idx].layer[0].SelfAttention.o  # type: ignore[attr-defined]
            else:  # ca
                return t5_model.block[layer_idx].layer[1].EncDecAttention.o  # type: ignore[attr-defined]

    def _get_attention_module(
        self,
        layer_idx: int,
        attention_type: Literal["sa", "ca"],
        stack_type: Literal["encoder", "decoder"] = "decoder",
    ) -> torch.nn.Module:
        """
        NOTE: Specific to Chronos and Chronos-Bolt CircuitLens
        Get the appropriate attention module based on type and stack.

        Args:
            layer_idx: Index of the layer to get
            attention_type: Type of attention layer ("sa" for self-attention or "ca" for cross-attention)
            stack_type: Which stack to get the layer from ("encoder" or "decoder")

        Returns:
            The full attention module including layer norm and residual connections

        Raises:
            ValueError: If trying to get cross-attention from encoder
        """
        if stack_type == "encoder":
            t5_model = self.model.encoder
            if attention_type == "sa":
                return t5_model.block[layer_idx].layer[0]  # type: ignore[attr-defined]
            else:
                raise ValueError("Encoder only has self-attention, not cross-attention")
        else:  # decoder
            t5_model = self.model.decoder
            if attention_type == "sa":
                return t5_model.block[layer_idx].layer[0]  # type: ignore[attr-defined]
            else:  # ca
                return t5_model.block[layer_idx].layer[1]  # type: ignore[attr-defined]

    def add_head_ablation_hooks(
        self,
        heads_to_ablate: list[tuple[int, int]],
        ablation_method: Literal["zero", "mean"] = "zero",
        attention_type: Literal["sa", "ca"] = "ca",
        stack_type: Literal["encoder", "decoder"] = "decoder",
        verbose: bool = False,
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
        if verbose:
            print(f"---> heads_by_layer: {heads_by_layer}")
        # Create hooks for each layer
        handles = getattr(self, f"{attention_type}_head_ablation_handles")
        for layer_idx, head_indices in heads_by_layer.items():
            target_layer = self._get_attention_output_layer(layer_idx, attention_type, stack_type)
            if verbose:
                print(f"---> target_layer {layer_idx}: {target_layer}")
            hook = target_layer.register_forward_hook(
                partial(
                    self._ablate_head_hook_fn,
                    head_indices=head_indices,
                    ablation_method=ablation_method,
                )
            )
            handles[layer_idx] = hook

        return handles

    def add_mlp_ablation_hooks(
        self,
        mlp_to_ablate: list[int],
        ablation_method: Literal["zero", "mean"] = "zero",
        stack_type: Literal["encoder", "decoder"] = "decoder",
    ) -> dict[int, RemovableHandle]:
        """Add hooks to ablate MLP outputs."""
        if ablation_method not in ["zero", "mean"]:
            raise ValueError(f"Invalid ablation method: {ablation_method}")

        if stack_type == "encoder":
            t5_model = self.model.encoder
            for layer_idx in mlp_to_ablate:
                target_layer = t5_model.block[layer_idx].layer[1].DenseReluDense  # type: ignore[attr-defined]
                hook_fn = partial(self._ablate_mlp_hook_fn, ablation_method=ablation_method)
                hook = target_layer.register_forward_hook(hook_fn)
                self.mlp_ablation_handles[layer_idx] = hook
                self.mlp_ablation_positions.append(layer_idx)
        else:  # decoder
            t5_model = self.model.decoder
            for layer_idx in mlp_to_ablate:
                target_layer = t5_model.block[layer_idx].layer[2].DenseReluDense  # type: ignore[attr-defined]
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
            target_layer = self._get_attention_output_layer(layer_idx, attention_type, stack_type)
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

    def add_attn_attribution_hooks(
        self,
        layer_idxs: list[int],
        attention_type: Literal["sa", "ca"] = "ca",
        stack_type: Literal["encoder", "decoder"] = "decoder",
    ) -> dict[int, RemovableHandle]:
        """Add hooks to attribute attention head outputs for a given layer."""
        handles = getattr(self, f"{attention_type}_attribution_handles")
        for layer_idx in layer_idxs:
            target_layer = self._get_attention_module(layer_idx, attention_type, stack_type)
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
        if stack_type == "encoder":
            t5_model = self.model.encoder
            for layer_idx in mlp_to_attribute:
                target_layer = t5_model.block[layer_idx].layer[1]  # type: ignore[attr-defined]
                hook = target_layer.register_forward_hook(partial(self._attribute_mlp_hook_fn, layer_idx=layer_idx))  # type: ignore[attr-defined]
                self.mlp_attribution_positions.append(layer_idx)
                self.mlp_attribution_handles[layer_idx] = hook
        else:  # decoder
            t5_model = self.model.decoder
            for layer_idx in mlp_to_attribute:
                target_layer = t5_model.block[layer_idx].layer[2]  # type: ignore[attr-defined]
                hook = target_layer.register_forward_hook(partial(self._attribute_mlp_hook_fn, layer_idx=layer_idx))  # type: ignore[attr-defined]
                self.mlp_attribution_positions.append(layer_idx)
                self.mlp_attribution_handles[layer_idx] = hook

        return self.mlp_attribution_handles

    def add_read_stream_hooks(
        self,
        layer_idxs: list[int],
        stack_type: Literal["encoder", "decoder"] = "decoder",
    ) -> dict[int, RemovableHandle]:
        """Add hooks to read residual streams at specified layers."""
        if stack_type == "encoder":
            t5_model = self.model.encoder
            for layer_idx in layer_idxs:
                target_layer: torch.nn.Module = t5_model.block[layer_idx]  # type: ignore[attr-defined]
                hook = target_layer.register_forward_hook(
                    partial(
                        self._read_stream_hook_fn,
                        layer_idx=layer_idx,
                    )
                )
                self.read_stream_handles[layer_idx] = hook
        else:  # decoder
            t5_model = self.model.decoder
            for layer_idx in layer_idxs:
                target_layer: torch.nn.Module = t5_model.block[layer_idx]
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
                    handles = getattr(self, f"{attn_type}_attribution_handles")
                    for hook in handles.values():
                        hook.remove()
                    handles.clear()
                    getattr(self, f"{attn_type}_attribution_inputs").clear()
                    getattr(self, f"{attn_type}_attribution_outputs").clear()

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
        self.custom_attribution_outputs.clear()

    def unembed_residual(self, residual: torch.Tensor) -> torch.Tensor:
        """Unembed residual stream by applying final layer norm and output patch embedding."""
        # For ChronosBolt, we need to apply the output patch embedding instead of lm_head
        if not hasattr(self.model, "output_patch_embedding"):
            raise AttributeError("Model does not have expected output_patch_embedding")

        # Apply the output patch embedding to get the final predictions
        return self.model.output_patch_embedding(residual)

    def sample_tokens(self, residual: torch.Tensor, do_sample: bool = False) -> torch.Tensor:
        """Sample tokens from residual stream using greedy decoding."""
        if do_sample:
            raise NotImplementedError("Sampling not yet implemented")

        logits = self.unembed_residual(residual)
        return torch.argmax(logits, dim=-1)
