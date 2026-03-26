# Abstract base class for CircuitLens

from typing import Any, Literal

import torch
from torch.utils.hooks import RemovableHandle


class BaseCircuitLens:
    """Base class for CircuitLens implementations.

    NOTE: currently works for encoder-decoder models
    For decoder-only models, should be very straightforward to extend
    i.e. only SA heads (no CA heads) for decoder-only models

    This class defines the interface for circuit analysis tools.
    Subclasses should implement all methods marked with NotImplementedError.
    """

    device: torch.device = torch.device("cuda")  # NOTE: this is just a dummy placeholder

    def __init__(self, model):
        self.model = model
        self.num_layers: int = 0  # Should be set by subclasses
        self.num_heads: int = 0  # Should be set by subclasses
        self.is_decoder_only: bool = False  # Should be set by subclasses
        self.read_stream_handles: dict[int, RemovableHandle] = {}
        self.read_stream_outputs: dict[int, list[torch.Tensor]] = {}
        self.read_stream_inputs: dict[int, list[torch.Tensor]] = {}

    def __post_init__(self):
        raise NotImplementedError("Subclasses must implement __post_init__ to set num_layers and num_heads")

    def set_to_eval_mode(self) -> None:
        """Set the model to evaluation mode."""
        raise NotImplementedError("Subclasses must implement set_to_eval_mode")

    def _ablate_head_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        head_indices: list[int],
        ablation_method: Literal["zero", "mean"],
    ) -> Any:
        raise NotImplementedError("Subclasses must implement _ablate_head_hook_fn")

    def _ablate_mlp_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        ablation_method: Literal["zero", "mean"],
    ) -> Any:
        raise NotImplementedError("Subclasses must implement _ablate_mlp_hook_fn")

    def add_ablation_hooks_explicit(
        self,
        ablations_types: list[Literal["head", "mlp"]],
        layers_to_ablate_mlp: list[int],
        heads_to_ablate: list[tuple[int, int]],
    ) -> None:
        raise NotImplementedError("Subclasses must implement add_ablation_hooks_explicit")

    def add_head_ablation_hooks(
        self,
        heads_to_ablate: list[tuple[int, int]],
        ablation_method: Literal["zero", "mean"] = "zero",
        attention_type: Literal["sa", "ca"] = "ca",
        stack_type: Literal["encoder", "decoder"] | None = None,
    ) -> dict[int, RemovableHandle]:
        raise NotImplementedError("Subclasses must implement add_head_ablation_hooks")

    def add_mlp_ablation_hooks(
        self,
        mlp_to_ablate: list[int],
        ablation_method: Literal["zero", "mean"] = "zero",
        stack_type: Literal["encoder", "decoder"] | None = None,
    ) -> dict[int, RemovableHandle]:
        raise NotImplementedError("Subclasses must implement add_mlp_ablation_hooks")

    def remove_head_ablation_hooks(self, attention_type: Literal["sa", "ca"] = "ca") -> None:
        raise NotImplementedError("Subclasses must implement remove_head_ablation_hooks")

    def remove_all_hooks(self) -> None:
        raise NotImplementedError("Subclasses must implement remove_all_hooks")

    def unembed_residual(self, residual: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement unembed_residual")

    def sample_tokens(self, residual: torch.Tensor, do_sample: bool = False) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement sample_tokens")

    def _read_stream_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        layer_idx: int,
    ) -> Any:
        raise NotImplementedError("Subclasses must implement _read_stream_hook_fn")

    def add_read_stream_hooks(
        self,
        layer_idxs: list[int],
        stack_type: Literal["encoder", "decoder"] | None = None,
    ) -> dict[int, RemovableHandle]:
        raise NotImplementedError("Subclasses must implement add_read_stream_hooks")

    def remove_read_stream_hooks(self) -> None:
        raise NotImplementedError("Subclasses must implement remove_read_stream_hooks")

    ### NOTE: all attribution functions are optional for child classes ###
    def reset_attribution_inputs_and_outputs(self) -> None:
        pass

    def _logit_attribution_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        key: str,
    ) -> Any:
        pass

    def _attribute_head_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        head_indices: list[int],
        layer_idx: int,
        attention_type: Literal["sa", "ca"],
    ) -> Any:
        pass

    def _attribute_attn_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        layer_idx: int,
        attention_type: Literal["sa", "ca"],
    ) -> Any:
        pass

    def _attribute_mlp_hook_fn(
        self,
        module: torch.nn.Module,
        input: Any,
        output: Any,
        layer_idx: int,
    ) -> Any:
        pass

    def add_head_attribution_hooks(
        self,
        heads_to_attribute: list[tuple[int, int]],
        attention_type: Literal["sa", "ca"] = "ca",
        stack_type: Literal["encoder", "decoder"] | None = None,
    ) -> dict[int, RemovableHandle] | None:
        pass

    def add_attn_attribution_hooks(
        self,
        layer_idxs: list[int],
        attention_type: Literal["sa", "ca"] = "ca",
        stack_type: Literal["encoder", "decoder"] | None = None,
    ) -> dict[int, RemovableHandle] | None:
        pass

    def add_mlp_attribution_hooks(
        self,
        mlp_to_attribute: list[int],
        stack_type: Literal["encoder", "decoder"] | None = None,
    ) -> dict[int, RemovableHandle] | None:
        pass

    def add_logit_attribution_hooks(
        self,
        target_layer: torch.nn.Module,
        key: str,
    ) -> dict[str, RemovableHandle] | None:
        pass

    def remove_head_attribution_hooks(self, attention_type: Literal["sa", "ca"] = "ca") -> None:
        pass

    def remove_attn_attribution_hooks(self, attention_type: Literal["sa", "ca"] = "ca") -> None:
        pass

    def remove_mlp_attribution_hooks(self) -> None:
        pass

    def remove_custom_attribution_hooks(self) -> None:
        pass
