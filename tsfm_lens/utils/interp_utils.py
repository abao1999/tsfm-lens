from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from chronos import ChronosModel
from chronos.chronos_bolt import ChronosBoltModelForForecasting
from scipy.signal import find_peaks
from toto.model.backbone import TotoBackbone

from tsfm_lens.circuitlens import BaseCircuitLens


def collect_attributions(attributions: list[torch.Tensor]):
    """
    Collect head attributions for a given layer.
    Args:
        attributions: List of head attributions for each time step
    Returns:
        Tensor of shape (b, T, d_model)
    """
    return torch.cat([att[..., -1, :].unsqueeze(-2) for att in attributions], dim=-2)


# ========================================================
# for direct logit attribution visualization (see notebooks/logit_attribution.ipynb)


def plot_output_logits_across_layers(
    pipeline: "BaseCircuitLens",
    outputs: dict[int, torch.Tensor] | list[torch.Tensor],
    prediction_length: int,
    plot_save_path: str,
    output_type: Literal["MLP", "Read Stream", "Head"] | None = None,
    selected_parallel_sample_idx: int = 0,
    token_id_range: tuple[int, int] | None = None,
    figsize: tuple[int, int] = (20, 10),
    title: str | None = None,
) -> float:
    """
    Plots a concise visualization of logits for MLP outputs across all layers.

    Args:
        pipeline (BaseCircuitLens): The BaseCircuitLens pipeline object, which provides the unembedding method.
        mlp_outputs (dict[int, torch.Tensor]): Dictionary mapping layer indices to MLP output tensors of shape [num_samples, seq_len, d_model].
        prediction_length (int): Number of timesteps to plot on the x-axis.
        plot_save_path (str): Path to save the generated plot (PDF).
        selected_parallel_sample_idx (int, optional): Index of the parallel sample to visualize. Defaults to 0.
        token_id_range (tuple[int, int], optional): Tuple (start, end) for token ID y-axis range. Defaults to (1200, 3000).
        figsize (tuple[int, int], optional): Figure size for the plot. Defaults to (20, 10).

    Returns:
        float: The vabs of the logits, for consistent plotting across different runs.
    """

    if output_type and output_type == "Head":
        num_heads = len(outputs)
        assert num_heads == pipeline.model.model.config.num_heads, (
            f"Number of heads mismatch: {num_heads} != {pipeline.model.model.config.num_heads}"
        )
        print(f"num_heads: {num_heads}")
        n = num_heads
        label = "Head"

    # elif output_type in {"MLP", "Read Stream"}:
    else:
        num_layers = len(outputs)
        assert num_layers == pipeline.model.model.config.num_decoder_layers, (
            f"Number of layers mismatch: {num_layers} != {pipeline.model.model.config.num_decoder_layers}"
        )
        print(f"num_layers: {num_layers}")
        n = num_layers
        label = "Layer"

    # Dynamically choose nrows and ncols for a square-like grid, prioritizing width (ncols >= nrows)
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    print(f"nrows: {nrows}, ncols: {ncols}")

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    overall_vabs = 0
    for i, ax in enumerate(axes[:n]):
        logits = pipeline.unembed_residual(outputs[i][selected_parallel_sample_idx]).detach().cpu().float().numpy()
        vabs = np.nanmax(np.abs(logits)) if np.any(np.isfinite(logits)) else 0
        overall_vabs = max(overall_vabs, vabs)
        im = ax.imshow(
            logits[:prediction_length, :].T,
            aspect="auto",
            cmap="RdBu",
            vmin=-vabs,
            vmax=vabs,
        )
        if token_id_range is not None:
            ax.set_ylim(token_id_range[1], token_id_range[0])
        ax.set_title(f"{label} {i}", fontweight="bold")
        ax.set_ylabel("Token ID", fontweight="bold")
        ax.set_xlabel("Timestep", fontweight="bold")
        ax.invert_yaxis()

    # Make room for colorbar, but keep it closer to the plots
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes((0.89, 0.15, 0.02, 0.7))  # move colorbar closer to plots
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel("Logit Value", fontsize=16, weight="bold")
    cbar.ax.tick_params(labelsize=14)

    # Center the suptitle over the plots (not the colorbar)
    if title:
        title = f"{title or output_type} Logit Maps"
        plt.suptitle(title, fontweight="bold", x=0.44, y=0.96, fontsize=22)
    plt.tight_layout(rect=(0, 0, 0.88, 0.95))
    if token_id_range is not None:
        plot_save_path = plot_save_path.replace(".pdf", f"_token_range-{token_id_range[0]}-{token_id_range[1]}.pdf")
    plt.savefig(plot_save_path, bbox_inches="tight")
    return overall_vabs


def plot_output_logits(
    pipeline: "BaseCircuitLens",
    mlp_outputs: dict[int, torch.Tensor],
    prediction_length: int,
    plot_save_path: str | None = None,
    selected_parallel_sample_idx: int = 0,
    selected_layer_idx: int = 11,
    token_id_range: tuple[int, int] | None = None,
    figsize: tuple[int, int] = (6, 6),
    enforced_vabs: float | None = None,
    show_colorbar: bool = True,
    show_yticks: bool = True,
) -> float:
    """
    Plot the logits produced by the MLP output of a specified layer.

    Args:
        pipeline (BaseCircuitLens): The BaseCircuitLens pipeline object, which provides the unembedding method.
        mlp_outputs (list): List of MLP output tensors for each layer, shape [num_layers][num_samples, seq_len, d_model].
        prediction_length (int): Number of timesteps to plot on the x-axis.
        plot_save_path (str): Path to save the generated plot (PDF).
        selected_parallel_sample_idx (int, optional): Index of the parallel sample to visualize. Defaults to 0.
        selected_layer_idx (int, optional): Layer index to visualize. Defaults to 11.
        token_id_range (tuple[int, int], optional): Tuple (start, end) for token ID y-axis range. Defaults to (1200, 3000).
        figsize (tuple[int, int], optional): Figure size for the plot. Defaults to (6, 6).
        enforced_vabs (float | None, optional): If not None, the absolute value of the logits will be clipped to this value. Defaults to None.

    Returns:
        float: The vabs of the logits, for consistent plotting across different runs.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    logits = (
        pipeline.unembed_residual(mlp_outputs[selected_layer_idx][selected_parallel_sample_idx])
        .detach()
        .cpu()
        .float()
        .numpy()
    )
    vabs = np.nanmax(np.abs(logits)) if np.any(np.isfinite(logits)) else 0
    if enforced_vabs is not None:
        vabs = enforced_vabs
    im = ax.imshow(
        logits[:prediction_length, :].T,
        aspect="auto",
        cmap="RdBu",
        vmin=-vabs,
        vmax=vabs,
    )
    if token_id_range is not None:
        ax.set_ylim(token_id_range[1], token_id_range[0])
    ax.set_title(f"Layer {selected_layer_idx}", fontweight="bold", fontsize=16)
    if show_yticks:
        ax.set_ylabel("Token ID", fontweight="bold", fontsize=12)
    else:
        ax.set_yticks([])
    ax.set_xlabel("Timestep", fontweight="bold", fontsize=12)
    ax.invert_yaxis()

    if show_colorbar:
        cbar_ax = fig.add_axes((0.82, 0.15, 0.02, 0.7))
        cbar = fig.colorbar(im, cax=cbar_ax)
        fig.subplots_adjust(right=0.81)
        cbar.ax.set_position((0.82, ax.get_position().y0, 0.025, ax.get_position().height))
        # cbar.ax.set_ylabel("Logit Value", weight="bold")
        plt.tight_layout(rect=(0, 0, 0.83, 0.95))
    else:
        plt.tight_layout()
    if token_id_range is not None and plot_save_path is not None:
        plot_save_path = plot_save_path.replace(".pdf", f"_token_range-{token_id_range[0]}-{token_id_range[1]}.pdf")
    if plot_save_path is not None:
        plt.savefig(plot_save_path, bbox_inches="tight")
    return vabs


def compute_logit_metrics(
    logits: np.ndarray,
    temperature: float = 1.0,
    threshold: float = 0.01,
    top_k: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Shannon entropy, effective vocabulary size, and top-k entropy of the probability distribution at each timestep.
    """
    logits = np.asarray(logits)
    added_batch = False
    if logits.ndim == 2:
        logits = logits[None, ...]
        added_batch = True
    elif logits.ndim not in (3, 4):
        raise ValueError(f"Unsupported logits shape: {logits.shape}")

    # Softmax with temperature
    logits = logits / temperature
    # logits_max = np.max(logits, axis=-1, keepdims=True)
    # exp_logits = np.exp(logits - logits_max)
    # probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()

    # Entropy and effective vocab size
    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
    effective_vocab_size = np.sum(probs > threshold, axis=-1)

    # Top-k entropy
    k = int(np.max(effective_vocab_size)) if top_k is None else top_k
    k = max(1, k)
    # Use np.partition for efficiency
    idx = np.argpartition(logits, -k, axis=-1)[..., -k:]
    top_k_logits = np.take_along_axis(logits, idx, axis=-1)
    top_k_logits_max = np.max(top_k_logits, axis=-1, keepdims=True)
    exp_top_k_logits = np.exp(top_k_logits - top_k_logits_max)
    top_k_probs = exp_top_k_logits / np.sum(exp_top_k_logits, axis=-1, keepdims=True)
    top_k_entropy = -np.sum(top_k_probs * np.log(top_k_probs + 1e-10), axis=-1)

    peak_counts, _ = peak_count_prominence(probs, height=0.02, min_distance=2, ensure_pmf=False)

    if added_batch:
        entropy = entropy[0]
        effective_vocab_size = effective_vocab_size[0]
        top_k_entropy = top_k_entropy[0]
        peak_counts = peak_counts[0]

    return entropy, effective_vocab_size, top_k_entropy, peak_counts


def _ensure_pmf(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= eps:
        return np.ones_like(p) / len(p)
    return p / s


def peak_count_prominence(
    probs: np.ndarray,
    height: float | None = None,
    min_prominence: float | None = None,
    min_distance: int = 1,
    ensure_pmf: bool = True,
) -> tuple[np.ndarray, list]:
    """
    Counts the number of peaks in each row of a 2D probability array, or in the last dimension of a higher-dimensional array,
    using a prominence threshold.

    Args:
        probs (np.ndarray): An array of shape (..., T, V), where each [..., t, :] is a probability mass function (PMF).
        min_prominence (float, optional): Minimum required prominence of peaks to be counted.
        min_distance (int, optional): Minimum required distance (in index units) between peaks.
        ensure_pmf (bool, optional): Whether to ensure that the PMF is normalized.
    Returns:
        Tuple[np.ndarray, List]:
            - counts: array of shape (..., T), where each entry is the number of peaks found in the corresponding row.
            - all_peaks: nested list of lists, where each sublist contains the indices of the peaks found in the corresponding row.
    """
    # Accept input of shape (..., T, V)
    if probs.ndim < 2:
        raise ValueError("probs must have at least 2 dimensions (T, V)")
    *batch_dims, T, V = probs.shape
    flat = probs.reshape(-1, T, V)
    n_batch = flat.shape[0]

    counts = np.zeros((n_batch, T), dtype=int)
    all_peaks: list[list[list[int]]] = []

    for b in range(n_batch):
        peaks_for_batch = []
        for t in range(T):
            if ensure_pmf:
                p = _ensure_pmf(flat[b, t])
            else:
                p = flat[b, t]
            peaks, _ = find_peaks(p, height=height, prominence=min_prominence, distance=min_distance)
            peaks = peaks.tolist()
            counts[b, t] = len(peaks)
            peaks_for_batch.append(peaks)
        all_peaks.append(peaks_for_batch)

    # Reshape counts to (..., T)
    counts = counts.reshape(*batch_dims, T)
    # all_peaks is a nested list: [batch][timestep][peak_indices]
    # Optionally, could reshape all_peaks to match batch_dims, but leave as nested list for generality
    return counts, all_peaks


def extract_projection_weights_Chronos(
    model: ChronosModel | ChronosBoltModelForForecasting,
    selected_layer: int,
    selected_head: int,
    selected_component: str,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract query (Q), key (K), value (V), and output (O) projection weight matrices
    for a specific attention head from a Chronos model.

    The function extracts the weights from the linear projection layers that transform
    input embeddings into query, key, and value vectors for multi-head attention, as well
    as the output projection that combines head outputs. Each head operates on a slice of
    the full model dimension.

    Args:
        model: The Chronos or ChronosBolt model instance to extract weights from
        selected_layer: Layer index (0-indexed) in the transformer decoder
        selected_head: Head index (0-indexed) within the attention layer. Must be in
            range [0, num_heads)
        selected_component: Component type to extract from:
            - "SA": Self-Attention layer
            - "CA": Cross-Attention (Encoder-Decoder Attention) layer
            - "MLP": MLP layer (though Q/K/V/O extraction is only meaningful for attention)
        verbose: If True, prints diagnostic information about shapes and indices

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing
            (head_WQ, head_WK, head_WV, head_WO) where:
            - head_WQ: Query projection weights with shape (head_dim, d_model)
            - head_WK: Key projection weights with shape (head_dim, d_model)
            - head_WV: Value projection weights with shape (head_dim, d_model)
            - head_WO: Output projection weights with shape (head_dim, d_model)
                * head_dim: Dimension per attention head (d_model // num_heads)
                * d_model: Full model embedding dimension

            All arrays are float32 numpy arrays detached from the computation graph.
            Q, K, V matrices project input vectors to query/key/value vectors.
            O matrix projects each head's output back to the full model dimension.

    NOTE: we adopt the convention that all per-head projection matrices are of shape (head_dim, d_model),
    where head_dim = d_model // num_heads. This could be confusing when looking at the mathematical form of the attention operation

    Example:
        >>> WQ, WK, WV, WO = extract_projection_weights_Chronos(
        ...     model, selected_layer=2, selected_head=5, selected_component="SA"
        ... )
        >>> print(f"Query weights shape: {WQ.shape}")  # e.g., (64, 768)
        >>> print(f"Key weights shape: {WK.shape}")    # e.g., (64, 768)
        >>> print(f"Value weights shape: {WV.shape}")  # e.g., (64, 768)
        >>> print(f"Output weights shape: {WO.shape}") # e.g., (64, 768)
    """
    component_type_mapping = {
        "SA": (0, "SelfAttention"),
        "CA": (1, "EncDecAttention"),
        "MLP": (2, "DenseReluDense"),
    }

    component_idx, component_name = component_type_mapping[selected_component]
    if verbose:
        print(f"component_idx: {component_idx}, component_name: {component_name}")

    layer = model.decoder.block[selected_layer].layer  # type: ignore
    layer_module_list = layer[component_idx]  # type: ignore
    module = getattr(layer_module_list, component_name)
    # print(f"module: {module}")

    num_heads: int = model.config.num_heads  # type: ignore
    if verbose:
        print(f"num_heads: {num_heads}")

    # Q, K, V: chunk along dim=0 (rows) since we're splitting the output space
    # Each produces d_model outputs that get split into num_heads chunks
    head_WQ = module.q.weight.chunk(num_heads, dim=0)[selected_head].float().detach().cpu().numpy()
    head_WK = module.k.weight.chunk(num_heads, dim=0)[selected_head].float().detach().cpu().numpy()
    head_WV = module.v.weight.chunk(num_heads, dim=0)[selected_head].float().detach().cpu().numpy()

    # O: chunk along dim=1 (columns) since it processes concatenated head outputs as input
    # Each head contributes head_dim dimensions to the concatenated input
    head_WO = module.o.weight.chunk(num_heads, dim=1)[selected_head].T.float().detach().cpu().numpy()

    if verbose:
        print(f"head_WQ shape: {head_WQ.shape}")
        print(f"head_WK shape: {head_WK.shape}")
        print(f"head_WV shape: {head_WV.shape}")
        print(f"head_WO shape: {head_WO.shape}")

    return head_WQ, head_WK, head_WV, head_WO


def extract_projection_weights_Toto(
    model: TotoBackbone,
    selected_layer: int,
    selected_head: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract query (Q), key (K), value (V), and output (O) projection weight matrices
    for a specific attention head from a Toto model.

    Unlike separate Q/K/V projections in standard transformers, Toto uses a fused QKV
    projection layer that computes all three projections simultaneously. This function
    extracts and separates the Q, K, V, and O components for a specific head.

    Args:
        model: The TotoBackbone model instance to extract weights from
        selected_layer: Layer index (0-indexed) in the transformer
        selected_head: Head index (0-indexed) within the attention layer. Must be in
            range [0, num_heads)
        verbose: If True, prints diagnostic information about shapes and indices

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing
            (head_WQ, head_WK, head_WV, head_WO) where:
            - head_WQ: Query projection weights with shape (head_dim, d_model)
            - head_WK: Key projection weights with shape (head_dim, d_model)
            - head_WV: Value projection weights with shape (head_dim, d_model)
            - head_WO: Output projection weights with shape (head_dim, d_model)
                * head_dim: Dimension per attention head (d_model // num_heads)
                * d_model: Full model embedding dimension

            All arrays are float32 numpy arrays detached from the computation graph.
            Q, K, V matrices project input vectors to query/key/value vectors.
            O matrix projects each head's output back to the full model dimension.

            Note: Shape convention matches extract_projection_weights_Chronos for
            consistency across different model architectures.

    Raises:
        ValueError: If the specified layer does not have a wQKV matrix

    NOTE: we adopt the convention that all per-head projection matrices are of shape (head_dim, d_model),
    where head_dim = d_model // num_heads. This could be confusing when looking at the mathematical form of the attention operation

    Example:
        >>> WQ, WK, WV, WO = extract_projection_weights_Toto(
        ...     model, selected_layer=2, selected_head=5
        ... )
        >>> print(f"Query weights shape: {WQ.shape}")  # e.g., (64, 768)
        >>> print(f"Key weights shape: {WK.shape}")    # e.g., (64, 768)
        >>> print(f"Value weights shape: {WV.shape}")  # e.g., (64, 768)
        >>> print(f"Output weights shape: {WO.shape}") # e.g., (64, 768)
    """

    if not hasattr(model.transformer.layers[selected_layer].attention, "wQKV"):
        raise ValueError(f"Layer {selected_layer} does not have a wQKV matrix")

    wQKV_layer = model.transformer.layers[selected_layer].attention.wQKV  # type: ignore
    wO_layer = model.transformer.layers[selected_layer].attention.wO  # type: ignore

    num_heads: int = model.transformer.layers[selected_layer].num_heads  # type: ignore
    if verbose:
        print(f"num_heads: {num_heads}")

    wO_weight = wO_layer.weight  # type: ignore
    wO = torch.tensor(wO_weight.T)
    if verbose:
        print(f"wO_weight shape: {wO_weight.shape}")
        print(f"wO transposed shape: {wO.shape}")

    # shape: (out_features, in_features) = (768*3, 768)
    wQKV_weight = wQKV_layer.weight  # type: ignore
    wQKV_weight_T = wQKV_weight.T  # Transpose to get (in_features, out_features) = (768, 768*3)
    if verbose:
        print(f"wQKV_weight shape: {wQKV_weight.shape}")
        print(f"wQKV_weight transposed shape: {wQKV_weight_T.shape}")

    wQ, wK, wV = wQKV_weight_T.chunk(3, dim=-1)  # type: ignore
    if verbose:
        print(f"wQ shape: {wQ.shape}, wK shape: {wK.shape}, wV shape: {wV.shape}")

    # Now separate each into individual heads
    # Each head has dimension d_head = d_model / num_heads = 768 / 12 = 64
    d_model = wQ.shape[0]
    d_head = d_model // num_heads

    print(f"\nd_model: {d_model}, num_heads: {num_heads}, d_head: {d_head}")

    # Split along output dimension (dim=1) into num_heads chunks
    # Each will be (768, 64)
    wQ_heads = wQ.chunk(num_heads, dim=1)  # List of num_heads tensors, each (768, 64)
    wK_heads = wK.chunk(num_heads, dim=1)
    wV_heads = wV.chunk(num_heads, dim=1)
    # For wO, split along input dimension (dim=0) since it processes concatenated head outputs
    wO_heads = wO.chunk(num_heads, dim=0)  # List of num_heads tensors, each (64, 768)

    # Example: Access weights for head i
    # Transpose Q, K, V to match (head_dim, d_model) convention used in extract_projection_weights_Chronos
    selected_head_WQ = wQ_heads[selected_head].T.float().detach().cpu().numpy()
    selected_head_WK = wK_heads[selected_head].T.float().detach().cpu().numpy()
    selected_head_WV = wV_heads[selected_head].T.float().detach().cpu().numpy()
    # O is already (head_dim, d_model) after chunking dim=0, no transpose needed
    selected_head_WO = wO_heads[selected_head].float().detach().cpu().numpy()

    if verbose:
        print(f"\nNumber of Q heads: {len(wQ_heads)}")
        print(f"Shape of each Q head (before transpose): {wQ_heads[0].shape}")
        print(f"Number of K heads: {len(wK_heads)}")
        print(f"Shape of each K head (before transpose): {wK_heads[0].shape}")
        # print(f"Number of V heads: {len(wV_heads)}")
        # print(f"Shape of each V head: {wV_heads[0].shape}")

        print(f"wQ[head {selected_head}] shape (after transpose): {selected_head_WQ.shape}")
        print(f"wK[head {selected_head}] shape (after transpose): {selected_head_WK.shape}")
        print(f"wV[head {selected_head}] shape (after transpose): {selected_head_WV.shape}")
        print(f"wO[head {selected_head}] shape: {selected_head_WO.shape}")

    return selected_head_WQ, selected_head_WK, selected_head_WV, selected_head_WO


def extract_projection_weights_TimesFM2p5(
    model,  # TimesFM 2.5 model
    selected_layer: int,
    selected_head: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract query (Q), key (K), value (V), and output (O) projection weight matrices
    for a specific attention head from a TimesFM 2.5 model.

    TimesFM 2.5 uses a fused QKV projection layer similar to Toto. This function
    extracts and separates the Q, K, V, and O components for a specific head.

    Args:
        model: The TimesFM 2.5 model instance to extract weights from
        selected_layer: Layer index (0-indexed) in the transformer. Must be in range [0, 20)
        selected_head: Head index (0-indexed) within the attention layer. Must be in
            range [0, 16)
        verbose: If True, prints diagnostic information about shapes and indices

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing
            (head_WQ, head_WK, head_WV, head_WO) where:
            - head_WQ: Query projection weights with shape (head_dim, d_model)
            - head_WK: Key projection weights with shape (head_dim, d_model)
            - head_WV: Value projection weights with shape (head_dim, d_model)
            - head_WO: Output projection weights with shape (head_dim, d_model)
                * head_dim: Dimension per attention head (1280 // 16 = 80)
                * d_model: Full model embedding dimension (1280)

            All arrays are float32 numpy arrays detached from the computation graph.
            Q, K, V matrices project input vectors to query/key/value vectors.
            O matrix projects each head's output back to the full model dimension.

            Note: Shape convention matches extract_projection_weights_Chronos and
            extract_projection_weights_Toto for consistency across different model
            architectures.

    Raises:
        ValueError: If the specified layer does not have a qkv_proj matrix or out projection

    NOTE: we adopt the convention that all per-head projection matrices are of shape (head_dim, d_model),
    where head_dim = d_model // num_heads. This could be confusing when looking at the mathematical form of the attention operation

    Example:
        >>> WQ, WK, WV, WO = extract_projection_weights_TimesFM2p5(
        ...     model, selected_layer=2, selected_head=5
        ... )
        >>> print(f"Query weights shape: {WQ.shape}")   # (80, 1280)
        >>> print(f"Key weights shape: {WK.shape}")     # (80, 1280)
        >>> print(f"Value weights shape: {WV.shape}")   # (80, 1280)
        >>> print(f"Output weights shape: {WO.shape}")  # (80, 1280)
    """

    if not hasattr(model.stacked_xf[selected_layer].attn, "qkv_proj"):
        raise ValueError(f"Layer {selected_layer} does not have a qkv_proj matrix")
    if not hasattr(model.stacked_xf[selected_layer].attn, "out"):
        raise ValueError(f"Layer {selected_layer} does not have an out projection")

    qkv_proj_layer = model.stacked_xf[selected_layer].attn.qkv_proj
    out_proj_layer = model.stacked_xf[selected_layer].attn.out

    # TimesFM 2.5 has 16 heads per layer
    num_heads = 16

    if verbose:
        print(f"num_heads: {num_heads}")

    # Access the weight tensor from the Linear module
    # shape: (out_features, in_features) = (3840, 1280) = (1280*3, 1280)
    qkv_weight = qkv_proj_layer.weight

    if verbose:
        print(f"qkv_weight shape: {qkv_weight.shape}")

    # Transpose to get (in_features, out_features) = (1280, 3840) for easier interpretation
    qkv_weight_T = qkv_weight.T

    if verbose:
        print(f"qkv_weight transposed shape: {qkv_weight_T.shape}")

    # Split into Q, K, V: each will be (1280, 1280)
    wQ, wK, wV = qkv_weight_T.chunk(3, dim=-1)  # type: ignore

    if verbose:
        print(f"wQ shape: {wQ.shape}, wK shape: {wK.shape}, wV shape: {wV.shape}")

    # Now separate each into individual heads
    # Each head has dimension d_head = d_model / num_heads = 1280 / 16 = 80
    d_model = wQ.shape[0]
    d_head = d_model // num_heads

    if verbose:
        print(f"\nd_model: {d_model}, num_heads: {num_heads}, d_head: {d_head}")

    # Split along output dimension (dim=1) into num_heads chunks
    # Each will be (1280, 80)
    wQ_heads = wQ.chunk(num_heads, dim=1)  # List of num_heads tensors, each (1280, 80)
    wK_heads = wK.chunk(num_heads, dim=1)
    wV_heads = wV.chunk(num_heads, dim=1)

    # Transpose Q, K, V to match (head_dim, d_model) convention
    selected_head_WQ = wQ_heads[selected_head].T.float().detach().cpu().numpy()
    selected_head_WK = wK_heads[selected_head].T.float().detach().cpu().numpy()
    selected_head_WV = wV_heads[selected_head].T.float().detach().cpu().numpy()

    # Extract output projection
    # out.weight has shape (1280, 1280) = (out_features, in_features)
    # Chunk along dim=1 (columns) since it processes concatenated head outputs as input
    # NOTE: we don't transpose this here (we could, but then we would chunk along dim=0 later for out_heads)
    out_weight = out_proj_layer.weight
    if verbose:
        print(f"\nout_weight shape: {out_weight.shape}")

    # Chunk along dim=1 to get per-head contribution, then transpose
    out_heads = out_weight.chunk(num_heads, dim=1)  # Each (1280, 80)
    selected_head_WO = out_heads[selected_head].T.float().detach().cpu().numpy()  # (80, 1280)

    if verbose:
        print(f"\nNumber of Q heads: {len(wQ_heads)}")
        print(f"Shape of each Q head (before transpose): {wQ_heads[0].shape}")
        print(f"Number of K heads: {len(wK_heads)}")
        print(f"Shape of each K head (before transpose): {wK_heads[0].shape}")
        print(f"Number of V heads: {len(wV_heads)}")
        print(f"Shape of each V head (before transpose): {wV_heads[0].shape}")
        print(f"\nwQ[head {selected_head}] shape (after transpose): {selected_head_WQ.shape}")
        print(f"wK[head {selected_head}] shape (after transpose): {selected_head_WK.shape}")
        print(f"wV[head {selected_head}] shape (after transpose): {selected_head_WV.shape}")
        print(f"wO[head {selected_head}] shape (after transpose): {selected_head_WO.shape}")

    return selected_head_WQ, selected_head_WK, selected_head_WV, selected_head_WO


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


def _row_entropy(P: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return -np.sum(P * np.log(P + eps), axis=axis)


def _orth(A: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Return orthonormal basis for col(A) via QR with rank trimming."""
    Q, R = np.linalg.qr(A)
    r = np.sum(np.abs(np.diag(R)) > eps)
    return Q[:, :r]


def _numerical_rank(s: np.ndarray, tol: float | None = None) -> int:
    if tol is None:
        tol = np.max(s) * max(len(s), 1) * np.finfo(s.dtype).eps
    return int(np.sum(s > tol))


def diagnose_attention(
    W_Q: np.ndarray,
    W_K: np.ndarray,
    H: np.ndarray | None = None,
    scale_by_sqrt_dk: bool = True,
    top_k: int = 10,
    eps: float = 1e-10,
) -> dict[str, Any]:
    """
    Perform comprehensive diagnostic analysis of an attention head's query-key interaction matrix.

    This function analyzes the attention pattern matrix M = W_Q^T @ W_K to understand the
    geometric and algebraic properties of how queries and keys interact. It computes various
    metrics useful for mechanistic interpretability, including rank analysis, spectral properties,
    symmetric/skew decomposition, and optionally attention sharpness on actual hidden states.

    The analysis helps answer questions like:
    - Is the attention pattern low-rank? (spectral analysis)
    - Is attention symmetric or directional? (symmetric/skew decomposition)
    - How aligned are the query and key subspaces? (principal angles)
    - How sharp/diffuse are attention distributions? (entropy analysis with H)

    Args:
        W_Q: Query projection weights with shape (head_dim, d_model)
            This matrix projects d_model-dimensional inputs to head_dim-dimensional queries.
        W_K: Key projection weights with shape (head_dim, d_model), matching W_Q's shape.
            This matrix projects d_model-dimensional inputs to head_dim-dimensional keys.
        H: Optional hidden states with shape (seq_len, d_model) for computing attention
            distributions and sharpness metrics. If provided, computes softmax statistics,
            attention entropy, and alignment scores. If None, only weight-based analysis is done.
        scale_by_sqrt_dk: If True, scales attention logits by 1/sqrt(head_dim) before softmax,
            following standard scaled dot-product attention. Only used when H is provided.
        top_k: Number of top singular values to report and analyze in the summary.
        eps: Small constant for numerical stability in divisions and rank estimation.

    Returns:
        dict[str, Any]: Comprehensive diagnostics dictionary containing:

            Spectral Properties:
                - spectral_norm (float): Largest singular value σ₁ of M
                - spectral_gap_sigma1_over_sigma2 (float): Ratio σ₁/σ₂, measures rank-1 dominance
                - frob_norm (float): Frobenius norm ‖M‖_F
                - nuclear_norm (float): Sum of singular values Σσᵢ
                - rank_numerical (int): Numerical rank (number of σᵢ > tol)
                - singular_values (np.ndarray): All singular values in descending order
                - explained_energy (np.ndarray): Cumulative energy explained by top-k singular values
                - explained_nuclear_norm (np.ndarray): Cumulative nuclear norm by top-k singular values

            Decomposition:
                - U (np.ndarray): Left singular vectors of M
                - V (np.ndarray): Right singular vectors of M
                - symmetric_part (np.ndarray): Symmetric component H_sym = (M + M^T)/2
                - skew_part (np.ndarray): Skew-symmetric component S_skew = (M - M^T)/2
                - skew_fraction_of_energy (float): Fraction of energy in skew part,
                  measures how directional (vs symmetric) the attention is

            Geometric Properties:
                - principal_angles_radians (np.ndarray): Principal angles between Q and K subspaces
                - principal_cosines (np.ndarray): Cosines of principal angles (singular values of Q^T K)

            Sequence-Level Statistics (only if H is provided):
                - softmax_stats (dict | None): Contains:
                    * entropy_mean, entropy_std: Mean/std of attention entropies per query
                    * max_weight_mean, max_weight_std: Mean/std of max attention weights per query
                    * alignment_mean, alignment_std: Mean/std of query alignment to top singular vectors
                    * corr_alignment_vs_neg_entropy: Correlation between alignment and sharpness
                - per_query (dict | None): Per-query arrays:
                    * entropies: Attention entropy for each query position
                    * max_weight: Maximum attention weight for each query position
                    * alignment_scores: Alignment score for each query position

            Summary:
                - human_readable_summary (str): Multi-line formatted summary of key findings

    Example:
        >>> # Extract weights from a model
        >>> WQ, WK = extract_projection_weights_Chronos(model, layer=2, head=5, component="SA")
        >>>
        >>> # Basic diagnostics without hidden states
        >>> diag = diagnose_attention(WQ, WK)
        >>> print(diag["human_readable_summary"])
        >>> print(f"Rank: {diag['rank_numerical']}, Skew fraction: {diag['skew_fraction_of_energy']:.3f}")
        >>>
        >>> # Full diagnostics with hidden states
        >>> hidden_states = ...  # shape (seq_len, d_model)
        >>> diag_full = diagnose_attention(WQ, WK, H=hidden_states)
        >>> print(f"Avg entropy: {diag_full['softmax_stats']['entropy_mean']:.3f}")
    """
    d_k, d_h = W_Q.shape
    assert W_K.shape == (d_k, d_h)
    M = W_Q.T @ W_K

    # SVD and basic norms
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    V = Vt.T
    spectral_norm = float(s[0]) if len(s) else 0.0
    frob_norm = float(np.linalg.norm(M, "fro"))
    nuclear_norm = float(np.sum(s))
    rank_num = _numerical_rank(s)
    spectral_gap = float(s[0] / s[1]) if len(s) > 1 and s[1] > eps else np.inf

    # Symmetric/skew decomposition
    Hsym = 0.5 * (M + M.T)
    Smew = 0.5 * (M - M.T)
    sym_energy = float(np.linalg.norm(Hsym, "fro") ** 2)
    skew_energy = float(np.linalg.norm(Smew, "fro") ** 2)
    skew_frac = float(skew_energy / (sym_energy + skew_energy + eps))

    # Principal angles
    Qb, Kb = _orth(W_Q.T), _orth(W_K.T)
    C = np.linalg.svd(Qb.T @ Kb, full_matrices=False)[1] if (Qb.size and Kb.size) else np.array([])
    principal_angles = np.arccos(np.clip(C, 0.0, 1.0))

    # Sequence-level analysis if H provided
    softmax_stats = None
    per_query = None
    if H is not None:
        assert H.shape[1] == d_h
        L = H @ M @ H.T
        if scale_by_sqrt_dk:
            L = L / np.sqrt(d_k)
        P = _softmax(L, axis=1)
        entropies = _row_entropy(P, axis=1)
        max_weight = np.max(P, axis=1)

        alpha = H @ U
        alignment_scores = (alpha**2) @ s / (np.linalg.norm(H @ U, axis=1) ** 2 + eps)

        x = alignment_scores - np.mean(alignment_scores)
        y = (-entropies) - np.mean(-entropies)
        corr = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + eps))

        softmax_stats = {
            "entropy_mean": float(np.mean(entropies)),
            "entropy_std": float(np.std(entropies)),
            "max_weight_mean": float(np.mean(max_weight)),
            "max_weight_std": float(np.std(max_weight)),
            "alignment_mean": float(np.mean(alignment_scores)),
            "alignment_std": float(np.std(alignment_scores)),
            "corr_alignment_vs_neg_entropy": corr,
        }
        per_query = {"entropies": entropies, "max_weight": max_weight, "alignment_scores": alignment_scores}

    # Summary
    k = min(top_k, len(s))
    explained_nuclear_norm = np.cumsum(s[:k]) / (np.sum(s) + eps)
    explained_energy = np.cumsum(s[:k] ** 2) / (np.sum(s**2) + eps)

    summary = [
        f"(d_h={d_h}, d_k={d_k}) rank≈{rank_num}/{d_h} ‖M‖₂={spectral_norm:.4g} gap={spectral_gap:.3g}",
        f"Skew fraction={skew_frac:.3f}",
    ]
    if principal_angles.size:
        summary.append(
            f"Principal angles: median={np.degrees(np.median(principal_angles)):.1f}° max_cos={np.max(C):.3f}"
        )
    summary.append(f"Top-{k} σ: {np.array2string(s[:k], precision=4, suppress_small=True)}")
    summary.append(f"Explained (nuclear): {np.array2string(explained_nuclear_norm, precision=3, suppress_small=True)}")
    summary.append(f"Explained (Frobenius): {np.array2string(explained_energy, precision=3, suppress_small=True)}")
    if softmax_stats:
        summary.append(
            f"Entropy: {softmax_stats['entropy_mean']:.3f}±{softmax_stats['entropy_std']:.3f} "
            f"Max_p: {softmax_stats['max_weight_mean']:.3f}±{softmax_stats['max_weight_std']:.3f}"
        )
        summary.append(f"Alignment↔sharpness corr: {softmax_stats['corr_alignment_vs_neg_entropy']:.3f}")

    return {
        "spectral_norm": spectral_norm,
        "spectral_gap_sigma1_over_sigma2": spectral_gap,
        "frob_norm": frob_norm,
        "nuclear_norm": nuclear_norm,
        "rank_numerical": rank_num,
        "singular_values": s,
        "U": U,
        "V": V,
        "symmetric_part": Hsym,
        "skew_part": Smew,
        "skew_fraction_of_energy": skew_frac,
        "principal_angles_radians": principal_angles,
        "principal_cosines": C if principal_angles.size else np.array([]),
        "softmax_stats": softmax_stats,
        "per_query": per_query,
        "human_readable_summary": "\n".join(summary),
        "explained_energy": explained_energy,
        "explained_nuclear_norm": explained_nuclear_norm,
    }


# Helper function to create combined corner plot
def plot_combined_corner(
    U_vectors: np.ndarray,
    V_vectors: np.ndarray,
    title: str,
    num_svs: int = 6,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    save_path: str | None = None,
):
    fig, axes = plt.subplots(num_svs, num_svs, figsize=(3 * num_svs, 3 * num_svs))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for i in range(num_svs):
        for j in range(num_svs):
            ax = axes[i, j]
            if i == j:
                # Diagonal: histograms for both U and V
                combined_data = np.concatenate([U_vectors[:, i], V_vectors[:, i]])
                bins = np.linspace(combined_data.min(), combined_data.max(), 30 + 1)
                ax.hist(
                    U_vectors[:, i], bins=bins, color="tab:blue", alpha=0.5, histtype="stepfilled", label="U (Left SVs)"
                )
                ax.hist(
                    V_vectors[:, i], bins=bins, color="tab:red", alpha=0.5, histtype="stepfilled", label="V (Right SVs)"
                )
                ax.set_title(f"SV {i + 1}", fontweight="bold")
                ax.legend()
            elif i > j:
                # Lower triangle: left singular vectors (U)
                ax.hist2d(U_vectors[:, j], U_vectors[:, i], bins=30, cmap="Blues")
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
            else:
                # Upper triangle: right singular vectors (V)
                ax.hist2d(V_vectors[:, j], V_vectors[:, i], bins=30, cmap="Oranges")
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)

    # plt.subplots_adjust(top=0.92)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
