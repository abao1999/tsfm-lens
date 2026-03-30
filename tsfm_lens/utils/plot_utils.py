import os
import re
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from omegaconf import OmegaConf

DEFAULT_COLORS = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
DEFAULT_MARKERS = ["o", "s", "v", "D", "X", "P", "H", "h", "d", "p", "x"]

METRICS_TRANSFORMS = {
    "mae": (lambda x: np.array(x, float), "MAE"),
    "smape": (lambda x: np.array(x, float), "sMAPE"),
    "spectral_hellinger": (lambda x: np.array(x, float), "Spectral Hellinger"),
    "spearman": (lambda x: 1 - np.array(x, float), "Spearman Distance"),
    "pearson": (lambda x: 1 - np.array(x, float), "Pearson Distance"),
    "ssim": (lambda x: 1 - np.array(x, float), "1 - SSIM"),
    "ms_ssim": (lambda x: np.array(x, float), "MS-SSIM"),
    "mmd": (lambda x: np.array(x, float), "Maximum Mean Discrepancy"),
    "spectral_mmd": (lambda x: np.array(x, float), "Spectral MMD-RBF"),
    "spectral_wasserstein-1": (lambda x: np.array(x, float), "Spectral Wasserstein-1"),
    "spectral_wasserstein-2": (lambda x: np.array(x, float), "Spectral Wasserstein-2"),
    "cross_spectral_phase_similarity": (lambda x: 1 - np.array(x, float), "1 - CSPS"),
    "mean_coherence": (lambda x: np.array(x, float), "Mean Coherence"),
    "energy_distance": (lambda x: np.array(x, float), "Energy Distance"),
}

# ===== CORE UTILITY FUNCTIONS =====


def _parse_layer_ablation_key(key: str):
    """Parse ablation key to extract layer information."""
    match = re.search(r"za_(heads|mlps|heads_mlps)_layers_([\d\-]+)", key)
    if not match:
        raise ValueError(f"Unrecognized ablation key format: {key}")
    layers = [int(x) for x in match.group(2).split("-")]
    return layers[0], layers


def _extract_metric_values(horizon_data, metric_name):
    """Extract and flatten metric values from horizon data."""
    all_vals = []
    for system_data in horizon_data.values():
        metric_values = system_data.get(metric_name, [])
        if isinstance(metric_values, list):
            all_vals.extend(metric_values)
        elif isinstance(metric_values, (int, float)) and not np.isnan(metric_values):
            all_vals.append(metric_values)
    return all_vals


def _get_metric_transform(metric_name):
    """Get metric transform function and display name."""
    return METRICS_TRANSFORMS.get(metric_name, (lambda x: np.array(x, dtype=float), metric_name.upper()))


def _process_metric_data(horizon_data, metric_name):
    """Process metric data: extract, transform, and filter NaN values."""
    transform, _ = _get_metric_transform(metric_name)
    all_vals = _extract_metric_values(horizon_data, metric_name)
    arr = transform(all_vals)
    return arr[~np.isnan(arr)]


def _compute_stats(arr, percentile_envelope=(25, 75)):
    """Compute median and percentile envelope for an array."""
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return (np.median(arr), np.percentile(arr, percentile_envelope[0]), np.percentile(arr, percentile_envelope[1]))


def _compute_aggregates(values_arr, aggregation, percentile_range=(25, 75)):
    """Compute aggregate statistics for an array."""
    if len(values_arr) == 0:
        return np.nan, np.nan, np.nan, np.nan
    agg_func = np.nanmedian if aggregation == "median" else np.nanmean
    aggregate = agg_func(values_arr)
    std_err = np.nanstd(values_arr) / np.sqrt(len(values_arr))
    iqr = np.nanpercentile(values_arr, percentile_range[1]) - np.nanpercentile(values_arr, percentile_range[0])
    mad = np.nanmean(np.abs(values_arr - np.nanmedian(values_arr)))
    return aggregate, std_err, iqr, mad


def get_layergroup_metrics(
    data: dict[str, dict[str, dict[str, dict[str, float | list[float]]]]], metric_name: str, prediction_horizon: str
) -> tuple[list[np.ndarray], list[str]]:
    horizon_data = {k: v[prediction_horizon] for k, v in data.items() if prediction_horizon in v}
    if not horizon_data:
        raise ValueError(f"No data found for prediction horizon '{prediction_horizon}'")
    layer_info, x_labels_full = _get_layer_info_and_labels(horizon_data)
    layer_group_metrics = [_process_metric_data(horizon_data[layer_key], metric_name) for _, _, layer_key in layer_info]
    if not layer_group_metrics or all(arr.size == 0 for arr in layer_group_metrics):
        raise ValueError("No valid metric values to plot after filtering.")
    return layer_group_metrics, x_labels_full


### Plotting Utilities ###
def _setup_plot_formatting(ax, xlabel, ylabel, title, x_labels=None, rotation=45):
    """Apply common plot formatting."""
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    if title:
        ax.set_title(title, fontweight="bold")
    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=rotation, ha="right")


def _save_plot(save_path):
    """Save plot with tight layout."""
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")


def _get_colors(cmap, n_colors, cmap_range=(0.1, 0.9)):
    """Get color array from colormap."""
    return plt.cm.get_cmap(cmap)(np.linspace(cmap_range[0], cmap_range[1], n_colors))


def _get_layer_info_and_labels(data, layergroup_indices_to_plot=None):
    """Get sorted layer information and labels from ablation data."""
    layer_info = sorted(
        ((_parse_layer_ablation_key(k)[0], _parse_layer_ablation_key(k)[1], k) for k in data),
        key=lambda x: x[0],
    )

    if layergroup_indices_to_plot is not None:
        layer_info = [info for info in layer_info if info[0] in layergroup_indices_to_plot]

    labels = [f"{layers[0]}" if len(layers) == 1 else f"{layers[0]}-{layers[-1]}" for _, layers, _ in layer_info]
    return layer_info, labels


def apply_custom_style(config_path: str):
    """Apply custom matplotlib style from config file with rcparams."""
    if not os.path.exists(config_path):
        print(f"Warning: Plotting config not found at {config_path}")
        return

    cfg = OmegaConf.load(config_path)
    plt.style.use(cfg.base_style)

    custom_rcparams = OmegaConf.to_container(cfg.matplotlib_style, resolve=True)
    for category, settings in custom_rcparams.items():  # type: ignore
        if isinstance(settings, dict):
            for param, value in settings.items():
                if isinstance(value, dict):
                    for subparam, subvalue in value.items():
                        plt.rcParams[f"{category}.{param}.{subparam}"] = subvalue
                else:
                    plt.rcParams[f"{category}.{param}"] = value


def plot_3d_and_univariate_with_predictions(
    context: np.ndarray,
    future_vals: np.ndarray,
    predictions: np.ndarray,
    title: str | None = None,
    title_kwargs: dict[str, Any] | None = None,
    save_path: str | None = None,
    figsize: tuple[int, int] = (6, 8),
    colormap: str = "Blues",
) -> None:
    """Plot context, ground truth, and predictions in 3D and as univariate time series."""
    title_kwargs = title_kwargs or {}

    def to_2d(arr):
        arr = np.asarray(arr)
        return arr[0] if arr.ndim == 3 and arr.shape[0] == 1 else np.squeeze(arr)

    def prepend_last_context(context, arr, n_context, n_arr):
        if n_context > 0 and n_arr > 0:
            arr = np.concatenate((context[:, -1][:, None], arr), axis=1)
            ts = np.arange(n_context - 1, n_context + n_arr)
        else:
            ts = np.arange(n_context, n_context + n_arr)
        return arr, ts

    context = to_2d(context)
    future_vals = to_2d(future_vals)
    predictions = np.asarray(predictions)

    assert context.ndim == 2 and future_vals.ndim == 2 and context.shape[0] >= 3

    n_context, n_future = context.shape[1], future_vals.shape[1]
    predictions_is_samples = predictions.ndim == 3
    n_pred = predictions.shape[1] if not predictions_is_samples else predictions.shape[2]

    future_vals, future_ts = prepend_last_context(context, future_vals, n_context, n_future)

    if predictions_is_samples:
        context_last = context[:, -1][None, :, None]
        predictions_with_context = np.concatenate(
            (np.tile(context_last, (predictions.shape[0], 1, 1)), predictions), axis=2
        )
        pred_ts = np.arange(n_context - 1, n_context + n_pred)
        median_pred = np.median(predictions_with_context, axis=0)
    else:
        predictions_with_context, pred_ts = prepend_last_context(context, predictions, n_context, n_pred)

    # Setup plot
    fig, axes = plt.subplots(4, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1, 1, 1]})
    axes[0].remove()
    ax3d = fig.add_subplot(4, 1, 1, projection="3d")

    # 3D plot
    ax3d.plot(context[0], context[1], context[2], color="black", alpha=0.5, linewidth=1, label="Context")
    ax3d.plot(
        future_vals[0], future_vals[1], future_vals[2], color="black", linestyle="-", linewidth=1, label="Ground Truth"
    )

    cmap = plt.get_cmap(colormap)
    color_for_median, sample_color = cmap(0.9), cmap(0.5)

    if predictions_is_samples:
        for s in range(predictions_with_context.shape[0]):
            ax3d.plot(
                predictions_with_context[s, 0],
                predictions_with_context[s, 1],
                predictions_with_context[s, 2],
                color=sample_color,
                alpha=0.4,
                linewidth=0.5,
                zorder=1,
            )
        ax3d.plot(
            median_pred[0],
            median_pred[1],
            median_pred[2],
            color=color_for_median,
            linewidth=1,
            label="Prediction (median)",
            zorder=2,
        )
    else:
        ax3d.plot(
            predictions_with_context[0],
            predictions_with_context[1],
            predictions_with_context[2],
            color=color_for_median,
            linewidth=1,
            label="Prediction",
        )

    ax3d.grid(False)
    if title:
        ax3d.set_title(title, fontweight="bold", **title_kwargs)

    # Univariate plots
    context_ts = np.arange(n_context)
    for i, ax in enumerate(axes[1:]):
        ax.plot(context_ts, context[i], color="black", alpha=0.5, linewidth=1)
        ax.plot(future_ts, future_vals[i], color="black", linestyle="-", linewidth=1)

        if predictions_is_samples:
            for s in range(predictions_with_context.shape[0]):
                ax.plot(pred_ts, predictions_with_context[s, i], color=sample_color, alpha=0.5, linewidth=0.5, zorder=1)
            ax.plot(pred_ts, median_pred[i], color=color_for_median, linewidth=1, label="Prediction (median)", zorder=2)
        else:
            ax.plot(pred_ts, predictions_with_context[i], color=color_for_median, linewidth=1, label="Prediction")

        ax.set_ylabel(f"Dimension {i}", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("auto")

    axes[-1].set_xlabel("Timestep", fontweight="bold")
    _save_plot(save_path)
    plt.show()


def plot_3d_and_univariate(
    coords: np.ndarray | torch.Tensor,
    figsize: tuple[int, int] = (6, 8),
    custom_colors: list[str] | None = None,
    plot_kwargs: dict[str, Any] | None = None,
    plot_title: str | None = None,
    title_kwargs: dict[str, Any] | None = None,
    save_path: str | None = None,
) -> None:
    """Create a combined 3D and univariate visualization of coordinate data."""
    if coords.ndim == 2:
        coords = coords[None, :, :]
    assert coords.ndim == 3, "coords must have shape (n_samples, n_dimensions, n_time_steps)"
    assert coords.shape[1] >= 3, "At least 3 dimensions are required for 3D plot"

    num_samples, _, num_time_steps = coords.shape
    fig, axes = plt.subplots(4, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1, 1, 1]})
    axes[0].remove()
    ax1 = fig.add_subplot(4, 1, 1, projection="3d")

    custom_colors = custom_colors or DEFAULT_COLORS
    plot_kwargs = plot_kwargs or {}
    title_kwargs = title_kwargs or {}
    time_steps = np.arange(num_time_steps)

    for i in range(num_samples):
        coord_sample = coords[i]
        color = custom_colors[i % len(custom_colors)]  # type: ignore

        # 3D plot
        ax1.plot(coord_sample[0], coord_sample[1], coord_sample[2], color=color, **plot_kwargs)
        ax1.grid(False)

        # Univariate plots
        for j, ax in enumerate(axes[1:]):
            ax.plot(time_steps, coord_sample[j, :], color=color, **plot_kwargs)
            ax.set_ylabel(f"Dimension {j}", fontweight="bold")
            ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Timestep", fontweight="bold")
    if plot_title is not None:
        ax1.set_title(f"{plot_title}", **title_kwargs)

    _save_plot(save_path)
    plt.show()


def plot_multi_3d_and_univariate(
    coords_dict: dict[str, np.ndarray | torch.Tensor],
    colors: dict[str, str],
    show_legend: bool = True,
    use_colorbar_legend: bool = False,
    figsize: tuple[float, float] = (6, 8),
) -> None:
    """Create a combined 3D and univariate visualization of coordinate data."""
    assert coords_dict.keys() == colors.keys(), "Keys in coords_dict and colors must match"

    fig, axes = plt.subplots(4, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1, 1, 1]})
    axes[0].remove()
    ax1 = fig.add_subplot(4, 1, 1, projection="3d")

    # 3D plot
    for key, coords in coords_dict.items():
        ax1.plot(coords[0], coords[1], coords[2], alpha=0.8, color=colors[key], label=key)
    ax1.grid(False)

    if show_legend:
        if use_colorbar_legend:
            cmap = ListedColormap(list(colors.values()))
            norm = plt.Normalize(0, len(colors) - 1)  # type: ignore
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = plt.colorbar(sm, ax=ax1, ticks=range(len(colors)), shrink=0.8, pad=0.15)
            cbar.set_ticklabels(list(colors.keys()))
        else:
            ax1.legend()

    # Univariate plots
    for key, coords in coords_dict.items():
        time_steps = np.arange(coords.shape[1])
        for i, ax in enumerate(axes[1:]):
            ax.plot(time_steps, coords[i], alpha=0.8, color=colors[key], label=key)
            ax.grid(True, alpha=0.3)
            ax.set_ylabel(f"Dimension {i}", fontweight="bold")
    axes[-1].set_xlabel("Time Steps", fontweight="bold")


def plot_single_p_prediction(
    pred_data_dict: dict[float, dict[str, np.ndarray]],
    selected_p: float = 0.0,
    dim_to_plot: int = 0,
    plot_median: bool = True,
    colors: list[str] | None = None,
    save_path: str | None = None,
):
    """Visualize context, ground truth, and model predictions for a single edit probability p."""
    colors = colors or DEFAULT_COLORS
    required_keys = {"context", "future_vals", "median_predictions", "corrected_timesteps", "edit_prob"}

    pred_data = pred_data_dict[selected_p]
    missing_keys = required_keys - set(pred_data.keys())
    if missing_keys:
        raise ValueError(f"pred_data_dict is missing required keys: {missing_keys}")

    context, future_vals = pred_data["context"], pred_data["future_vals"]
    predictions = (
        pred_data["median_predictions"][dim_to_plot].reshape(1, -1)
        if plot_median
        else pred_data["predictions"][:, dim_to_plot, :]
    )
    corrected_timesteps = pred_data["corrected_timesteps"]

    assert predictions.shape[-1] == future_vals.shape[-1], "Prediction length mismatch!"

    context_length = context.shape[-1]
    future_indices = np.arange(context_length, context_length + future_vals.shape[-1])

    fig, ax = plt.subplots(figsize=(8, 2))

    # Plot context and ground truth
    ax.plot(np.arange(context_length), context[dim_to_plot], color="black", label="Context")
    ax.plot(future_indices, future_vals[dim_to_plot], color="black", linestyle="--", label="Ground Truth", zorder=-1)

    # Plot predictions
    for i, pred_sample in enumerate(predictions):
        ax.plot(future_indices, pred_sample, label=f"p={selected_p:.2f}", color=colors[i])

    # Plot correction points if p > 0
    if selected_p > 0.0:
        corrected_indices = np.where(corrected_timesteps)[0]
        corrected_indices = corrected_indices[corrected_indices < predictions.shape[-1]]
        if len(corrected_indices) > 0:
            ax.scatter(
                future_indices[corrected_indices],
                predictions[0, corrected_indices],
                color="black",
                marker="o",
                s=10,
                zorder=10,
            )

    _save_plot(save_path)
    plt.show()


def plot_forecast_and_rmse(
    metrics_dict: dict[float, dict[str, np.ndarray]],
    pred_data_dict: dict[float, dict[str, np.ndarray]],
    dim_to_plot: int = 0,
    cmap: str = "viridis_r",
    figsize: tuple[float, float] = (12, 8),
    save_path: str | None = None,
):
    """Plot forecast trajectories and RMSE vs p for decoder edit experiments."""
    context = pred_data_dict[0.0]["context"]
    future_vals = pred_data_dict[0.0]["future_vals"]

    # Validate data consistency
    for p in pred_data_dict:
        assert np.all(pred_data_dict[p]["context"] == context), f"Context mismatch for p={p}"
        assert np.all(pred_data_dict[p]["future_vals"] == future_vals), f"Future vals mismatch for p={p}"

    context_length = context.shape[-1]
    future_indices = np.arange(context_length, context_length + future_vals.shape[-1])
    p_sorted = np.sort(np.array(list(pred_data_dict.keys())))
    colors = plt.get_cmap(cmap)(np.linspace(0.2, 1.0, len(p_sorted)))

    # Setup grid
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[2, 1], figure=fig)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax_legend = fig.add_subplot(gs[0, 1])
    ax_legend.axis("off")

    # Plot context and ground truth
    (context_line,) = ax0.plot(
        np.arange(context_length), context[dim_to_plot], color="black", label="Context", linewidth=1
    )
    (gt_line,) = ax0.plot(
        future_indices, future_vals[dim_to_plot], color="black", linestyle="--", label="Ground Truth", linewidth=1
    )
    ax0.axvline(x=context_length - 0.5, color="gray", linestyle="--")
    ax0.set_title("Median Forecasts", fontweight="bold")
    ax0.set_xlabel("Time", fontweight="bold")

    # Plot forecasts
    forecast_lines = []
    for i, p in enumerate(p_sorted):
        median_pred = np.median(pred_data_dict[p]["predictions"], axis=0)
        (line,) = ax0.plot(future_indices, median_pred[dim_to_plot], color=colors[i], linewidth=1, alpha=0.8)
        forecast_lines.append(line)

        # Plot correction points for last p value
        if i == len(p_sorted) - 1:
            corrected_idx = np.where(pred_data_dict[p]["corrected_timesteps"])[0]
            corrected_idx = corrected_idx[corrected_idx < len(median_pred[dim_to_plot])]
            if len(corrected_idx):
                ax0.scatter(
                    future_indices[corrected_idx],
                    median_pred[dim_to_plot][corrected_idx],
                    color=colors[i],
                    marker="o",
                    facecolors="none",
                    edgecolors=colors[i],
                    s=10,
                    linewidths=1,
                )

    # Legend
    legend_handles = [context_line, gt_line] + forecast_lines
    legend_labels = ["Context", "Ground Truth"] + [f"p = {pred_data_dict[p]['edit_prob']:.2f}" for p in p_sorted]
    ax_legend.legend(legend_handles, legend_labels, loc="center left", frameon=True, ncols=1)

    # RMSE plot
    rmse_vals = np.array([metrics_dict[p]["rmse"] for p in p_sorted])
    ax1.plot(p_sorted, rmse_vals, color="tab:blue", marker="o", markersize=4, linewidth=1)
    ax1.set_title("RMSE (Excluding Corrected Points)", fontweight="bold")
    ax1.set_xlabel("Correction Probability (p)", fontweight="bold")
    ax1.set_ylabel("RMSE", fontweight="bold")

    _save_plot(save_path)
    plt.show()


###### Plot ablations ######


def _create_custom_boxplot(ax, data_groups, x_labels, colors, percentile_range, whisker_range, show_mean_line):
    """Create custom box plot with specified percentile ranges."""
    positions = np.arange(len(x_labels))
    box_elements = {"boxes": [], "medians": [], "means": []}

    for i, (data_group, color) in enumerate(zip(data_groups, colors)):
        if len(data_group) == 0:
            continue

        # Calculate statistics
        lower_p, upper_p = percentile_range
        q1, q3 = np.percentile(data_group, [lower_p, upper_p])
        median, mean = np.median(data_group), np.mean(data_group)

        whisker_low = np.percentile(data_group, whisker_range[0])
        whisker_high = np.percentile(data_group, whisker_range[1])

        # Draw box
        box_height = q3 - q1
        box = plt.Rectangle(  # type: ignore
            (positions[i] - 0.3, q1), 0.6, box_height, facecolor=color, alpha=0.7, edgecolor="black", linewidth=0.5
        )
        ax.add_patch(box)

        # Draw median line
        (median_line,) = ax.plot([positions[i] - 0.3, positions[i] + 0.3], [median, median], color="black", linewidth=2)
        box_elements["medians"].append(median_line)

        # Draw mean line if requested
        if show_mean_line:
            (mean_line,) = ax.plot(
                [positions[i] - 0.3, positions[i] + 0.3],
                [mean, mean],
                color="black",
                linewidth=1,
                linestyle="--",
                alpha=0.8,
            )
            box_elements["means"].append(mean_line)

        # Draw whiskers and caps
        for start, end in [(q1, whisker_low), (q3, whisker_high)]:
            ax.plot([positions[i], positions[i]], [start, end], color="black", linewidth=1)
            ax.plot([positions[i] - 0.1, positions[i] + 0.1], [end, end], color="black", linewidth=1)

    return box_elements


def plot_ablation_metrics_distributional(
    data: dict[str, dict[str, dict[str, dict[str, float | list[float]]]]],
    metric_name: str,
    prediction_horizon: str,
    figsize: tuple[int, int] = (8, 5),
    title: str | None = None,
    save_path: str | None = None,
    plot_type: Literal["violin", "box"] = "violin",
    box_plot_kwargs: dict[str, Any] | None = None,
    show_mean_line: bool = True,
    cmap: str = "Blues",
    cmap_range: tuple[float, float] = (0.1, 0.9),
    percentile_range: tuple[float, float] = (25, 75),
    whisker_range: tuple[float, float] = (10, 90),
    tick_rotation: int = 45,
    verbose: bool = False,
    hide_indices: list[int] | None = None,
    exclude_outliers_iqr: float | None = None,
    show_median_trend: bool = False,
) -> None:
    """Plot violin or box plots of ablation metrics across different layer ablations.

    Args:
        whisker_range: Percentile range for caps/whiskers (default (10, 90)).
            For violin plots: data outside this range is excluded (outliers clipped).
            For box plots: determines whisker extent.
        hide_indices: List of x-axis indices (0-based) to exclude from the plot entirely.
        exclude_outliers_iqr: If set, exclude outliers using IQR method before applying whisker_range.
            Values outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR] are removed.
            Common values: 1.5 (standard), 3.0 (extreme outliers only).
        show_median_trend: If True, draw a red line connecting the medians across groups.
    """
    _, metric_name_title = _get_metric_transform(metric_name)
    layer_group_metrics, x_labels = get_layergroup_metrics(data, metric_name, prediction_horizon)

    # Filter hidden indices
    if hide_indices:
        keep = [i for i in range(len(layer_group_metrics)) if i not in set(hide_indices)]
        layer_group_metrics = [layer_group_metrics[i] for i in keep]
        x_labels = [x_labels[i] for i in keep]

    # Exclude IQR outliers
    if exclude_outliers_iqr is not None:
        layer_group_metrics = [
            arr[
                (arr >= (q1 := np.percentile(arr, 25)) - exclude_outliers_iqr * (iqr := np.percentile(arr, 75) - q1))
                & (arr <= q1 + iqr + exclude_outliers_iqr * iqr)
            ]
            for arr in layer_group_metrics
        ]

    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(x_labels))
    colors = _get_colors(cmap, len(layer_group_metrics), cmap_range)

    def style_bodies(elements, fill_colors):
        for elem, color in zip(elements, fill_colors):
            elem.set_facecolor(color)
            elem.set_alpha(0.7)
            elem.set_edgecolor("black")
            elem.set_linewidth(0.5)

    def style_lines(collection, color="black", linewidth=1):
        """Style a LineCollection or list of lines."""
        if collection is None:
            return
        try:
            for line in collection:
                line.set_color(color)
                line.set_linewidth(linewidth)
        except TypeError:
            collection.set_color(color)
            collection.set_linewidth(linewidth)

    # Data used for plotting (and for median trend calculation)
    plot_data = layer_group_metrics

    if plot_type == "violin":
        # Clip to whisker_range for violin plots
        plot_data = [
            arr[(arr >= np.percentile(arr, whisker_range[0])) & (arr <= np.percentile(arr, whisker_range[1]))]
            for arr in layer_group_metrics
        ]

        parts = ax.violinplot(plot_data, positions=positions, showmeans=True, showmedians=True, widths=0.8)
        style_bodies(parts["bodies"], colors)

        # Style all line elements
        for key in ["cbars", "cmins", "cmaxes"]:
            style_lines(parts.get(key))

        if parts.get("cmeans"):
            if show_mean_line:
                style_lines(parts["cmeans"], linewidth=1)
                parts["cmeans"].set_linestyle("--")
                parts["cmeans"].set_alpha(0.8)
            else:
                parts["cmeans"].set_alpha(0.0)

        if parts.get("cmedians"):
            style_lines(parts["cmedians"], linewidth=2)

    elif plot_type == "box":
        if percentile_range != (25, 75):
            box = _create_custom_boxplot(
                ax, layer_group_metrics, x_labels, colors, percentile_range, whisker_range, show_mean_line
            )
        else:
            box = ax.boxplot(
                layer_group_metrics,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                showmeans=True,
                meanline=True,
                medianprops=dict(color="black", linewidth=2),
                meanprops=dict(color="black", linewidth=1, linestyle="--") if show_mean_line else {},
                **{**(box_plot_kwargs or {}), "whis": whisker_range, "showfliers": False},
            )
            style_bodies(box["boxes"], colors)

        for mean in box.get("means", []):
            if not show_mean_line:
                mean.set_alpha(0.0)
        for median in box.get("medians", []):
            median.set_color("black")
            median.set_linewidth(2)
    else:
        raise ValueError(f"Unknown plot_type '{plot_type}'. Use 'violin' or 'box'.")

    # Draw median trend line (uses the same data that's displayed in the plot)
    if show_median_trend:
        medians = [np.median(arr) for arr in plot_data]
        ax.plot(positions, medians, color="red", linewidth=3, zorder=10, marker="o", markersize=4, alpha=0.8)

    _setup_plot_formatting(
        ax,
        "Ablated Layers",
        metric_name_title,
        title or f"{metric_name_title} Distribution vs Ablated Layers (Horizon: {prediction_horizon})",
        x_labels=x_labels,
        rotation=tick_rotation,
    )
    _save_plot(save_path)

    if verbose:
        for label, arr in zip(x_labels, plot_data):
            median = np.median(arr)
            mean = np.mean(arr)
            std = np.std(arr)
            p5 = np.percentile(arr, 5)
            p25 = np.percentile(arr, 25)
            p75 = np.percentile(arr, 75)
            p95 = np.percentile(arr, 95)
            print(
                f"{label}: Median = {median:.2f}, Mean = {mean:.2f}, Std = {std:.2f}, "
                f"P5 = {p5:.2f}, P25 = {p25:.2f}, P75 = {p75:.2f}, P95 = {p95:.2f}"
            )


def plot_ablation_metrics_lines_over_horizon(
    data: dict[str, dict[str, dict[str, dict[str, float | list[float]]]]],
    metric_name: str,
    prediction_horizons: list[str],
    figsize: tuple[int, int] = (10, 6),
    percentile_envelope: tuple[float, float] = (25, 75),
    title: str | None = None,
    save_path: str | None = None,
    layergroup_indices_to_plot: list[int] | None = None,
    cmap: str = "Blues",
    cmap_range: tuple[float, float] = (0.1, 0.9),
    legend_kwargs: dict[str, Any] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    envelope_type: Literal["percentile", "mad"] = "percentile",
    mad_multiplier: float = 1.0,
) -> None:
    """Plot lines for each ablation group with x-axis as prediction horizon.

    Args:
        envelope_type: Type of envelope to plot. Either "percentile" or "mad".
            - "percentile": Use percentile_envelope parameter for lower/upper bounds
            - "mad": Use mean absolute deviation (MAD) about the median
        mad_multiplier: Multiplier for MAD envelope width (only used if envelope_type="mad").
            The envelope will be median ± (mad_multiplier * MAD).
    """
    _, metric_name_title = _get_metric_transform(metric_name)

    # Get ablation groups
    layer_info, ablation_labels = _get_layer_info_and_labels(data, None)
    horizon_ints = sorted([int(h) for h in prediction_horizons])
    horizon_strs = [str(h) for h in horizon_ints]

    # Collect stats for each ablation group
    ablation_group_stats = []
    for _, _, ablation_key in layer_info:
        group_medians, group_p_low, group_p_high = [], [], []
        for horizon in horizon_strs:
            horizon_data = data[ablation_key].get(horizon, {})
            arr = _process_metric_data(horizon_data, metric_name)

            if arr.size == 0:
                group_medians.append(np.nan)
                group_p_low.append(np.nan)
                group_p_high.append(np.nan)
            else:
                median = np.median(arr)
                group_medians.append(median)

                if envelope_type == "mad":
                    # Mean Absolute Deviation (MAD) about the median
                    mad = np.mean(np.abs(arr - median))
                    group_p_low.append(median - mad_multiplier * mad)
                    group_p_high.append(median + mad_multiplier * mad)
                else:  # percentile
                    group_p_low.append(np.percentile(arr, percentile_envelope[0]))
                    group_p_high.append(np.percentile(arr, percentile_envelope[1]))
        ablation_group_stats.append((group_medians, group_p_low, group_p_high))

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    colors = _get_colors(cmap, len(layer_info), cmap_range) * 0.95

    for i, (label, (medians, p_low, p_high), color) in enumerate(zip(ablation_labels, ablation_group_stats, colors)):
        if layergroup_indices_to_plot is not None and i not in layergroup_indices_to_plot:
            continue
        medians, p_low, p_high = map(np.array, [medians, p_low, p_high])
        if not np.all(np.isnan(medians)):
            ax.plot(
                horizon_ints,
                medians,
                label=label,
                color=color,
                marker="o",
                linewidth=2,
                markeredgecolor="gray",
                markeredgewidth=0.5,
            )
            ax.fill_between(horizon_ints, p_low, p_high, color=color, alpha=0.08, zorder=-i)

    ax.set_ylim(vmin, vmax)
    _setup_plot_formatting(
        ax,
        "Prediction Horizon",
        metric_name_title,
        title or f"{metric_name_title} vs Prediction Horizon for Each Ablated Layer Group",
    )
    ax.set_xticks(horizon_ints)
    ax.set_xticklabels([str(h) for h in horizon_ints])

    legend_kwargs = legend_kwargs or {
        "title": "Ablated Layers Group",
        "loc": "best",
        "frameon": True,
        "framealpha": 0.9,
    }
    ax.legend(**legend_kwargs)
    _save_plot(save_path)


def plot_ablation_metrics(
    data: dict[str, dict[str, dict[str, dict[str, float | list[float]]]]],
    metric_name: str,
    prediction_horizon: str,
    figsize: tuple[int, int] = (5, 5),
    title: str | None = None,
    save_path: str | None = None,
    aggregation: Literal["median", "mean"] = "median",
    show_envelope: bool = True,
    percentile_range: tuple[float, float] = (25, 75),
    show_mad: bool = False,
    mad_scale: float = 1.0,
    verbose: bool = False,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Plot ablation metrics across different layer ablations."""
    _, metric_name_title = _get_metric_transform(metric_name)

    # Filter and sort data
    horizon_data = {k: v[prediction_horizon] for k, v in data.items() if prediction_horizon in v}
    if not horizon_data:
        raise ValueError(f"No data found for prediction horizon '{prediction_horizon}'")

    layer_info, x_labels = _get_layer_info_and_labels(horizon_data)

    # Collect and process data
    layer_group_values = [_process_metric_data(horizon_data[layer_key], metric_name) for _, _, layer_key in layer_info]

    # Calculate aggregates
    aggregates = [_compute_aggregates(values, aggregation, percentile_range) for values in layer_group_values]
    aggregate_values, std_errors, semi_interquartile_range, mad = map(np.array, zip(*aggregates))
    mad = mad * mad_scale
    x_pos = np.arange(len(layer_group_values))

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_pos, aggregate_values, marker="o", linewidth=2, markersize=5, color="black")
    ax.set_ylim(ylim)

    # Add envelope if requested
    if show_envelope:
        envelope_values = semi_interquartile_range if aggregation == "median" else std_errors
        if show_mad:
            envelope_values = mad
        ax.fill_between(
            x_pos, aggregate_values - envelope_values, aggregate_values + envelope_values, color="black", alpha=0.08
        )

    _setup_plot_formatting(
        ax,
        "Ablated Layers Group",
        metric_name_title,
        title or f"{metric_name_title} ($L_{{\mathrm{{pred}}}} = {int(prediction_horizon)}$)",
        x_labels,
    )
    _save_plot(save_path)

    if verbose:
        print(f"{aggregation} of {metric_name} at horizon {prediction_horizon}: {aggregate_values}")


def plot_ablation_metrics_lines_over_layergroup(
    data: dict[str, dict[str, dict[str, dict[str, float | list[float]]]]],
    metric_name: str,
    prediction_horizons: list[str],
    figsize: tuple[int, int] = (10, 6),
    percentile_envelope: tuple[float, float] = (25, 75),
    title: str | None = None,
    save_path: str | None = None,
    cmap: str = "Blues",
    cmap_range: tuple[float, float] = (0.2, 0.9),
    legend_kwargs: dict[str, Any] | None = None,
    marker: str = "o",
) -> None:
    """Plot lines for each prediction horizon with x-axis as ablation group."""
    _, metric_name_title = _get_metric_transform(metric_name)

    # Get ablation groups
    layer_info, ablation_labels = _get_layer_info_and_labels(data)
    x = np.arange(len(layer_info))

    horizon_ints = sorted([int(h) for h in prediction_horizons])
    horizon_strs = [str(h) for h in horizon_ints]

    # Collect stats for each horizon
    horizon_stats = []
    for horizon in horizon_strs:
        medians, p_low, p_high = [], [], []
        for _, _, ablation_key in layer_info:
            horizon_data = data[ablation_key].get(horizon, {})
            arr = _process_metric_data(horizon_data, metric_name)

            if arr.size == 0:
                medians.append(np.nan)
                p_low.append(np.nan)
                p_high.append(np.nan)
            else:
                medians.append(np.median(arr))
                p_low.append(np.percentile(arr, percentile_envelope[0]))
                p_high.append(np.percentile(arr, percentile_envelope[1]))
        horizon_stats.append((medians, p_low, p_high))

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    colors = _get_colors(cmap, len(horizon_strs), cmap_range)

    for i, (horizon, (medians, p_low, p_high), color) in enumerate(zip(horizon_strs, horizon_stats, colors)):
        medians, p_low, p_high = map(np.array, [medians, p_low, p_high])
        if not np.all(np.isnan(medians)):
            ax.plot(
                x,
                medians,
                label=str(horizon),
                color=color,
                marker=marker,
                markersize=6 if marker != "*" else 10,
                linewidth=2,
            )
            ax.fill_between(x, p_low, p_high, color=color, alpha=0.08, zorder=-i)

    _setup_plot_formatting(ax, "Ablated Layers Group", metric_name_title, title, ablation_labels)
    legend_kwargs = legend_kwargs or {
        "title": r"Prediction Horizon ($L_{\mathrm{pred}}$)",
        "loc": "best",
        "frameon": True,
        "framealpha": 0.9,
    }
    ax.legend(**legend_kwargs)
    _save_plot(save_path)


def plot_ablation_metrics_lines_over_layergroup_combined_v2(
    data_combined: dict[str, dict[str, dict[str, dict[str, dict[str, float | list[float]]]]]],
    metric_name: str,
    prediction_horizons: list[str],
    figsize: tuple[int, int] = (10, 6),
    percentile_envelope: tuple[float, float] = (25, 75),
    title: str | None = None,
    save_path: str | None = None,
    cmap: str = "Blues",
    cmap_range: tuple[float, float] = (0.2, 0.9),
    legend_kwargs: dict | None = None,  # type: ignore
    markers: list[str] | None = None,
    linestyles: list[str] | None = None,
    tick_rotation: int = 45,
    verbose: bool = False,
) -> None:
    """Plot ablation metrics with lines for each prediction horizon and component."""
    _, metric_name_title = _get_metric_transform(metric_name)

    # Setup styles and data structures
    markers = markers or ["o", "s", "D", "P"]
    linestyles = linestyles or ["-", "--", "-.", ":"]
    component_styles = {
        k: {"marker": markers[i % len(markers)], "linestyle": linestyles[i % len(linestyles)]}
        for i, k in enumerate(data_combined.keys())
    }

    # Get unique layer groups
    all_ablation_keys = set().union(*(comp_data.keys() for comp_data in data_combined.values()))
    ablation_key_to_layers = {k: tuple(_parse_layer_ablation_key(k)[1]) for k in all_ablation_keys}
    sorted_layer_tuples = sorted(set(ablation_key_to_layers.values()), key=lambda x: x[0])
    ablation_labels = [
        f"{layers[0]}" if len(layers) == 1 else f"{layers[0]}-{layers[-1]}" for layers in sorted_layer_tuples
    ]
    x = np.arange(len(sorted_layer_tuples))

    # Setup horizons and colors
    horizon_strs = [str(h) for h in sorted([int(h) for h in prediction_horizons])]
    base_colors = _get_colors(cmap, len(horizon_strs), cmap_range)

    fig, ax = plt.subplots(figsize=figsize)
    color_handles, color_labels = [], []
    comp_handles, comp_labels = [], []

    # Plot data
    for components_str, comp_data in data_combined.items():
        style = component_styles[components_str]

        for i, horizon in enumerate(horizon_strs):
            # Collect data for all layer groups
            stats_data = []
            for layers in sorted_layer_tuples:
                ablation_key = next((k for k, _ in comp_data.items() if ablation_key_to_layers.get(k) == layers), None)

                if ablation_key:
                    horizon_data = comp_data.get(ablation_key, {}).get(horizon, {})
                    arr = _process_metric_data(horizon_data, metric_name)
                    stats_data.append(_compute_stats(arr, percentile_envelope))
                else:
                    stats_data.append((np.nan, np.nan, np.nan))

            # Convert to arrays and plot
            medians_array, p_low_array, p_high_array = map(np.array, zip(*stats_data))
            valid_mask = ~np.isnan(medians_array)

            if verbose:
                print(
                    f"median of {metric_name} for {components_str} at horizons {prediction_horizons}: {medians_array}"
                )

            if np.any(valid_mask):
                ax.plot(
                    x[valid_mask],
                    medians_array[valid_mask],
                    color=base_colors[i],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    linewidth=2,
                    markersize=6 if style["marker"] != "*" else 10,
                    markeredgewidth=1.0,
                    markeredgecolor="gray",
                    alpha=0.8,
                )
                ax.fill_between(
                    x[valid_mask],
                    p_low_array[valid_mask],
                    p_high_array[valid_mask],
                    color=base_colors[i],
                    alpha=0.08,
                    zorder=-i,
                )

        # Add component to legend
        comp_handles.append(
            Line2D(
                [0],
                [0],
                color="gray",
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1,
                markersize=4.5 if style["marker"] != "*" else 7.5,
                alpha=0.8,
            )
        )
        comp_labels.append(components_str)

    # Add horizon handles to legend
    color_handles = [
        Line2D([0], [0], color=base_colors[i], linestyle="-", linewidth=2) for i in range(len(horizon_strs))
    ]
    color_labels = [f"$L_{{\mathrm{{pred}}}} = {horizon}$" for horizon in horizon_strs]

    # Formatting
    _setup_plot_formatting(
        ax, "Ablated Layers Group", metric_name_title, title, x_labels=ablation_labels, rotation=tick_rotation
    )

    # Legends
    legend_kwargs = legend_kwargs or {"frameon": True, "framealpha": 0.9}

    # First create the right legend to get its position
    leg2 = ax.legend(
        comp_handles,
        comp_labels,
        title="Ablated Components",
        loc="upper right",
        fontsize=8,
        title_fontsize=8,
        **legend_kwargs,
    )

    # Position the left legend to the left of the right legend
    # Get the bounding box of leg2 in figure coordinates
    fig = plt.gcf()
    fig.canvas.draw()
    leg2_bbox = leg2.get_window_extent().transformed(fig.transFigure.inverted())

    # Create leg1 and position it to the left of leg2
    leg1 = ax.legend(
        color_handles,
        color_labels,
        title=r"Prediction Horizon ($L_{\mathrm{pred}}$)",
        loc="upper right",
        bbox_to_anchor=(leg2_bbox.x0 - 0.01, 1.0),
        fontsize=8,
        title_fontsize=8,
        **legend_kwargs,
    )
    ax.add_artist(leg1)
    ax.add_artist(leg2)

    _save_plot(save_path)


def plot_ablation_metrics_lines_over_layergroup_combined(
    data_combined: dict[str, dict[str, dict[str, dict[str, dict[str, float | list[float]]]]]],
    metric_name: str,
    prediction_horizon: str,
    figsize: tuple[int, int] = (10, 6),
    percentile_envelope: tuple[float, float] = (25, 75),
    title: str | None = None,
    save_path: str | None = None,
    colors: list[str] | None = None,
    legend_kwargs: dict | None = None,  # type: ignore
    markers: list[str] | None = None,
    linestyles: list[str] | None = None,
    tick_rotation: int = 45,
    verbose: bool = False,
) -> None:
    """Plot ablation metrics with lines for each prediction horizon and component."""
    _, metric_name_title = _get_metric_transform(metric_name)

    # Setup styles and data structures
    markers = markers or ["o", "s", "D", "P"]
    linestyles = linestyles or ["-", "--", "-.", ":"]
    component_styles = {
        k: {"marker": markers[i % len(markers)], "linestyle": linestyles[i % len(linestyles)]}
        for i, k in enumerate(data_combined.keys())
    }

    # Get unique layer groups
    all_ablation_keys = set().union(*(comp_data.keys() for comp_data in data_combined.values()))
    ablation_key_to_layers = {k: tuple(_parse_layer_ablation_key(k)[1]) for k in all_ablation_keys}
    sorted_layer_tuples = sorted(set(ablation_key_to_layers.values()), key=lambda x: x[0])
    ablation_labels = [
        f"{layers[0]}" if len(layers) == 1 else f"{layers[0]}-{layers[-1]}" for layers in sorted_layer_tuples
    ]
    x = np.arange(len(sorted_layer_tuples))

    colors = colors or DEFAULT_COLORS

    fig, ax = plt.subplots(figsize=figsize)
    color_handles, color_labels = [], []
    comp_handles, comp_labels = [], []

    # Plot data
    for i, (components_str, comp_data) in enumerate(data_combined.items()):
        style = component_styles[components_str]

        # Collect data for all layer groups
        stats_data = []
        for layers in sorted_layer_tuples:
            ablation_key = next((k for k, _ in comp_data.items() if ablation_key_to_layers.get(k) == layers), None)

            if ablation_key:
                horizon_data = comp_data.get(ablation_key, {}).get(prediction_horizon, {})
                arr = _process_metric_data(horizon_data, metric_name)
                stats_data.append(_compute_stats(arr, percentile_envelope))
            else:
                stats_data.append((np.nan, np.nan, np.nan))

        # Convert to arrays and plot
        medians_array, p_low_array, p_high_array = map(np.array, zip(*stats_data))
        valid_mask = ~np.isnan(medians_array)

        if verbose:
            print(f"median of {metric_name} for {components_str} at horizon {prediction_horizon}: {medians_array}")

        if np.any(valid_mask):
            ax.plot(
                x[valid_mask],
                medians_array[valid_mask],
                color=colors[i],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=6 if style["marker"] != "*" else 10,
                markerfacecolor="none",
                markeredgewidth=2.0,
                alpha=0.8,
            )
            ax.fill_between(
                x[valid_mask],
                p_low_array[valid_mask],
                p_high_array[valid_mask],
                color=colors[i],
                alpha=0.08,
                zorder=-i,
            )

        # Add component to legend
        comp_handles.append(
            Line2D(
                [0],
                [0],
                color=colors[i],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=6 if style["marker"] != "*" else 10,
                markerfacecolor="none",
                markeredgewidth=2.0,
                alpha=0.8,
            )
        )
        comp_labels.append(components_str)

    # Formatting
    _setup_plot_formatting(
        ax, "Ablated Layer", metric_name_title, title, x_labels=ablation_labels, rotation=tick_rotation
    )

    # Legends
    legend_kwargs = legend_kwargs or {"frameon": True, "framealpha": 0.9}

    ax.legend(
        comp_handles,
        comp_labels,
        title="Ablated Components",
        loc="upper right",
        fontsize=8,
        title_fontsize=8,
        **legend_kwargs,
    )

    _save_plot(save_path)


def plot_sorted_datasets(
    datasets_sorted: dict,
    description: str,
    n_datasets: int = 12,
    sort_reverse: bool = False,
    save_path: str | None = None,
    threshold_multiplier_upper: float = 1.2,
    threshold_multiplier_lower: float = 0.8,
    ablation_level_meaning_alternative_str: str = "Ablation Level",
    metric_pretty_name: str = "Metric",
    model_name: str = "Model",
    selected_metric: str = "selected_metric",
):
    """Plot bar charts showing performance across ablation levels for selected datasets."""
    N_DATASETS_TO_PLOT = min(n_datasets, len(datasets_sorted))

    if N_DATASETS_TO_PLOT == 0:
        print(f"No datasets found within {threshold_multiplier_upper}x of baseline at max ablation level.")
        return

    # Create subplots
    n_cols = 3
    n_rows = (N_DATASETS_TO_PLOT + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten() if N_DATASETS_TO_PLOT > 1 else [axes]

    sorted_datasets = sorted(datasets_sorted.items(), key=lambda x: x[1]["relative_performance"], reverse=sort_reverse)

    for idx, (dataset, info) in enumerate(sorted_datasets[:N_DATASETS_TO_PLOT]):
        ax = axes[idx]
        data = info["all_data"].sort_values("ablation_level")
        baseline = info["baseline_value"]

        # Bar plot with color coding
        colors = ["green" if level == 0 else "tab:blue" for level in data["ablation_level"]]
        ax.bar(data["ablation_level"], data[selected_metric], color=colors, alpha=0.3, edgecolor="black")

        # Reference lines
        ax.axhline(y=baseline, color="red", linestyle="--", linewidth=2, label="Baseline", alpha=0.7)
        ax.axhline(
            y=baseline * threshold_multiplier_upper,
            color="red",
            linestyle=":",
            linewidth=2,
            label=f"{threshold_multiplier_upper}x Baseline",
            alpha=1.0,
        )
        ax.axhline(
            y=baseline * threshold_multiplier_lower,
            color="red",
            linestyle=":",
            linewidth=2,
            label=f"{threshold_multiplier_lower}x Baseline",
            alpha=1.0,
            zorder=5,
        )

        # Formatting
        ax.set_xlabel(ablation_level_meaning_alternative_str, fontsize=9)
        ax.set_ylabel(metric_pretty_name, fontsize=9)
        short_name = dataset if len(dataset) <= 35 else dataset[:32] + "..."
        ax.set_title(f"{short_name}\nRel: {info['relative_performance']:.3f}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y", which="both")
        ax.grid(False, axis="x")
        ax.set_xticks(sorted(data["ablation_level"].unique()))

        if idx == 0:
            ax.legend(fontsize=8, loc="best", framealpha=1.0)

    # Hide unused subplots
    for idx in range(N_DATASETS_TO_PLOT, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(
        f"{description}: {metric_pretty_name} Across Ablation Levels ({model_name})\n",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved bar plot to: {save_path}")
    plt.show()


# # save_path = os.path.join(figs_save_dir, f"{ablation_meaning_str}_vulnerable_datasets_bar_plot_{model_name}.pdf")
# save_path = None
# plot_sorted_datasets(
#     datasets_vulnerable,
#     "Most Vulnerable Datasets",
#     n_datasets=12,
#     sort_reverse=True,
#     save_path=save_path if save_figs else None,
# )


def plot_ablation_box_plot(
    box_data,
    ablation_level_meaning_alternative_str: str,
    metric_pretty_name: str,
    ablation_meaning_str: str,
    model_name: str,
    selected_metric: str,
    figs_save_dir: str | None = None,
    save_figs: bool = False,
    figsize: tuple[int, int] = (6, 6),
) -> None:
    """Create box plot showing metric distribution across ablation levels.

    Args:
        box_data: DataFrame containing ablation data with 'ablation_level' column
        ablation_level_meaning_alternative_str: Label for x-axis (ablation level description)
        metric_pretty_name: Label for y-axis (metric name)
        ablation_meaning_str: Description of ablation type for title and filename
        model_name: Name of the model being analyzed
        selected_metric: Column name in box_data to plot
        figs_save_dir: Directory to save figure (required if save_figs=True)
        save_figs: Whether to save the figure
        figsize: Figure size as (width, height)
    """
    from matplotlib.patches import Patch
    from scipy.stats.mstats import gmean

    # Create box plot
    fig, ax = plt.subplots(figsize=figsize)

    # Prepare data for each ablation level
    ablation_levels = sorted(box_data["ablation_level"].unique())
    box_plot_data = [box_data[box_data["ablation_level"] == level][selected_metric].values for level in ablation_levels]

    # Create box plot
    ax.boxplot(
        box_plot_data,
        positions=np.arange(1, len(ablation_levels) + 1),
        patch_artist=True,
        showfliers=False,
        showmeans=False,
        widths=0.6,
        whis=(10, 90),
        medianprops=dict(color="black", linewidth=2),
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    # Add geometric mean markers
    geom_mean_marker = None
    for i, level in enumerate(ablation_levels, start=1):
        level_data = box_data[box_data["ablation_level"] == level][selected_metric].values
        geom_mean = gmean(level_data)
        marker = ax.plot(
            i,
            geom_mean,
            marker="D",
            markerfacecolor="red",
            markersize=8,
            markeredgecolor="darkred",
            markeredgewidth=1,
            zorder=3,
        )
        if i == 1:
            geom_mean_marker = marker[0]

    ax.set_xlabel(ablation_level_meaning_alternative_str, fontweight="bold")
    ax.set_ylabel(metric_pretty_name, fontweight="bold")
    ax.set_title(f"{ablation_meaning_str} ({model_name})", fontweight="bold", pad=20)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(np.arange(1, len(ablation_levels) + 1))
    ax.set_xticklabels(ablation_levels)

    # Add legend
    legend_handles = [
        geom_mean_marker,
        Line2D([0], [0], color="black", linewidth=2),
        Patch(facecolor="lightblue", alpha=0.7),
    ]
    ax.legend(
        handles=legend_handles,
        labels=["Geometric Mean", "Median", "IQR (25th-75th percentile)"],
        loc="upper left",
        frameon=True,
    )

    plt.tight_layout()

    if save_figs and figs_save_dir:
        save_path = os.path.join(figs_save_dir, f"{ablation_meaning_str}_box_plot_{model_name}.pdf")
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved box plot to: {save_path}")

    plt.show()


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
