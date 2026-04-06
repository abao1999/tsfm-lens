import logging
import os
import pickle

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

from tsfm_lens.chronos.circuitlens import CircuitLensChronos

# from chronos import ChronosPipeline
from tsfm_lens.utils.data_utils import load_dyst_samples

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Function to find the indices of the top k highest induction scores
def find_max_induction_score(induction_scores: torch.Tensor, k: int | None = None) -> list[tuple[int, int]]:
    """
        Find the indices of the top k highest induction scores in the matrix.

        Args:
            induction_scores: torch tensor of induction scores
            k: number of top scores to return
    w
        Returns:
            List of (layer, head) tuples corresponding to the top k induction scores
    """
    # Flatten the tensor to find the top k values
    flattened = induction_scores.flatten()

    # Get the indices of the top k values
    if k is None or k > flattened.numel():
        k = flattened.numel()

    top_k_values, top_k_indices = torch.topk(flattened, k)

    # Convert flat indices back to 2D indices (layer, head)
    num_heads = induction_scores.shape[1]
    layer_indices = top_k_indices // num_heads
    head_indices = top_k_indices % num_heads

    # Create list of (layer, head) tuples
    top_k_heads = [(int(layer_indices[i].item()), int(head_indices[i].item())) for i in range(k)]

    return top_k_heads


# Calculate RMSE between predictions and ground truth
def calculate_rmse(predictions: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Calculate Root Mean Squared Error between predictions and ground truth.

    Args:
        predictions: Predicted values
        ground_truth: Ground truth values

    Returns:
        RMSE value
    """
    return float(torch.sqrt(torch.mean(torch.pow(predictions - ground_truth, 2))))


def get_median_prediction(
    input_series: torch.Tensor,
    pipeline: CircuitLensChronos,
    heads_to_ablate: list[tuple[int, int]],
    max_new_tokens: int = 64,
    num_samples: int = 12,
    ablation_method: str = "zero",
    attention_type: str = "ca",
) -> torch.Tensor:
    """
    Generate predictions with specified heads ablated and return the median prediction.

    Args:
        input_series: Input time series data (shape: [batch_size, seq_len])
        pipeline: The model pipeline
        heads_to_ablate: List of (layer_idx, head_idx) tuples to ablate
        max_new_tokens: Maximum number of tokens to generate
        num_samples: Number of samples to generate
        ablation_method: Ablation method ("zero" or "mean")

    Returns:
        Median prediction across samples
    """
    # Install head ablation hooks if requested
    added_hooks = False
    try:
        if len(heads_to_ablate) > 0:
            if attention_type == "both":
                pipeline.add_head_ablation_hooks(heads_to_ablate, ablation_method=ablation_method, attention_type="ca")
                pipeline.add_head_ablation_hooks(heads_to_ablate, ablation_method=ablation_method, attention_type="sa")
            else:
                pipeline.add_head_ablation_hooks(
                    heads_to_ablate,
                    ablation_method=ablation_method,  # type: ignore[arg-type]
                    attention_type=attention_type,  # type: ignore[arg-type]
                )
            added_hooks = True

        # Generate predictions
        predictions = pipeline.predict(  # type: ignore
            context=input_series,
            prediction_length=max_new_tokens,
            num_samples=num_samples,
            limit_prediction_length=False,
        )

        # predictions: [B, N, T]
        pred_values = predictions[0, :, :]
    finally:
        # Clean up hooks so subsequent runs are not affected
        if added_hooks:
            if attention_type == "both":
                pipeline.remove_head_ablation_hooks(attention_type="ca")
                pipeline.remove_head_ablation_hooks(attention_type="sa")
            else:
                pipeline.remove_head_ablation_hooks(attention_type=attention_type)  # type: ignore[arg-type]

    # Compute median across samples
    median_pred = torch.median(pred_values, dim=0).values

    return median_pred


def visualize_ablation_results(
    context: torch.Tensor,
    ground_truth: torch.Tensor,
    ablation_predictions: dict,
    ablation_rmse_values: dict,
    ablation_spearman_distances: dict,
    context_length: int,
    prediction_length: int,
    save_path: str | None = None,
    title_prefix: str = "",
):
    """
    Visualize the results of ablation experiments.

    Args:
        context: Historical data used as context
        ground_truth: Ground truth future values
        ablation_predictions: Dictionary with ablation methods as keys and lists of predictions as values
        ablation_rmse_values: Dictionary with ablation methods as keys and lists of RMSE values as values
        context_length: Length of the context
        prediction_length: Length of the prediction
        save_path: Path to save the figure (if None, figure is shown but not saved)
        title_prefix: Prefix for the plot title
    """
    # For each ablation method, create a visualization
    for ablation_method, predictions in ablation_predictions.items():
        max_k = len(predictions) - 1  # -1 because we include no ablation case

        # Create plot for ablation
        fig_ablation, axs_ablation = plt.subplots(
            3, 1, figsize=(12, 12), sharex=False, gridspec_kw={"height_ratios": [3, 1, 1]}
        )

        # Plot historical data
        axs_ablation[0].plot(context[0].cpu().numpy(), color="royalblue", label="Historical Data")

        # Plot ground truth forecast
        forecast_index = range(context_length, context_length + prediction_length)
        axs_ablation[0].plot(
            forecast_index,
            ground_truth.cpu().numpy(),
            color="black",
            linestyle="--",
            label="Ground Truth",
        )

        # Plot predictions for each ablation level with color gradient
        cmap = plt.cm.get_cmap("viridis")
        for k in range(max_k + 1):
            # Normalize color by number of ablated heads
            color = cmap(k / max_k if max_k > 0 else 0)

            # Only label a few lines to avoid cluttering the legend
            if k == 0:
                label = "No ablation"
            elif k == max_k:
                label = f"Top {k} heads ablated"
            elif k in [5, 50, 100] or k % 20 == 0:
                label = f"{k} heads ablated"
            else:
                label = None

            axs_ablation[0].plot(forecast_index, predictions[k].cpu().numpy(), color=color, label=label)

        # Plot RMSE vs number of ablated heads
        rmse_values = ablation_rmse_values[ablation_method]
        axs_ablation[1].plot(range(max_k + 1), rmse_values, marker="o", linestyle="-", color="blue")
        axs_ablation[1].scatter(range(max_k + 1), rmse_values, color="blue", s=50)

        # Plot Spearman distance vs number of ablated heads
        spearman_distances = ablation_spearman_distances[ablation_method]
        axs_ablation[2].plot(range(max_k + 1), spearman_distances, marker="o", linestyle="-", color="darkred")
        axs_ablation[2].scatter(range(max_k + 1), spearman_distances, color="darkred", s=50)

        # Add labels and legends
        if ablation_method.endswith("_r"):
            title = f"{title_prefix} Forecast with Reversed {ablation_method[:-2]}-ablated heads"
        elif ablation_method.endswith("_s"):
            title = f"{title_prefix} Forecast with Shuffled {ablation_method[:-2]}-ablated heads"
        else:
            title = f"{title_prefix} Forecast with {ablation_method}-ablated heads"

        axs_ablation[0].set_title(title)

        axs_ablation[0].set_ylabel("Value")
        axs_ablation[0].grid(True)
        axs_ablation[0].legend(loc="upper left")

        axs_ablation[1].set_title(f"RMSE vs Number of {ablation_method}-Ablated Heads (vs Unablated)")
        axs_ablation[1].set_xlabel("Number of Ablated Heads")
        axs_ablation[1].set_ylabel("RMSE")
        axs_ablation[1].grid(True)

        axs_ablation[2].set_title(f"Spearman Distance vs Number of {ablation_method}-Ablated Heads (1 - s)")
        axs_ablation[2].set_xlabel("Number of Ablated Heads")
        axs_ablation[2].set_ylabel("Spearman Distance (1 - s)")
        axs_ablation[2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.show()


def load_rrt_induction_scores(
    rrt_results_root: str,
    model_dir_name: str,
    config_key: str,
) -> dict[str, torch.Tensor]:
    """
    Load RRT induction scores saved by `rrt_induction_scores.py`.

    Expects the following structure created by the RRT script:
      {rrt_results_root}/{model_dir_name}/{config_key}/
        - center_scores.pkl  -> {"mean": Tensor[L,H], "std": Tensor[L,H]}
        - right_scores.pkl   -> {"mean": Tensor[L,H], "std": Tensor[L,H]}

    Returns a dict with keys: "center_scores", "right_scores" containing the mean tensors.
    """
    base_dir = os.path.join(rrt_results_root, model_dir_name, config_key)

    center_path = os.path.join(base_dir, "center_scores.pkl")
    right_path = os.path.join(base_dir, "right_scores.pkl")

    if not os.path.exists(center_path) or not os.path.exists(right_path):
        raise FileNotFoundError(
            "Missing RRT induction score files. Expected at:\n"
            f"  {center_path}\n  {right_path}\n"
            "Run scripts/rrt_induction_scores.py to generate them for the desired configs."
        )

    with open(center_path, "rb") as f:
        center_scores_dict = pickle.load(f)
    with open(right_path, "rb") as f:
        right_scores_dict = pickle.load(f)

    center_mean = center_scores_dict["mean"]
    right_mean = right_scores_dict["mean"]

    # Ensure tensors
    if not isinstance(center_mean, torch.Tensor):
        center_mean = torch.tensor(center_mean)
    if not isinstance(right_mean, torch.Tensor):
        right_mean = torch.tensor(right_mean)

    return {"center_scores": center_mean, "right_scores": right_mean}


def ablate_heads_sequentially(
    context: torch.Tensor,
    ground_truth: torch.Tensor,
    pipeline: CircuitLensChronos,
    heads_to_ablate: list[tuple[int, int]],
    prediction_length: int,
    num_samples: int = 12,
    ablation_method: str = "zero",
    attention_type: str = "ca",
) -> tuple[list[torch.Tensor], list[float], list[float]]:
    """
    Ablate heads sequentially and return predictions, RMSE (vs unablated), and Spearman distance
    (1 - Spearman correlation vs unablated) for each ablation level.

    Args:
        context: Input context tensor
        ground_truth: Ground truth values for calculating RMSE
        pipeline: Model pipeline
        heads_to_ablate: List of (layer_idx, head_idx) tuples to ablate sequentially
        prediction_length: Length of the prediction
        num_samples: Number of samples to generate
        ablation_method: Ablation method ("zero" or "mean") with optional suffixes:
                        "_r" to reverse the order of heads
                        "_s" to shuffle the order of heads

    Returns:
        Tuple of (predictions, rmse_values, spearman_distances) where each is a list corresponding to
        different numbers of ablated heads, starting with 0 heads ablated
    """
    predictions = []
    rmse_values = []
    spearman_distances = []

    # Process the ablation method for special suffixes
    base_ablation_method = ablation_method
    heads_order = "sequential"

    # Check for reverse suffix
    if ablation_method.endswith("_r"):
        base_ablation_method = ablation_method[:-2]
        heads_to_ablate = list(reversed(heads_to_ablate))
        heads_order = "reversed"

    # Check for shuffle suffix
    elif ablation_method.endswith("_s"):
        base_ablation_method = ablation_method[:-2]
        # Create a copy to avoid modifying the original
        heads_to_ablate = heads_to_ablate.copy()
        # Shuffle the heads
        np.random.shuffle(heads_to_ablate)
        heads_order = "shuffled"

    print(f"Starting ablation with {base_ablation_method} method in {heads_order} order")

    # First run with no ablation
    print("Generating baseline prediction with no ablation...")
    median_pred = get_median_prediction(
        context,
        pipeline,
        [],  # No heads ablated
        max_new_tokens=prediction_length,
        num_samples=num_samples,
        ablation_method=base_ablation_method,
        attention_type=attention_type,
    )

    # Baseline prediction (no ablation). RMSE and Spearman distance vs itself are zero.
    baseline_pred = median_pred
    rmse = calculate_rmse(baseline_pred, baseline_pred)
    spearman_distance = 0.0

    # Store results
    predictions.append(median_pred)
    rmse_values.append(rmse)
    spearman_distances.append(spearman_distance)

    # Then ablate heads one by one
    total_heads = len(heads_to_ablate)
    print(f"Ablating {total_heads} heads using {base_ablation_method} method in {heads_order} order...")

    # Create a progress bar
    pbar = tqdm(range(1, total_heads + 1), desc=f"{base_ablation_method}_{heads_order}")

    for k in pbar:
        # Get the first k heads to ablate
        current_heads_to_ablate = heads_to_ablate[:k]

        # Update progress bar description
        pbar.set_description(f"{base_ablation_method}_{heads_order} [{k}/{total_heads} heads]")

        # Get median prediction with ablated heads
        median_pred = get_median_prediction(
            context,
            pipeline,
            current_heads_to_ablate,
            max_new_tokens=prediction_length,
            num_samples=num_samples,
            ablation_method=base_ablation_method,
            attention_type=attention_type,
        )

        # Calculate RMSE vs unablated baseline
        rmse = calculate_rmse(median_pred, baseline_pred)

        # Calculate Spearman distance vs unablated baseline
        corr, _ = spearmanr(
            baseline_pred.detach().cpu().numpy(),
            median_pred.detach().cpu().numpy(),
        )
        # Handle possible NaNs (e.g., constant arrays) by mapping to zero distance
        if np.isnan(corr):
            spearman_distance = 0.0
        else:
            spearman_distance = 1.0 - float(corr)

        # Store results
        predictions.append(median_pred)
        rmse_values.append(rmse)
        spearman_distances.append(spearman_distance)

        # Update progress bar with RMSE
        pbar.set_postfix(RMSE=f"{rmse:.4f}", SpearmanDist=f"{spearman_distance:.4f}")

    return predictions, rmse_values, spearman_distances


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    # set seed
    torch.manual_seed(cfg.seed)
    # np.random.seed(cfg.seed)

    # Load data from ensemble
    print("Loading data from ensemble...")

    split_name = "base40"
    split_dir = os.path.join(DATA_DIR, split_name)
    # ensemble = make_ensemble_from_arrow_dir(split_dir, dyst_names_lst=[system_name], num_samples=1)

    context_length = 512
    prediction_length = 512
    sample_idx = 0
    selected_dim = 0

    for model_name in cfg.induction_ablation.model_names:
        readable_model_name = model_name.split("/")[-1]
        print(f"\n{'=' * 50}\nProcessing model: {readable_model_name}\n{'=' * 50}")

        # Load the model
        print(f"Loading model from {model_name}...")
        pipeline = CircuitLensChronos.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.bfloat16,
        )

        # Where RRT script saves per-model, per-config files
        rrt_results_dir = os.path.join(cfg.eval.results_save_dir, cfg.induction_scores.rrt_scores_dir)
        print(f"Loading induction scores from {rrt_results_dir}...")

        # Iterate over systems and their start times
        for system_config in cfg.induction_ablation.systems:
            system_name = system_config.name
            print(f"\n{'=' * 60}\nProcessing system: {system_name}\n{'=' * 60}")

            # Load system data
            dyst_coords = load_dyst_samples(system_name, split_dir, one_dim_target=False, num_samples=1)
            dyst_coords = torch.tensor(dyst_coords[sample_idx, selected_dim, :]).unsqueeze(0)

            for start_time in system_config.start_times:
                print(f"\n{'-' * 50}\nProcessing start time: {start_time}\n{'-' * 50}")

                context_start_time = start_time
                context_end_time = context_start_time + context_length

                # Prepare input series for this start time
                context = dyst_coords[:, context_start_time:context_end_time]
                future_vals = dyst_coords[:, context_end_time : context_end_time + prediction_length]

                print(f"Context shape: {context.shape}, Future values shape: {future_vals.shape}")

                # Get ground truth data
                ground_truth = future_vals[0]

                for induction_config in cfg.induction_ablation.induction_config:
                    print(f"\n{'-' * 40}\nProcessing induction config: {induction_config}\n{'-' * 40}")
                    try:
                        induction_scores_config = load_rrt_induction_scores(
                            rrt_results_dir, readable_model_name, induction_config
                        )
                    except FileNotFoundError as e:
                        logger.error(str(e))
                        logger.error(
                            "Skipping this config. Ensure it is included in cfg.induction_scores.{repeat_factors,sequence_lengths}."
                        )
                        continue

                    # Take the max of each element in the two matrices
                    max_induction_scores = torch.maximum(
                        induction_scores_config["center_scores"],
                        induction_scores_config["right_scores"],
                    )

                    # Find the top k=total heads with highest induction scores
                    max_k = None
                    print(
                        f"Finding top {torch.numel(max_induction_scores) if max_k is None else max_k} heads with highest induction scores..."
                    )
                    top_k_heads = find_max_induction_score(max_induction_scores, max_k)

                    # Number of samples to generate for each ablation level
                    num_samples = cfg.induction_ablation.num_samples

                    # Iterate over attention types
                    for attention_type in cfg.induction_ablation.attention_types:
                        print(f"\n{'-' * 30}\nProcessing attention type: {attention_type}\n{'-' * 30}")

                        # Store results for each ablation level and method
                        ablation_predictions = {}
                        ablation_rmse_values = {}
                        ablation_spearman_distances = {}

                        for ablation_method in cfg.induction_ablation.ablation_methods:
                            # Run ablation experiment using the new function
                            print(f"\n{'-' * 20}\nRunning {ablation_method} ablation experiment...\n{'-' * 20}")

                            predictions, rmse_values, spearman_distances = ablate_heads_sequentially(
                                context,
                                ground_truth,
                                pipeline,
                                top_k_heads,
                                prediction_length,
                                num_samples=num_samples,
                                ablation_method=ablation_method,
                                attention_type=attention_type,
                            )

                            # Store results
                            ablation_predictions[ablation_method] = predictions
                            ablation_rmse_values[ablation_method] = rmse_values
                            ablation_spearman_distances[ablation_method] = spearman_distances

                            # Create visualization with new save path structure
                            save_dir = os.path.join(
                                cfg.eval.results_save_dir,
                                cfg.induction_ablation.ablation_results_dir,
                                induction_config,
                                system_name,
                                str(start_time),
                                attention_type,
                            )
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(save_dir, f"{ablation_method}_ablation_experiment_results.png")

                            print(f"Creating visualization and saving to {save_path}...")
                            visualize_ablation_results(
                                context,
                                ground_truth,
                                {ablation_method: ablation_predictions[ablation_method]},
                                {ablation_method: ablation_rmse_values[ablation_method]},
                                {ablation_method: ablation_spearman_distances[ablation_method]},
                                context_length,
                                prediction_length,
                                save_path=save_path,
                                title_prefix=f"{readable_model_name} {induction_config} {system_name} t{start_time} {attention_type}",
                            )

                        print(
                            f"Completed all experiments for {readable_model_name} with {induction_config} configuration, {system_name} system, start time {start_time}, and {attention_type} attention type."
                        )


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
