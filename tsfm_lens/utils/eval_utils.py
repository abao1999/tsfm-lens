"""
utils for evaluation scripts
"""

import gc
import json
import logging
import os
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    set the random seed for torch, numpy, torch backend, cuda, everything
    NOTE: DANGER! This is bad for Chronos:
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    """
    # set the seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    # transformers.set_seed(seed)


def clear_cuda_cache(device: torch.device) -> None:
    """
    Clear the CUDA cache.
    """
    # Clear CUDA cache and load model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()


def reshape_batch_data(
    data_list: list[np.ndarray],
    num_datasets: int,
    num_eval_windows: int,
    dim: int,
    target_shape: tuple[int, ...],
) -> np.ndarray:
    """
    Reshape a list of batch data arrays with explicit dimension tracking.

    Args:
        data_list (List[np.ndarray]): List of arrays to concatenate and reshape.
        num_datasets (int): Number of systems.
        num_eval_windows (int): Number of evaluation windows per system.
        dim (int): Number of dimensions in the data.
        target_shape (Tuple[int, ...]): Shape of the target data (excluding system, window, and dim).

    Returns:
        np.ndarray: Reshaped array of shape (num_datasets * num_eval_windows, *target_shape, dim).

    Notes:
    - The original data is stored in memory as (num_samples, total_eval_windows, prediction_length) where total_eval_windows = num_datasets * num_eval_windows * dim
    - The data is organized as [system1_window1_dim1, system1_window1_dim2, ..., system1_window2_dim1, ...] in memory
    - Reshape behavior: reshape() only changes the view of the data without reordering elements in memory. It would incorrectly group the dimensions.
    - Transpose behavior: transpose() reorders the dimensions in memory. It would correctly group the dimensions.
    """
    # Concatenate along the appropriate axis
    concatenated = np.concatenate(data_list, axis=0 if len(data_list[0].shape) == 2 else 1)

    # Reshape to separate dimensions: (num_datasets, num_eval_windows, dim, ...)
    reshaped = concatenated.reshape(num_datasets, num_eval_windows, dim, *target_shape)

    # Transpose to get desired order: (num_datasets, num_eval_windows, ..., dim)
    transposed = reshaped.transpose(0, 1, *range(3, len(reshaped.shape)), 2)

    # Reshape to final form: (num_datasets * num_eval_windows, ..., dim)
    return transposed.reshape(num_datasets * num_eval_windows, *target_shape, dim)


def validate_and_get_sample_count(
    predictions: np.ndarray,
    num_datasets: int,
    num_eval_windows: int,
    prediction_length: int,
    dim: int,
) -> int:
    """
    Validate the shape of the predictions array and return the number of samples after reduction.

    Args:
        predictions (np.ndarray): Predictions array, expected to be either 3D or 4D.
        num_datasets (int): Number of systems.
        num_eval_windows (int): Number of evaluation windows per system.
        prediction_length (int): Number of timesteps in each prediction.
        dim (int): Number of dimensions in the data.

    Returns:
        int: Actual number of samples after reduction.

    Raises:
        ValueError: If the predictions array does not have the expected shape.
    """
    if len(predictions.shape) == 4:
        actual_num_samples = predictions.shape[0]
        expected_shape = (
            actual_num_samples,
            num_datasets * num_eval_windows,
            prediction_length,
            dim,
        )
    elif len(predictions.shape) == 3:
        actual_num_samples = 1
        expected_shape = (num_datasets * num_eval_windows, prediction_length, dim)
    else:
        raise ValueError(f"Unexpected predictions shape after reduction: {predictions.shape}")

    if predictions.shape != expected_shape:
        raise ValueError(f"Predictions shape {predictions.shape} != expected {expected_shape}")

    return actual_num_samples


def calculate_rmse(
    predictions: np.ndarray,
    true_vals: np.ndarray,
    corrected_timesteps: np.ndarray,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate RMSE between median prediction and true values, excluding corrected timesteps.

    Args:
        predictions: predictions of shape (batch_size, num_samples, num_timepoints)
        true_vals: Ground truth values of shape (batch_size, num_timepoints)
        corrected_timesteps: Boolean mask indicating which timesteps were corrected, of shape (num_timepoints,)

    Returns:
        Tuple containing:
        - RMSE value (float)
        - Standard deviation of sample errors (float)

    Note:
        Returns (0, 0) if no non-corrected timesteps exist (i.e. all timesteps were corrected).
    """
    batch_size, num_samples, num_timepoints = predictions.shape
    assert num_timepoints == true_vals.shape[-1] == len(corrected_timesteps), "Mismatch in number of timepoints"

    if not np.any(~corrected_timesteps):
        return np.zeros(batch_size), np.zeros(batch_size)

    # shape (batch_size, num_timesteps)
    median_pred = np.median(predictions, axis=1)

    # shape (batch_size,)
    rmse = np.sqrt(
        np.mean(
            (median_pred[:, ~corrected_timesteps] - true_vals[:, ~corrected_timesteps]) ** 2,
            axis=-1,
        )
    )

    # shape (batch_size, num_samples)
    sample_errors = np.sqrt(
        np.mean(
            (predictions[:, :, ~corrected_timesteps] - true_vals[:, None, ~corrected_timesteps]) ** 2,
            axis=-1,
        )
    )
    assert sample_errors.shape == (batch_size, num_samples), "Sample errors shape mismatch"

    rmse_std_val = np.std(sample_errors, axis=-1)

    if verbose:
        print(f"RMSE shape: {rmse.shape}")
        print(f"RMSE std val shape: {rmse_std_val.shape}")

    assert rmse.shape == rmse_std_val.shape == (batch_size,), "RMSE and RMSE std val shape mismatch"

    return rmse, rmse_std_val


def left_pad_and_stack_1D(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    Left pad a list of 1D tensors to the same length and stack them.
    Used in pipeline, if given context is a list of tensors.
    """
    max_len = max(len(c) for c in tensors)
    padded = []
    for c in tensors:
        assert isinstance(c, torch.Tensor)
        assert c.ndim == 1
        padding = torch.full(size=(max_len - len(c),), fill_value=torch.nan, device=c.device)
        padded.append(torch.concat((padding, c), dim=-1))
    return torch.stack(padded)


def save_evaluation_results(
    metrics: dict[int, dict[str, dict[str, float]]] | None = None,
    metrics_metadata: dict[str, dict[str, Any]] | None = None,
    metrics_save_dir: str = "results",
    metrics_fname: str | None = None,
    overwrite: bool = False,
) -> None:
    """
    Save prediction metrics and optionally forecast trajectories.

    Args:
        metrics: Nested dictionary containing computed metrics for each system.
        metrics_metadata: Dictionary containing metadata for the metrics.
            Keys are the quantity names, values are dictionaries containing metadata for each system.
                e.g. {"system_dims": {"system_name": 3}}
        metrics_save_dir: Directory to save metrics to.
        metrics_fname: Name of the metrics file to save.
        overwrite: Whether to overwrite an existing metrics file
    """
    if metrics is not None:
        for forecast_length, metric_dict in metrics.items():
            result_rows = [{"system": system, **metric_dict[system]} for system in metric_dict]
            results_df = pd.DataFrame(result_rows)
            if metrics_metadata is not None:
                for quantity_name in metrics_metadata:
                    results_df[quantity_name] = results_df["system"].map(metrics_metadata[quantity_name])  # type: ignore
            curr_metrics_fname = f"{metrics_fname or 'metrics'}_pred{forecast_length}.csv"
            metrics_save_path = os.path.join(metrics_save_dir, curr_metrics_fname)
            logger.info(f"Saving metrics to: {metrics_save_path}")
            os.makedirs(os.path.dirname(metrics_save_path), exist_ok=True)

            if os.path.isfile(metrics_save_path) and not overwrite:
                existing_df = pd.read_csv(metrics_save_path)
                results_df = pd.concat([existing_df, results_df], ignore_index=True)
            results_df.to_csv(metrics_save_path, index=False)


def load_json_data(file_path: str) -> dict:
    """
    Load a JSON file and return the data as a dictionary.
    """
    with open(file_path) as f:
        return json.load(f)


def summarize_metrics_dict(data_dict: dict) -> None:
    """
    given a metrics data dict of the same structure:
        za_heads_layers_{list-of-layers-ablated (ints separated by '-')}": {
            prediction_horizon: {
                system_name: {
                    metric_name: list[float]
    print a summary of the data dict, including the number of values for each metric, prediction horizon, and system
    """
    first_key = list(data_dict.keys())[0]
    first_horizon = list(data_dict[first_key].keys())[0]
    first_system = list(data_dict[first_key][first_horizon].keys())[0]
    layergroup_indices = [i for i in range(len(data_dict.keys()))]

    print(
        f"Available prediction horizons: {list(data_dict[first_key].keys())}\n"
        f"Available systems: {list(data_dict[first_key][first_horizon].keys())}\n"
        f"Available metrics: {list(data_dict[first_key][first_horizon][first_system].keys())}\n"
        f"Available layergroup indices: {layergroup_indices}"
        f"Available ablations: {data_dict.keys()}"
    )

    # NOTE: this is no longer valid way of counting when comparing against labels
    total_values = sum(
        len(data_dict[first_key][first_horizon][sys]["spearman"]) for sys in data_dict[first_key][first_horizon]
    )
    print(f"total number of context windows: {total_values}")


def combine_metrics_dicts(data_dicts: list[dict], verbose: bool = False) -> dict:
    """
    given a list of data dicts of the same structure:
        za_heads_layers_{list-of-layers-ablated (ints separated by '-')}": {
            prediction_horizon: {
                system_name: {
                    metric_name: list[float]
                }
            }
        }
    Return a combined data dict with the same structure, but with system_name metrics merged together (i.e. extending the lists)
    """
    n_data_dicts = len(data_dicts)
    if n_data_dicts == 0:
        raise ValueError("No data dicts to combine")
    elif n_data_dicts == 1:
        print("Only one data dict, returning it directly")
        return data_dicts[0]
    else:
        print(f"Combining {n_data_dicts} data dicts...")

    if verbose:
        max_num_dicts_to_summarize = min(n_data_dicts, 10)
        for i, data_dict in enumerate(data_dicts[:max_num_dicts_to_summarize]):
            print(f"Data dict {i + 1}:")
            summarize_metrics_dict(data_dict)

    # Use nested defaultdicts to automatically create missing dictionary levels
    combined_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

    # Combine all dictionaries
    for data_dict in data_dicts:
        for key in data_dict:
            for pred_horizon in data_dict[key]:
                for system in data_dict[key][pred_horizon]:
                    for metric, values in data_dict[key][pred_horizon][system].items():
                        combined_data[key][pred_horizon][system][metric].extend(values)

    if verbose:
        print("Combined result:")
        summarize_metrics_dict(combined_data)

    return dict(combined_data)


def unify_freq(df):
    """
    Unify and standardize frequency names in a DataFrame.

    This function cleans and standardizes time series frequency codes by removing
    numeric characters (e.g., "12H" -> "H"), removing suffixes after hyphens, and
    converting short frequency codes to their full descriptive names.

    Args:
        df (pd.DataFrame): A DataFrame containing a 'frequency' column with
            frequency codes that may include numeric values and suffixes.
            Examples: "12H", "1D-custom", "2M", "Q", etc.

    Returns:
        pd.DataFrame: The input DataFrame with the 'frequency' column modified
            in place, where frequency codes are replaced with standardized full
            names (e.g., "H" -> "Hourly", "D" -> "Daily", "M" -> "Monthly").

    Example:
        >>> df = pd.DataFrame({'frequency': ['12H', '1D-custom', '2M']})
        >>> df = unify_freq(df)
        >>> print(df['frequency'].tolist())
        ['Hourly', 'Daily', 'Monthly']

    Note:
        The function modifies the 'frequency' column in place and returns the
        entire DataFrame for chaining convenience.
    """
    # Remove all numeric characters from the 'frequency' column
    df["frequency"] = df["frequency"].str.replace(r"\d+", "", regex=True)
    # Remove everything after '-' if present
    df["frequency"] = df["frequency"].str.split("-").str[0]

    # Define the frequency conversion dictionary
    freq_conversion = {
        "T": "Minutely",
        "H": "Hourly",
        "D": "Daily",
        "W": "Weekly",
        "M": "Monthly",
        "Q": "Quarterly",
        "Y": "Yearly",
        "A": "Yearly",
        "S": "Secondly",
    }

    # Map the cleaned 'frequency' values using the dictionary
    df["frequency"] = df["frequency"].replace(freq_conversion)
    return df


def normalize_by_seasonal_naive(df, seasonal_naive_df):
    """
    Normalize forecasting evaluation metrics by seasonal naive baseline performance.

    This function normalizes multiple evaluation metrics (MSE, MAE, MASE, MAPE, etc.)
    by dividing them by corresponding seasonal naive baseline values. Normalization is
    performed per unique combination of dataset, frequency, and term_length, allowing
    for fair comparison of model performance across different forecasting scenarios.

    The function handles:
    - Parsing dataset identifiers in the format "dataset_name/frequency/term_length"
    - Correcting known dataset name variations
    - Standardizing frequency codes using unify_freq()
    - Handling missing values by filling with column means
    - Matching baseline values to data rows based on dataset/frequency/term_length

    Args:
        df (pd.DataFrame): Main DataFrame containing forecasting evaluation metrics.
            Must have a 'dataset' column with format "dataset_name/frequency/term_length"
            and metric columns like "eval_metrics/MSE[mean]", "eval_metrics/MAE[0.5]", etc.

        seasonal_naive_df (pd.DataFrame): DataFrame containing seasonal naive baseline
            metrics with the same structure as df. Used as the denominator for
            normalization.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with all metric columns normalized
            by their corresponding seasonal naive baseline values. Values are divided
            by baseline (e.g., normalized_MSE = model_MSE / seasonal_naive_MSE).
            Lower normalized values indicate better performance relative to the baseline.

    Metrics normalized:
        - MSE[mean], MSE[0.5]: Mean Squared Error
        - MAE[0.5]: Mean Absolute Error
        - MASE[0.5]: Mean Absolute Scaled Error
        - MAPE[0.5]: Mean Absolute Percentage Error
        - sMAPE[0.5]: Symmetric Mean Absolute Percentage Error
        - MSIS: Mean Scaled Interval Score
        - RMSE[mean]: Root Mean Squared Error
        - NRMSE[mean]: Normalized Root Mean Squared Error
        - ND[0.5]: Normalized Deviation
        - mean_weighted_sum_quantile_loss: Weighted quantile loss

    Dataset name corrections applied:
        - "saugeenday" -> "saugeen"
        - "temperature_rain_with_missing" -> "temperature_rain"
        - "kdd_cup_2018_with_missing" -> "kdd_cup_2018"
        - "car_parts_with_missing" -> "car_parts"

    Prints:
        Summary statistics including the number of successful normalizations
        and count of missing baseline entries.

    Raises:
        Prints error messages for normalization failures but continues processing.

    Note:
        - Missing values in metrics are filled with column means before normalization
        - Baseline values of 0 or NaN are skipped to avoid division errors
        - The function creates a copy of the input DataFrame, leaving the original unchanged

    Example:
        >>> df_norm = normalize_by_seasonal_naive(model_results, seasonal_naive_results)
        Normalization complete:
          - Successful metric normalizations: 1100
          - Missing baseline entries: 5
    """
    # Define metric columns
    metric_columns = [
        "eval_metrics/MSE[mean]",
        "eval_metrics/MSE[0.5]",
        "eval_metrics/MAE[0.5]",
        "eval_metrics/MASE[0.5]",
        "eval_metrics/MAPE[0.5]",
        "eval_metrics/sMAPE[0.5]",
        "eval_metrics/MSIS",
        "eval_metrics/RMSE[mean]",
        "eval_metrics/NRMSE[mean]",
        "eval_metrics/ND[0.5]",
        "eval_metrics/mean_weighted_sum_quantile_loss",
    ]

    # Make a copy to avoid modifying original
    df_normalized = df.copy()

    # Parse dataset column to extract dataset, frequency, and term_length
    # Format is: "dataset_name/frequency/term_length"
    dataset_parts = df_normalized["dataset"].str.split("/", expand=True)
    df_normalized["dataset_base"] = dataset_parts[0]
    df_normalized["frequency"] = dataset_parts[1] if len(dataset_parts.columns) > 1 else None
    df_normalized["term_length"] = dataset_parts[2] if len(dataset_parts.columns) > 2 else None

    # Apply dataset name corrections
    dataset_corrections = {
        "saugeenday": "saugeen",
        "temperature_rain_with_missing": "temperature_rain",
        "kdd_cup_2018_with_missing": "kdd_cup_2018",
        "car_parts_with_missing": "car_parts",
    }
    df_normalized["dataset_base"] = df_normalized["dataset_base"].replace(dataset_corrections)

    # Unify frequency names
    df_normalized = unify_freq(df_normalized)

    # Convert metric columns to float and fill NA with column mean
    for metric in metric_columns:
        if metric in df_normalized.columns:
            df_normalized[metric] = df_normalized[metric].astype(float)
            df_normalized[metric] = df_normalized[metric].fillna(df_normalized[metric].mean())

    # Prepare seasonal_naive_df
    seasonal_naive_normalized = seasonal_naive_df.copy()

    # Parse seasonal naive dataset column
    sn_parts = seasonal_naive_normalized["dataset"].str.split("/", expand=True)
    seasonal_naive_normalized["dataset_base"] = sn_parts[0]
    seasonal_naive_normalized["frequency"] = sn_parts[1] if len(sn_parts.columns) > 1 else None
    seasonal_naive_normalized["term_length"] = sn_parts[2] if len(sn_parts.columns) > 2 else None

    # Apply corrections to seasonal naive
    seasonal_naive_normalized["dataset_base"] = seasonal_naive_normalized["dataset_base"].replace(dataset_corrections)

    # Unify frequency names for seasonal naive
    seasonal_naive_normalized = unify_freq(seasonal_naive_normalized)

    # Convert seasonal naive metrics to float and fill NA
    for metric in metric_columns:
        if metric in seasonal_naive_normalized.columns:
            seasonal_naive_normalized[metric] = seasonal_naive_normalized[metric].astype(float)
            seasonal_naive_normalized[metric] = seasonal_naive_normalized[metric].fillna(
                seasonal_naive_normalized[metric].mean()
            )

    # Normalize by seasonal naive baseline
    normalization_count = 0
    missing_count = 0

    # Get unique combinations
    dataset_names = df_normalized["dataset_base"].unique()

    for dataset in dataset_names:
        term_lengths = df_normalized[df_normalized["dataset_base"] == dataset]["term_length"].unique()
        for term_length in term_lengths:
            frequencies = df_normalized[
                (df_normalized["dataset_base"] == dataset) & (df_normalized["term_length"] == term_length)
            ]["frequency"].unique()

            for frequency in frequencies:
                # Get seasonal naive baseline for this combination
                baseline = seasonal_naive_normalized[
                    (seasonal_naive_normalized["dataset_base"] == dataset)
                    & (seasonal_naive_normalized["frequency"] == frequency)
                    & (seasonal_naive_normalized["term_length"] == term_length)
                ]

                if len(baseline) > 0:
                    # Normalize each metric
                    for metric in metric_columns:
                        if metric in df_normalized.columns and metric in baseline.columns:
                            try:
                                baseline_value = baseline[metric].values[0]

                                if not np.isnan(baseline_value) and baseline_value != 0:
                                    mask = (
                                        (df_normalized["dataset_base"] == dataset)
                                        & (df_normalized["frequency"] == frequency)
                                        & (df_normalized["term_length"] == term_length)
                                    )
                                    df_normalized.loc[mask, metric] = df_normalized.loc[mask, metric] / baseline_value
                                    normalization_count += 1
                            except Exception as e:
                                print(f"Error normalizing {dataset}/{frequency}/{term_length} - {metric}: {e}")
                else:
                    missing_count += 1

    # Drop temporary columns
    df_normalized = df_normalized.drop(columns=["dataset_base", "frequency", "term_length"])

    print("\nNormalization complete:")
    print(f"  - Successful metric normalizations: {normalization_count}")
    print(f"  - Missing baseline entries: {missing_count}")

    return df_normalized
