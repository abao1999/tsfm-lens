import os
import warnings
from collections import defaultdict
from datetime import datetime
from itertools import islice
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Literal

import numpy as np
from gluonts.dataset.arrow import ArrowWriter
from gluonts.dataset.common import FileDataset

from tsfm_lens.dataset import GiftEvalDataset, Term


def get_eval_data_dict(
    data_dirs: list[str] | str,
    num_subdirs: int | None = None,
    num_samples_per_subdir: int | None = None,
    one_dim_target: bool = False,
    subdir_names: list[str] | None = None,
) -> dict[str, list[FileDataset]]:  # type: ignore
    """
    data_dirs: list of test data directories paths or a single test data directory path
    num_subdirs: number of subdirectories to consider
    num_samples_per_subdir: number of samples per subdirectory to consider
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    # get test data paths
    test_data_dirs_dict = defaultdict(list)  # maps subdirectory name to test data path
    for test_data_dir in data_dirs:
        test_data_dir = os.path.expandvars(test_data_dir)
        system_dirs = [d for d in Path(test_data_dir).iterdir() if d.is_dir()]
        for subdir in system_dirs:
            # can be either name of skew pair or name of the unique system (parameter perturbation)
            subdir_name = subdir.name
            if subdir_names is not None and subdir_name not in subdir_names:
                continue
            test_data_dirs_dict[subdir_name].append(subdir)

    test_data_dict = {}
    for subdir_name, subdirs in islice(test_data_dirs_dict.items(), num_subdirs):
        system_files = []
        for subdir in subdirs:
            system_files.extend(list(subdir.glob("*")))
        # sort system_files by sample_idx where the files in system_files are named like {sample_idx}_T-4096.arrow
        # also, take only the first cfg.chronos.num_samples_per_subdir files in each subdirectory
        # ---> This means only the first 10 parameter perturbations per skew pair, if the subdirectory names are the skew pairs
        system_files = sorted(
            system_files,
            key=lambda x: int(x.stem.split("_")[0]),
        )[:num_samples_per_subdir]

        test_data_dict[subdir_name] = [
            FileDataset(path=Path(file_path), freq="h", one_dim_target=one_dim_target)
            for file_path in system_files
            if file_path.is_file()
        ]
    return test_data_dict


def get_gift_eval_data_dict(
    data_dir: str,
    dataset_names: list[str] | None = None,
    term: Term | str = Term.SHORT,
    to_univariate: bool = False,
) -> dict[str, list[GiftEvalDataset]]:
    """
    Load Gift-Eval datasets and return them in a format compatible with ablations.py.

    Parameters
    ----------
    data_dir
        Directory containing Gift-Eval datasets (each dataset should be a subdirectory).
    dataset_names
        List of dataset names to load. If None, loads all available datasets.
    term
        Prediction term (short, medium, long).
    to_univariate
        Whether to convert multivariate datasets to univariate.

    Returns
    -------
    dict[str, list[GiftEvalDataset]]
        Dictionary mapping dataset names to lists of GiftEvalDataset instances.
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise ValueError(f"Data directory {data_dir} does not exist")

    # Get all dataset directories (handle nested structure)
    dataset_dirs = []

    def find_dataset_dirs(path: Path, depth: int = 0):
        """Recursively find directories that contain dataset files up to depth 2."""
        if depth > 2:
            return

        for item in path.iterdir():
            if item.is_dir():
                # Check if this directory contains dataset files and no subdirectories
                has_dataset_files = any(
                    f.suffix in [".arrow", ".parquet"] or f.name in ["dataset_info.json", "state.json"]
                    for f in item.iterdir()
                    if f.is_file()
                ) and not any(f.is_dir() for f in item.iterdir())
                if has_dataset_files:
                    dataset_dirs.append(item)
                else:
                    # Recursively search subdirectories up to depth 2
                    find_dataset_dirs(item, depth + 1)

    find_dataset_dirs(data_path)
    print(f"dataset_dirs: {dataset_dirs}")

    if dataset_names is not None:
        # Filter to only requested datasets
        dataset_dirs = [d for d in dataset_dirs if d.relative_to(data_path).as_posix() in dataset_names]
    if not dataset_dirs:
        raise ValueError(f"No datasets found in {data_dir}")

    # Load datasets
    gift_eval_datasets = {}
    for dataset_dir in dataset_dirs:
        dataset_name = dataset_dir.relative_to(data_path).as_posix()
        try:
            # Create GiftEvalDataset instance
            dataset = GiftEvalDataset(
                name=dataset_name,
                term=term,
                to_univariate=to_univariate,
                data_dir=str(data_path),  # Use parent directory as data_dir
            )
            gift_eval_datasets[dataset_name] = [dataset]
        except Exception as e:
            print(f"Warning: Failed to load dataset {dataset_name}: {e}")
            continue

    return gift_eval_datasets


def get_dim_from_dataset(dataset: FileDataset) -> int:  # type: ignore
    """
    helper function to get system dimension from file dataset
    """
    return next(iter(dataset))["target"].shape[0]


def safe_standardize(
    arr: np.ndarray,
    epsilon: float = 1e-10,
    axis: int = -1,
    context: np.ndarray | None = None,
    denormalize: bool = False,
) -> np.ndarray:
    """
    Standardize the trajectories by subtracting the mean and dividing by the standard deviation

    Args:
        arr: The array to standardize
        epsilon: A small value to prevent division by zero
        axis: The axis to standardize along
        context: The context to use for standardization. If provided, use the context to standardize the array.
        denormalize: If True, denormalize the array using the mean and standard deviation of the context.

    Returns:
        The standardized array
    """
    # if no context is provided, use the array itself
    context = arr if context is None else context

    assert arr.ndim == context.ndim, "arr and context must have the same num dims if context is provided"
    assert axis < arr.ndim and axis >= -arr.ndim, "invalid axis specified for standardization"
    mean = np.nanmean(context, axis=axis, keepdims=True)
    std = np.nanstd(context, axis=axis, keepdims=True)
    std = np.where(std < epsilon, epsilon, std)
    if denormalize:
        return arr * std + mean
    return (arr - mean) / std


def convert_to_arrow(
    path: str | Path,
    time_series: list[np.ndarray] | np.ndarray,
    compression: Literal["lz4", "zstd"] = "lz4",
    split_coords: bool = False,
):
    """
    Store a given set of series into Arrow format at the specified path.

    Input data can be either a list of 1D numpy arrays, or a single 2D
    numpy array of shape (num_series, time_length).
    """
    assert isinstance(time_series, list) or (isinstance(time_series, np.ndarray) and time_series.ndim == 2), (
        "time_series must be a list of 1D numpy arrays or a 2D numpy array"
    )

    # GluonTS requires this datetime format for reading arrow file
    start = datetime.now().strftime("%Y-%m-%d %H:%M")

    if split_coords:
        dataset = [{"start": start, "target": ts} for ts in time_series]
    else:
        dataset = [{"start": start, "target": time_series}]

    ArrowWriter(compression=compression).write_to_file(
        dataset,
        path=Path(path),
    )


def load_trajectory_from_arrow(filepath: Path | str, one_dim_target: bool = False) -> tuple[np.ndarray, tuple[Any]]:
    """Load a trajectory and metadata from an arrow file

    Args:
        filepath: The path to the arrow file
        one_dim_target: Whether the target is one-dimensional

    Returns:
        A tuple containing the trajectory and metadata
    """
    assert Path(filepath).exists(), f"Filepath {filepath} does not exist"
    dataset = FileDataset(path=Path(filepath), freq="h", one_dim_target=one_dim_target)
    coords, metadata = zip(*[(coord["target"], coord["start"]) for coord in dataset])
    coordinates = np.stack(coords)
    if coordinates.ndim > 2:  # if not one_dim_target:
        coordinates = coordinates.squeeze()
    return coordinates, metadata


def get_system_filepaths(system_name: str, data_dir: str, sort: bool = True) -> list[Path]:
    """
    Retrieve sorted filepaths for a given dynamical system.

    This function finds and sorts all Arrow files for a specified dynamical system
    within a given directory structure.

    Args:
        system_name (str): The name of the dynamical system.
        data_dir (Union[str, Path]): The base directory containing the data.

    Returns:
        List[Path]: A sorted list of Path objects for the Arrow files of the specified system.

    Raises:
        Exception: If the directory for the specified system does not exist.

    Note:
        The function assumes that the Arrow files are named with a numeric prefix
        (e.g., "1_T-1024.arrow") and sorts them based on this prefix.
    """
    dyst_dir = os.path.join(data_dir, system_name)
    if not os.path.exists(dyst_dir):
        raise Exception(f"Directory {dyst_dir} does not exist.")

    # NOTE: sorting by numerical order wrt sample index is very important for consistency
    if sort:
        filepaths = sorted(list(Path(dyst_dir).glob("*.arrow")), key=lambda x: int(x.stem.split("_")[0]))
    else:
        filepaths = list(Path(dyst_dir).glob("*.arrow"))
    return filepaths


def load_dyst_samples(
    dyst_name: str,
    data_dir: str,
    one_dim_target: bool = False,
    num_samples: int | None = None,
    transient_prop: float = 0.0,
    sort: bool = True,
) -> np.ndarray | None:
    """
    Load a set of sample trajectories of a given dynamical system from an arrow file

    Args:
        dyst_name: The name of the dynamical system
        data_dir: The base directory containing the data
        one_dim_target: Whether the target is one-dimensional
        num_samples: The number of samples to load. If None, all available samples are loaded.

    Returns:
        A numpy array containing the trajectories
        Array has shape (num_samples, dim, T)

    """

    filepaths = get_system_filepaths(dyst_name, data_dir, sort=sort)
    selected_filepaths = filepaths[:num_samples]
    print(f"Loading {len(selected_filepaths)} samples for {dyst_name}")
    dyst_coords_samples = []
    for filepath in selected_filepaths:
        dyst_coords, _ = load_trajectory_from_arrow(filepath, one_dim_target)
        transient_time = int(transient_prop * dyst_coords.shape[1])
        dyst_coords_samples.append(dyst_coords[:, transient_time:])

    dyst_coords_samples = np.array(dyst_coords_samples)  # type: ignore
    print(dyst_coords_samples.shape)
    return dyst_coords_samples


def make_ensemble_from_arrow_dir(
    data_dir: str,
    dyst_names_lst: list[str] | None = None,
    num_samples: int | None = None,
    num_systems: int | None = None,
    one_dim_target: bool = False,
    transient_prop: float = 0.0,
    num_processes: int = cpu_count(),
    verbose: bool = False,
    sort: bool = True,
) -> dict[str, np.ndarray]:
    """
    Loads an ensemble of sample trajectories for multiple dynamical systems from Arrow files in a directory.

    Args:
        data_dir (str): The base directory containing the data.
        dyst_names_lst (list[str] | None, optional): List of system names to load. If None, all systems in the directory are used.
        num_samples (int | None, optional): Number of samples to load per system.
                If None, all available samples are loaded.
        num_systems (int | None, optional): Number of systems to load. If None, all systems are loaded.
        one_dim_target (bool, optional): Whether the target is one-dimensional. Defaults to False.
        num_processes (int, optional): Number of processes to use for parallel loading. Defaults to the number of CPU cores.

    Returns:
        dict[str, np.ndarray]: A dictionary mapping system names to arrays of sample trajectories,
            where each array has shape (num_samples, num_features, num_timesteps).

    Raises:
        AssertionError: If num_systems is greater than the number of available systems.

    Note:
        This function uses multiprocessing to speed up loading of large ensembles.
        The Arrow files for each system are expected to be in subdirectories under data_dir,
        and named with a numeric prefix (e.g., "1_T-1024.arrow").
    """
    if num_samples == 1:
        if verbose:
            print("num_samples is 1, only loading the default ensemble (filenames 0_T-1024.arrow)")

    ensemble: dict[str, np.ndarray] = {}
    if dyst_names_lst is None:
        dyst_names_lst = sorted([folder.name for folder in Path(data_dir).iterdir() if folder.is_dir()])

    if num_systems is not None:
        if verbose:
            print(f"Only loading the first {num_systems} systems")
        assert num_systems <= len(dyst_names_lst), (
            f"num_systems {num_systems} must be less than or equal to the number of systems in the directory {len(dyst_names_lst)}"
        )
        dyst_names_lst = dyst_names_lst[:num_systems]

    # Prepare arguments for multiprocessing
    args = [(dyst_name, data_dir, one_dim_target, num_samples, transient_prop, sort) for dyst_name in dyst_names_lst]

    # Use multiprocessing to process each dyst_name
    with Pool(num_processes) as pool:
        results = pool.starmap(load_dyst_samples, args)

    # Collect results into the ensemble dictionary
    for dyst_name, dyst_coords_samples in zip(dyst_names_lst, results):
        if dyst_coords_samples is None:
            warnings.warn(f"No samples found for {dyst_name}")
            continue
        if verbose:
            print(f"Loaded {dyst_coords_samples.shape[0]} samples for {dyst_name}")
        ensemble[dyst_name] = dyst_coords_samples

    return ensemble


def process_trajs(
    base_dir: str,
    timeseries: dict[str, np.ndarray],
    split_coords: bool = False,
    overwrite: bool = False,
    verbose: bool = False,
    base_sample_idx: int = -1,
) -> None:
    """
    Saves each trajectory in timeseries ensemble to a separate directory

    Args:
        base_dir: The base directory to save the trajectories to
        timeseries: A dictionary mapping system names to numpy arrays of shape
            (num_eval_windows * num_datasets, num_channels, T) where T is the prediction
            length or context length.
        split_coords: Whether to split the coordinates by dimension
        overwrite: Whether to overwrite existing trajectories
        verbose: Whether to print verbose output
        base_sample_idx: The base sample index to use for the trajectories
    """
    for sys_name, trajectories in timeseries.items():
        if verbose:
            print(f"Processing trajectories of shape {trajectories.shape} for system {sys_name}")

        system_folder = os.path.join(base_dir, sys_name)
        os.makedirs(system_folder, exist_ok=True)

        if not overwrite:
            for filename in os.listdir(system_folder):
                if filename.endswith(".arrow"):
                    sample_idx = int(filename.split("_")[0])
                    base_sample_idx = max(base_sample_idx, sample_idx)

        for i, trajectory in enumerate(trajectories):
            # very hacky, if there is only one trajectory, we can just use the base_sample_idx
            curr_sample_idx = base_sample_idx + i + 1

            if trajectory.ndim == 1:
                trajectory = np.expand_dims(trajectory, axis=0)
            if verbose:
                print(f"Saving {sys_name} trajectory {curr_sample_idx} with shape {trajectory.shape}")

            path = os.path.join(system_folder, f"{curr_sample_idx}_T-{trajectory.shape[-1]}.arrow")

            convert_to_arrow(path, trajectory, split_coords=split_coords)


def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj
