"""
Dataset for Chronos
Modified from original Chronos codebase https://github.com/amazon-science/chronos-forecasting
    (under Apache-2.0 license):
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0
"""

import itertools
import math
from collections.abc import Callable, Generator, Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, partial
from pathlib import Path

import datasets
import numpy as np
import pyarrow.compute as pc
import torch
from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.itertools import Filter, Map
from gluonts.time_feature import norm_freq_str
from gluonts.transform import (
    InstanceSampler,
    InstanceSplitter,
    LeavesMissingValues,
    MissingValueImputation,
    Transformation,
)
from pandas.tseries.frequencies import to_offset
from toolz import compose
from torch.utils.data import IterableDataset

# used for prediction length in test mode when window style is single
# if you're predicting for more timepoints than this at a time...what are you doing??
MAX_PREDICTION_LENGTH = 1_000_000

TEST_SPLIT = 0.1
MAX_WINDOW = 20

M4_PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 18,
    "ME": 18,  # Month End (new pandas frequency code)
    "W": 13,
    "D": 14,
    "H": 48,
}

PRED_LENGTH_MAP = {
    "M": 12,
    "ME": 12,  # Month End (new pandas frequency code)
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "min": 48,  # Minute (alternative code for T)
    "S": 60,
}

TFB_PRED_LENGTH_MAP = {
    "A": 6,
    "H": 48,
    "Q": 8,
    "D": 14,
    "M": 18,
    "ME": 18,  # Month End (new pandas frequency code)
    "W": 13,
    "U": 8,
    "T": 8,
    "min": 8,  # Minute (alternative code for T)
}


class RegularWindowedSampler(InstanceSampler):
    """
    Sample regular context windows from each series.

    Parameters
    ----------
    stride: int
        stride of the sampled context windows
    """

    stride: int

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        if a > b:
            return np.array([], dtype=int)

        return np.arange(a, b + 1, self.stride)


class SingleContextSampler(InstanceSampler):
    """
    Sample a single context window from the beginning of each series.

    Used for autoregressive prediction where the model should predict the
    rest of the entire timeseries.
    """

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        if a > b:
            return np.array([], dtype=int)

        return np.array([a])


class NumInstanceSampler(InstanceSampler):
    """
    Samples N time points from each series.

    Parameters
    ----------
    N
        number of time points to sample from each time series.
    """

    N: int
    rng: np.random.Generator

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        if a > b:
            return np.array([], dtype=int)

        return self.rng.integers(a, b + 1, size=self.N)


class FixedStartSampler(InstanceSampler):
    """
    Sample a single context window starting at a specific time index.

    Parameters
    ----------
    start_time: int
        The time index where the window should start
    """

    start_time: int

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        if a > b:
            return np.array([], dtype=int)

        # Ensure the start_time is within valid bounds
        if self.start_time < a or self.start_time > b:
            return np.array([], dtype=int)

        return np.array([self.start_time])


class PseudoShuffledIterableDataset(IterableDataset):
    """
    Shuffle entries from an iterable by temporarily accumulating them
    in an intermediate buffer.

    Parameters
    ----------
    base_dataset
        The original iterable object, representing the dataset.
    shuffle_buffer_length
        Size of the buffer use to shuffle entries from the base dataset.
    """

    def __init__(self, base_dataset, shuffle_buffer_length: int = 100) -> None:
        super().__init__()
        self.base_dataset = base_dataset
        self.shuffle_buffer_length = shuffle_buffer_length
        self.generator = torch.Generator()

    def __iter__(self):
        shuffle_buffer = []

        for element in self.base_dataset:
            shuffle_buffer.append(element)
            if len(shuffle_buffer) >= self.shuffle_buffer_length:
                idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
                yield shuffle_buffer.pop(idx)

        while shuffle_buffer:
            idx = torch.randint(len(shuffle_buffer), size=(), generator=self.generator)
            yield shuffle_buffer.pop(idx)


class ShuffleMixin:
    """
    Mix-in class that datasets can inherit from to get
    shuffling functionality.
    """

    def shuffle(self, shuffle_buffer_length: int = 100):
        return PseudoShuffledIterableDataset(self, shuffle_buffer_length)


class RestartableIteratorWrapper:
    def __init__(self, generator_func, *args, **kwargs):
        self.generator_func = generator_func
        self.args = args
        self.kwargs = kwargs
        self._length = None

    def __iter__(self):
        yield from self.generator_func(*self.args, **self.kwargs)


@dataclass
class TestDataset(IterableDataset, ShuffleMixin):
    """
    Dataset class for evaluating time series models. Handles both univariate and multivariate time series data.

    For multivariate data (e.g., 3D coordinates [x, y, z]):
    - All dimensions use the same time window (ensuring temporal consistency)
    - Dimensions are processed in order (x, y, z) to preserve coordinate ordering
    - Each dimension becomes a separate univariate time series

    Parameters
    ----------
    datasets : list
        List of datasets containing the original time series data.
    patch_size : int or None, optional
        Size of patches to extract from time series. Default is None.
    context_length : int, optional
        Length of context window for each sample. Default is 512.
    prediction_length : int, optional
        Length of prediction window for each sample. Default is 64.
    imputation_method : MissingValueImputation or None, optional
        Method for handling missing values. Default is LeavesMissingValues().
    np_dtype : numpy.dtype, optional
        Numpy data type for values. Default is np.float32.
    num_test_instances : int, optional
        Number of test instances to sample per series. Default is 1.
    window_style : str, optional
        Style of window sampling:
        - "sampled": randomly sample eval windows from each timeseries
        - "rolling": take sliding windows with a stride of window_stride
        - "single": get only the first context window from each timeseries
        - "fixed_start": get a single window starting at window_start_time
        Default is "sampled".
    window_stride : int, optional
        Stride length for rolling windows. Default is 1.
    window_start_time : int or None, optional
        Starting time index for fixed_start window style. Default is None.
    transforms : list[Callable] or None, optional
        List of transform functions to apply to data. Default is None.
    random_seed : int, optional
        Random seed for reproducibility. Default is 8097.
    """

    datasets: list
    patch_size: int | None = None
    context_length: int = 512
    prediction_length: int = 64
    imputation_method: MissingValueImputation | None = None
    np_dtype: np.dtype = np.dtype(np.float32)
    num_test_instances: int = 1
    window_style: str = "sampled"
    window_stride: int = 1
    window_start_time: int | None = None
    transforms: list[Callable] | None = None
    random_seed: int = 8097

    def __post_init__(self):
        super().__init__()
        self.imputation_method = self.imputation_method or LeavesMissingValues()
        self.eval_rng = np.random.default_rng(self.random_seed)
        self.dataset_class_name = self.datasets[0].__class__.__name__

    def reset_rng(self):
        self.eval_rng = np.random.default_rng(self.random_seed)

    def preprocess_iter(self, entry: Filter) -> Generator[dict, None, None]:
        for item in entry:
            target = np.asarray(item["target"], dtype=self.np_dtype)

            for transform in self.transforms or []:
                target = transform(target)

            yield {"start": item["start"], "target": target}

    def _create_instance_splitter(self):
        assert self.window_style in [
            "sampled",  # randomly sample eval windows from each timeseries
            "rolling",  # take sliding windows of context_length with a stride of window_stride from each timeseries
            "single",  # get only the first context window from each timeseries, predict the rest
            "fixed_start",  # get a single window starting at a specific time index
        ], "evaluation windows can only be sampled, rolling, single, or fixed_start"

        context_length = self.context_length
        prediction_length = self.prediction_length

        if self.window_style == "fixed_start":
            if self.window_start_time is None:
                raise ValueError("window_start_time must be specified when window_style is 'fixed_start'")
            test_sampler = partial(FixedStartSampler, start_time=self.window_start_time)
        else:
            test_sampler = {
                "sampled": partial(NumInstanceSampler, N=self.num_test_instances, rng=self.eval_rng),
                "rolling": partial(RegularWindowedSampler, stride=self.window_stride),
                "single": SingleContextSampler,
            }[self.window_style]

        instance_sampler = test_sampler(
            min_past=context_length,
            min_future=prediction_length,
        )

        prediction_length = MAX_PREDICTION_LENGTH if self.window_style == "single" else prediction_length
        return InstanceSplitter(
            target_field="target",
            is_pad_field="is_pad",
            start_field="start",
            forecast_start_field="forecast_start",
            instance_sampler=instance_sampler,
            past_length=context_length,
            future_length=prediction_length,
            dummy_value=np.nan,
        )

    def create_test_data(self, data, system_dimension: int = 1):
        if self.dataset_class_name == "GiftEvalDataset":
            # GiftEvalDataset.test_data yields pairs of input/label dicts
            def reformat_test_data_generator():
                # NOTE: data.windows should equal data.test_data.windows
                tot_num_series = data.hf_dataset.num_rows * data.test_data.windows
                num_series_to_sample = min(self.num_test_instances, tot_num_series)
                for input_dict, label_dict in itertools.islice(data.test_data, num_series_to_sample):
                    context_target = np.asarray(input_dict["target"], dtype=self.np_dtype)
                    future_target = np.asarray(label_dict["target"], dtype=self.np_dtype)
                    if system_dimension == 1:  # or context_target.ndim == 1
                        yield {
                            "past_target": context_target,
                            "future_target": future_target,
                        }
                    else:
                        # context_target is of shape (system_dimension, context_length)
                        # future_target is of shape (system_dimension, prediction_length)
                        for dim_idx in range(system_dimension):
                            yield {
                                "past_target": context_target[dim_idx, :],  # shape (context_length,)
                                "future_target": future_target[dim_idx, :],  # shape (prediction_length,)
                            }

            return reformat_test_data_generator()
        else:
            return self._create_instance_splitter().apply(data, is_train=False)

    def to_hf_format_eval(self, entry: dict) -> dict:
        # shape (1, contex_length)
        past_target = torch.tensor(entry["past_target"]).unsqueeze(0)
        future_target = torch.tensor(entry["future_target"]).unsqueeze(0)
        return {
            "past_values": past_target,
            "future_values": future_target,
        }

    def __iter__(self) -> Iterator:
        """
        Iterator that yields univariate time series samples.

        For multivariate data (e.g., 3D coordinates [x, y, z]):
        - All dimensions use the same time window (ensuring temporal consistency)
        - Dimensions are processed in order (x, y, z) to preserve coordinate ordering
        - Each dimension becomes a separate univariate time series
        """

        # Special handling for GiftEvalDataset - bypass normal preprocessing
        if self.dataset_class_name == "GiftEvalDataset":
            system_dimension = self.datasets[0].target_dim
            iterables = [self.create_test_data(dataset, system_dimension=system_dimension) for dataset in self.datasets]
            iterators = list(map(iter, iterables))
            # cyclers aren't used here, so just chain iterators sequentially
            chained_iterators = itertools.chain(*iterators)
            for entry in chained_iterators:
                yield self.to_hf_format_eval(entry)
            return

        preprocessed_datasets = [RestartableIteratorWrapper(self.preprocess_iter, dataset) for dataset in self.datasets]

        # TODO: wrap this functionality into self.create_test_data
        # Check if we have multivariate data
        is_multivariate = False
        for dataset in preprocessed_datasets:
            for item in dataset:
                if item["target"].ndim > 1:
                    is_multivariate = True
                break
            break

        # TODO: wrap this functionality into self.create_test_data
        if is_multivariate:
            # For multivariate data, we need to handle it specially
            for dataset in preprocessed_datasets:
                for item in dataset:
                    # Apply the instance splitter to get the sampled windows
                    splitter = self._create_instance_splitter()
                    sampled_data = splitter.apply([item], is_train=False)

                    # For each sampled window, split into univariate series
                    for sampled_item in sampled_data:
                        past_target = sampled_item["past_target"]  # shape: (context_length, dim)
                        future_target = sampled_item["future_target"]  # shape: (prediction_length, dim)

                        # Split into univariate series for each dimension
                        # This preserves the original dimension ordering (e.g., [x, y, z])
                        # All dimensions use the same time window from the sampled_item
                        context_dimension = past_target.shape[1]
                        for dim_idx in range(context_dimension):
                            univariate_past = past_target[:, dim_idx]  # shape: (context_length,)
                            univariate_future = future_target[:, dim_idx]  # shape: (prediction_length,)

                            # Create the entry in the expected format
                            entry = {
                                "past_target": univariate_past,
                                "future_target": univariate_future,
                            }
                            yield self.to_hf_format_eval(entry)
        else:
            # Univariate case - use the original logic
            iterables = [self.create_test_data(dataset) for dataset in preprocessed_datasets]
            iterators = list(map(iter, iterables))
            # cyclers aren't used here, so just chain iterators sequentially
            for entry in itertools.chain(*iterators):
                yield self.to_hf_format_eval(entry)


### GIFT-EVAL DATASET ###


class Term(Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        if self == Term.SHORT:
            return 1
        elif self == Term.MEDIUM:
            return 10
        elif self == Term.LONG:
            return 15


def itemize_start(data_entry: DataEntry) -> DataEntry:
    data_entry["start"] = data_entry["start"].item()
    return data_entry


class GiftEvalMultivariateToUnivariate(Transformation):
    def __init__(self, field):
        self.field = field

    def __call__(self, data_it: Iterable[DataEntry], is_train: bool = False) -> Iterator:
        for data_entry in data_it:
            item_id = data_entry["item_id"]
            val_ls = list(data_entry[self.field])
            for id, val in enumerate(val_ls):
                univariate_entry = data_entry.copy()
                univariate_entry[self.field] = val
                univariate_entry["item_id"] = item_id + "_dim" + str(id)
                yield univariate_entry


class GiftEvalDataset:
    def __init__(
        self,
        name: str,
        term: Term | str = Term.SHORT,
        to_univariate: bool = False,
        data_dir: str = "data",
    ):
        self.hf_dataset = datasets.load_from_disk(str(Path(data_dir) / name)).with_format("numpy")
        process = ProcessDataEntry(
            self.freq,
            one_dim_target=self.target_dim == 1,
        )

        self.gluonts_dataset = Map(compose(process, itemize_start), self.hf_dataset)
        if to_univariate:
            self.gluonts_dataset = GiftEvalMultivariateToUnivariate("target").apply(self.gluonts_dataset)

        self.term = Term(term)
        self.name = name

    @cached_property
    def prediction_length(self) -> int:
        freq = norm_freq_str(to_offset(self.freq).name)
        pred_len = M4_PRED_LENGTH_MAP[freq] if "m4" in self.name else PRED_LENGTH_MAP[freq]
        return self.term.multiplier * pred_len

    @cached_property
    def freq(self) -> str:
        return self.hf_dataset[0]["freq"]

    @cached_property
    def target_dim(self) -> int:
        return target.shape[0] if len((target := self.hf_dataset[0]["target"]).shape) > 1 else 1

    @cached_property
    def past_feat_dynamic_real_dim(self) -> int:
        if "past_feat_dynamic_real" not in self.hf_dataset[0]:
            return 0
        elif len((past_feat_dynamic_real := self.hf_dataset[0]["past_feat_dynamic_real"]).shape) > 1:
            return past_feat_dynamic_real.shape[0]
        else:
            return 1

    @cached_property
    def windows(self) -> int:
        if "m4" in self.name:
            return 1
        w = math.ceil(TEST_SPLIT * self._min_series_length / self.prediction_length)
        return min(max(1, w), MAX_WINDOW)

    @cached_property
    def _min_series_length(self) -> int:
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(pc.list_flatten(pc.list_slice(self.hf_dataset.data.column("target"), 0, 1)))
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column("target"))
        return min(lengths.to_numpy())

    @cached_property
    def sum_series_length(self) -> int:
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(pc.list_flatten(self.hf_dataset.data.column("target")))
        else:
            lengths = pc.list_value_length(self.hf_dataset.data.column("target"))
        return sum(lengths.to_numpy())

    @property
    def training_dataset(self) -> TrainingDataset:
        training_dataset, _ = split(self.gluonts_dataset, offset=-self.prediction_length * (self.windows + 1))
        return training_dataset

    @property
    def validation_dataset(self) -> TrainingDataset:
        validation_dataset, _ = split(self.gluonts_dataset, offset=-self.prediction_length * self.windows)
        return validation_dataset

    def __iter__(self):
        """Make GiftEvalDataset iterable by iterating over gluonts_dataset."""
        yield from self.gluonts_dataset

    @property
    def test_data(self) -> TestData:
        _, test_template = split(self.gluonts_dataset, offset=-self.prediction_length * self.windows)
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )
        return test_data
