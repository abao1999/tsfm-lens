import itertools
import logging

import numpy as np
import torch
from gluonts.ev.metrics import BaseMetricDefinition
from gluonts.model import Forecast, evaluate_forecasts
from gluonts.model.forecast import QuantileForecast
from gluonts.time_feature import get_seasonality

from tsfm_lens.chronos2.circuitlens import CircuitLensChronos2
from tsfm_lens.dataset import GiftEvalDataset

logger = logging.getLogger(__name__)


class Chronos2Predictor:
    def __init__(
        self,
        pipeline: CircuitLensChronos2,
        prediction_length: int,
        batch_size: int,
        quantile_levels: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        predict_batches_jointly: bool = False,
        **kwargs,
    ):
        assert isinstance(pipeline, CircuitLensChronos2), (
            "This is Predictor is for Chronos-2, see other notebook for Chronos and Chronos-Bolt"
        )
        self.pipeline = pipeline
        if pipeline.pipeline is None:
            raise ValueError("Pipeline must be an instance of Chronos2Pipeline")
        self.prediction_length = prediction_length
        self.batch_size = batch_size
        self.quantile_levels = quantile_levels
        self.predict_batches_jointly = predict_batches_jointly

    def _pack_model_items(self, items):
        for item in items:
            model_input = {
                "target": item["target"],
            }
            yield model_input

    def predict(self, test_data_input) -> list[Forecast]:
        pipeline = self.pipeline
        model_batch_size = self.batch_size
        if self.predict_batches_jointly:
            logger.info(
                "Note: Using cross learning mode. Please ensure that different rolling windows of the same time series are not in `test_data_input` to avoid any potential leakage due to in-context learning."
            )

        # Generate forecasts
        forecast_outputs = []
        input_data = list(self._pack_model_items(test_data_input))
        is_univariate_data = input_data[0]["target"].ndim == 1  # homogenous across all intputs
        while True:
            try:
                quantiles, _ = pipeline.pipeline.predict_quantiles(  # type: ignore[attr-defined]
                    inputs=input_data,
                    prediction_length=self.prediction_length,
                    batch_size=model_batch_size,
                    quantile_levels=self.quantile_levels,
                    predict_batches_jointly=self.predict_batches_jointly,
                )
                quantiles = torch.stack(quantiles)
                # quantiles [batch, variates, seq_len, quantiles]
                quantiles = quantiles.permute(0, 3, 2, 1).cpu().numpy()
                # forecast_outputs [batch, quantiles, seq_len, variates]
                if is_univariate_data:
                    quantiles = quantiles.squeeze(-1)  # squeeze variate to avoid error in eval due to broadcasting
                assert quantiles.shape[1] == len(self.quantile_levels)
                assert quantiles.shape[2] == self.prediction_length
                forecast_outputs.append(quantiles)
                break
            except torch.cuda.OutOfMemoryError:
                logger.error(
                    f"OutOfMemoryError at model_batch_size {model_batch_size}, reducing to {model_batch_size // 2}"
                )
                model_batch_size //= 2

        # Convert forecasts into gluonts Forecast objects
        forecast_outputs = np.concatenate(forecast_outputs, axis=0)
        assert len(forecast_outputs) == len(input_data)
        forecasts = []
        for item, ts in zip(forecast_outputs, test_data_input):
            forecast_start_date = ts["start"] + len(ts["target"])
            forecast = QuantileForecast(
                forecast_arrays=item,
                forecast_keys=list(map(str, self.quantile_levels)),
                start_date=forecast_start_date,
            )
            forecasts.append(forecast)
        return forecasts


def evaluate_on_dataset(
    pipeline: CircuitLensChronos2,
    data_dir: str,
    ds_name: str,
    ds_term: str,
    batch_size: int,
    metrics: list[BaseMetricDefinition],
    use_multivariate_data: bool = True,
    **predictor_kwargs,
):
    is_multivariate_source = (
        GiftEvalDataset(
            name=ds_name,
            term=ds_term,
            to_univariate=False,
            data_dir=data_dir,
        ).target_dim
        > 1
    )

    dataset = GiftEvalDataset(
        name=ds_name,
        term=ds_term,
        to_univariate=is_multivariate_source and not use_multivariate_data,
        data_dir=data_dir,
    )

    predictor = Chronos2Predictor(
        pipeline=pipeline,
        prediction_length=dataset.prediction_length,
        batch_size=batch_size,
        **predictor_kwargs,
    )

    # Avoid cross batch leakage of rolling evalution by prediction of windows individually.
    forecast_windows = []
    n_windows = dataset.test_data.windows
    for window_idx in range(n_windows):
        entries_window_k = list(itertools.islice(dataset.test_data.input, window_idx, None, n_windows))
        forecasts_window_k = list(predictor.predict(entries_window_k))
        forecast_windows.append(forecasts_window_k)

    forecasts = [item for items in zip(*forecast_windows) for item in items]  # interleave results again
    season_length = get_seasonality(dataset.freq)
    return (
        evaluate_forecasts(
            forecasts,
            test_data=dataset.test_data,
            metrics=metrics,
            batch_size=1024,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length,
        )
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
