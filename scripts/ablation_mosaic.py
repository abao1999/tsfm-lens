import logging
import os
import pickle

import hydra
import numpy as np
import torch
from tsfm_lens.ablation import ablate_attention_head
from omegaconf import DictConfig
from tqdm import tqdm

from tsfm_lens.chronos.pipeline import ChronosPipelinetsfm_lens
from tsfm_lens.utils import make_ensemble_from_arrow_dir

logger = logging.getLogger(__name__)

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")


def compute_wasserstein_distance(dist1: np.ndarray | torch.Tensor, dist2: np.ndarray | torch.Tensor) -> torch.Tensor:
    """
    Compute the Wasserstein distance between two distributions

    Args:
        dist1: torch.Tensor, shape (time_steps, vocab_size)
        dist2: torch.Tensor, shape (time_steps, vocab_size)

    Returns:
        wasserstein_distance: torch.Tensor, shape (time_steps)
    """
    from scipy.stats import wasserstein_distance

    assert dist1.shape == dist2.shape
    x_values = torch.arange(dist1.shape[-1])

    if isinstance(dist1, torch.Tensor):
        dist1 = dist1.cpu().numpy()
    if isinstance(dist2, torch.Tensor):
        dist2 = dist2.cpu().numpy()

    # compute the wasserstein distance for each time step
    wasserstein_distances = []
    for i in range(dist1.shape[0]):
        wasserstein_distances.append(wasserstein_distance(x_values, x_values, dist1[i], dist2[i]))
    return torch.tensor(wasserstein_distances)


def average_scores(outputs):
    """
    Average the scores across the samples

    Args:
        outputs: list of outputs from the model

    Returns:
        average_scores: torch.Tensor, shape (num_tokens, vocab_size)
    """
    # Compute probabilities from scores
    scores = []
    for output in outputs:
        for score in output.scores:
            scores.append(score)

    probs = torch.nn.functional.softmax(torch.stack(scores), dim=-1)

    # Average over the samples
    return torch.mean(probs, dim=0)


def compute_wasserstein_mosaic(
    pipeline: ChronosPipelinetsfm_lens,
    context: torch.Tensor,
    prediction_length: int,
    num_samples: int = 20,
    ablation_method: str = "zero",
    progress_bar: bool = True,
) -> torch.Tensor:
    """
    Compute the Wasserstein distance between the baseline and ablated model outputs.

    Args:
        pipeline: ChronosPipelinetsfm_lens, the model to ablate
        context: torch.Tensor, the context to predict from
        prediction_length: int, the length of the prediction to make
        num_samples: int, the number of samples to use for the prediction
        ablation_method: str, the method to use for the ablation. If ends with "-r", then
                            only one head is not ablated at a time, all other heads are ablated.
        progress_bar: bool, whether to show a progress bar

    Returns:
        wasserstein_results: torch.Tensor, shape (num_layers, num_heads)
    """

    # Get baseline prediction with unablated model
    _, baseline_outputs = pipeline.predict_with_full_outputs(
        context,
        prediction_length=prediction_length,
        limit_prediction_length=False,
        num_samples=num_samples,
        output_scores=True,
        return_dict_in_generate=True,
    )

    # Compute baseline probabilities
    baseline_probs = average_scores(baseline_outputs).to("cpu")

    # Initialize results tensor: [layers, heads]
    n_layers = pipeline.model.model.config.num_layers
    n_heads = pipeline.model.model.config.num_heads
    wasserstein_results = torch.zeros(n_layers, n_heads)

    # Run ablation for each head
    total_iterations = n_layers * n_heads
    if progress_bar:
        pbar = tqdm(total=total_iterations, desc="Ablation Progress")

    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            if ablation_method.endswith("-r"):
                heads_to_ablate = [
                    (i, j) for i in range(n_layers) for j in range(n_heads) if (i != layer_idx or j != head_idx)
                ]
            else:
                heads_to_ablate = [(layer_idx, head_idx)]

            # Run ablation
            base_ablation_method = ablation_method.split("-")[0]

            # TODO: replace with CircuitLensChronos
            _, ablated_outputs = ablate_attention_head(
                context,
                pipeline,
                heads_to_ablate=heads_to_ablate,
                max_new_tokens=prediction_length,
                num_samples=num_samples,
                ablation_method=base_ablation_method,
            )

            # Compute ablated probabilities
            ablated_probs = average_scores(ablated_outputs).to("cpu")

            # Compute wasserstein distance
            wasserstein_dist = compute_wasserstein_distance(baseline_probs, ablated_probs)

            # Store average wasserstein distance
            wasserstein_results[layer_idx, head_idx] = torch.mean(wasserstein_dist)

            # Update progress bar
            if progress_bar:
                pbar.update(1)

    if progress_bar:
        pbar.close()

    return wasserstein_results


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    # Extract config values
    model_name = cfg.ablation_mosaicmodel_name
    data_dir = cfg.ablation_mosatsfm_lensta_dir
    context_length = cfg.ablation_mosaiccontext_length
    prediction_length = cfg.ablation_mosaicprediction_length
    num_samples = cfg.ablation_mosaicnum_samples
    ablation_methods = cfg.ablation_mosaicmethods
    output_dir = cfg.ablation_mosaicoutput_dir
    subsample_interval = cfg.ablation_mosaicsubsample_interval

    split_name = "base40"
    split_dir = os.path.join(DATA_DIR, split_name)

    ensemble = make_ensemble_from_arrow_dir(split_dir, dyst_names_lst=["Lorenz"], num_samples=num_samples)
    # Prepare input series
    dyst_coords = torch.tensor(ensemble["Lorenz"][0, 0])[::subsample_interval].unsqueeze(0)
    context = dyst_coords[:, :context_length]

    for ablation_method in ablation_methods:
        pipeline = ChronosPipelinetsfm_lens.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        wasserstein_results = compute_wasserstein_mosaic(
            pipeline, context, prediction_length, num_samples, ablation_method
        )

        # Save results
        model_dir = os.path.join(output_dir, model_name.split("/")[-1])
        os.makedirs(model_dir, exist_ok=True)

        output_file = os.path.join(model_dir, f"{ablation_method}_wasserstein_mosaic.pkl")

        with open(output_file, "wb") as f:
            pickle.dump(wasserstein_results, f)


if __name__ == "__main__":
    main()
