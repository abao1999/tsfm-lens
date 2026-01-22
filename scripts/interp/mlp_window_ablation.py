import logging
import os
import pickle

import hydra
import numpy as np
import torch
from numpy.fft import fft
from omegaconf import DictConfig
from tqdm import tqdm

from tsfm_lens.chronos.circuitlens import CircuitLensChronos

device = "cuda" if torch.cuda.is_available() else "cpu"
WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")

logger = logging.getLogger(__name__)


def sinusoidal(
    k: float | list[float],
    t_1: float = 1,
    steps: int = 100,
    amp: float | list[float] = 1,
    phase_shift: float = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a sinusoidal function with frequency k.

    Args:
        k: Frequency of the sinusoid in Hz or array of frequencies
        t_1: Total time period
        steps: Number of time steps
        amp: Amplitude of the sinusoid or array of amplitudes
        phase_shift: Phase shift in radians

    Returns:
        Tuple of (time array, sinusoidal values)
    """
    t = np.linspace(0, t_1, steps, endpoint=False)

    # Convert k to numpy array if it isn't already
    k_array = np.atleast_1d(k)

    # Handle amp: if k is array and amp is scalar, use the same amp for all frequencies
    if isinstance(amp, (int, float)) and len(k_array) > 1:
        amp_array = np.full_like(k_array, amp, dtype=float)
    else:
        amp_array = np.atleast_1d(amp)
        # Check that k and amp have the same length if amp is also an array
        if len(k_array) != len(amp_array):
            raise ValueError("k and amp must have the same length when passed as arrays")

    # Initialize y with zeros
    y = np.zeros(steps)

    # Sum the sinusoidal functions
    for k_i, amp_i in zip(k_array, amp_array):
        y += amp_i * np.sin(2 * np.pi * k_i * t + phase_shift)

    return t, y


def compute_second_order_fft(
    attns: dict[int, torch.Tensor], num_layers: int, num_heads: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compute the second-order FFT of attention head activations.

    Args:
        attns: Dictionary containing attention values per layer
        num_layers: Number of layers in the model
        num_heads: Number of attention heads per layer

    Returns:
        freqs: Frequency array for the second-order FFT
        second_order_ffts: Dictionary mapping head identifiers to second-order FFT arrays
    """
    # Dictionary to store second-order FFTs for each attention head
    second_order_ffts = {}

    # Get first attention head's FFT to determine the frequency range
    first_attn_fft = np.abs(fft(attns[0][0].detach().cpu().numpy()))
    freqs = np.fft.fftfreq(len(first_attn_fft))
    positive_freq_idxs = np.where(freqs > 0)[0]

    # Compute second-order FFT for each attention head
    for layer in range(num_layers):
        for head in range(num_heads):
            # First-order FFT of attention pattern
            attn_fft = np.abs(fft(attns[layer][head].detach().cpu().numpy()))

            # Second-order FFT (FFT of the FFT)
            second_fft = np.abs(fft(attn_fft))

            # Store only positive frequencies with layer-head identifier
            head_id = f"L{layer}H{head}"
            second_order_ffts[head_id] = second_fft[positive_freq_idxs]

    # Store frequencies for the second-order FFT (only positive frequencies)
    freq_values = freqs[positive_freq_idxs]

    return freq_values, second_order_ffts


def fft_with_ablated_mlp(
    pipeline: CircuitLensChronos,
    context: torch.Tensor,
    mlp_layers_to_ablate: list[int] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Run the experiment with specified MLP layers ablated.

    Args:
        pipeline: CircuitLensChronos pipeline
        context: Input context tensor
        mlp_layers_to_ablate: List of MLP layer indices to ablate

    Returns:
        freqs: Frequency array for second-order FFT
        second_order_ffts: Dictionary of second-order FFTs for all attention heads
    """
    # Reset any previous ablation hooks
    pipeline.remove_all_hooks()

    # Set up MLP ablation if specified
    if mlp_layers_to_ablate:
        pipeline.add_mlp_ablation_hooks(mlp_layers_to_ablate)
    else:
        pipeline.add_mlp_ablation_hooks([])

    # Run prediction
    prediction_length = 1
    preds, outputs = pipeline.predict_with_full_outputs(
        context=context,
        prediction_length=prediction_length,
        num_samples=1,
        do_sample=False,
        use_cache=False,
        return_dict_in_generate=True,
        output_attentions=True,
    )

    # Extract attention patterns
    rollout_idx = 0
    target_token = 0

    attns = {}
    num_layers = pipeline.model.model.config.num_decoder_layers
    num_heads = pipeline.model.model.config.num_heads
    for layer in range(num_layers):
        attns[layer] = outputs[rollout_idx].cross_attentions[0][layer][0, :, target_token, :-1]

    # Compute second-order FFTs
    second_order_freqs, second_order_ffts = compute_second_order_fft(attns, num_layers, num_heads)

    return second_order_freqs, second_order_ffts


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    model_names = cfg.mlp_window_ablation.model_names

    k_values = cfg.mlp_window_ablation.k_values
    amp_values = cfg.mlp_window_ablation.amp_values
    steps = cfg.mlp_window_ablation.steps
    t_1 = cfg.mlp_window_ablation.t_1
    output_dir = cfg.mlp_window_ablation.output_dir

    device = cfg.mlp_window_ablation.device

    for model_name in model_names:
        t, y = sinusoidal(k=k_values, amp=amp_values, steps=steps, t_1=t_1)
        context = torch.tensor(y).unsqueeze(0)

        result_dict = {}

        pipeline = CircuitLensChronos.from_pretrained(model_name, device_map=device)
        num_layers = pipeline.model.model.config.num_decoder_layers
        num_heads = pipeline.model.model.config.num_heads

        window_sizes = [i for i in range(1, num_layers + 1)]

        readable_model_name = model_name.split("/")[-1]
        model_output_dir = os.path.join(output_dir, readable_model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        for w in tqdm(window_sizes):
            result_dict = {}
            second_order_freqs, second_order_ffts = fft_with_ablated_mlp(pipeline, context)
            result_dict["freqs"] = second_order_freqs
            result_dict["none"] = second_order_ffts

            mlp_layers_to_ablate = list(range(w))
            second_order_freqs, second_order_ffts = fft_with_ablated_mlp(pipeline, context, mlp_layers_to_ablate)

            window_combinations = []
            # Single MLP ablations
            for i in range(w):
                window_combinations.append(([j for j in range(i)], f"0-{i}"))

            for start in range(1, num_layers - w + 1):
                window_combinations.append(
                    (
                        [j for j in range(start, start + w - 1)],
                        f"{start}-{start + w - 1}",
                    )
                )

            for ablation_layers, key in tqdm(window_combinations):
                mlp_layers_to_ablate = ablation_layers
                second_order_freqs, second_order_ffts = fft_with_ablated_mlp(pipeline, context, mlp_layers_to_ablate)
                result_dict[key] = second_order_ffts

                assert np.allclose(second_order_freqs, result_dict["freqs"]), f"Frequency mismatch for ablation {key}"

            with open(os.path.join(model_output_dir, f"w{w}_mlp_za.pkl"), "wb") as f:
                pickle.dump(result_dict, f)

        print(f"Experiment completed for {model_name}")


if __name__ == "__main__":
    main()
