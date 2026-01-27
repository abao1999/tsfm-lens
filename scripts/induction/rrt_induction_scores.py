import gc
import logging
import os
import pickle

import hydra
import torch
from tqdm import tqdm

from tsfm_lens.chronos.circuitlens import CircuitLensChronos
from tsfm_lens.utils import set_seed


def generate_rrt_sequences(
    vocab_range: tuple[int, int] = (1911, 2187),
    batch_size: int = 100,
    num_unique_sequences: int = 1,
    repeat_factor: int = 4,
    extension: int = 0,
    sub_extension: int = 1,
    sequence_length: int = 10,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate data for repeated random token (RRT) induction experiments.

    Args:
        vocab_range: Tuple of (min, max) vocab indices to use
        batch_size: Number of sequences in the batch
        num_unique_sequences: Number of unique sequences in the batch
        repeat_factor: Number of times to repeat the sequences
        extension: After repeating everything, repeat this many sequences again
        sub_extension: Some additional tokens in the next sequence
        sequence_length: Length of each sequence
        device: Device to put tensors on

    Returns:
        token_ids: The token ids of the data.
        attention_mask: The attention mask of the data.
        decoder_input_ids: The decoder input ids of the data.
    """
    # Define constants

    # Define the vocab as all the tokens within the specified range
    vocab = torch.tensor([i for i in range(4096) if i >= vocab_range[0] and i <= vocab_range[1]])

    # Generate random token sequences (inlined from rrt.generate_random_token_ids)
    tokens = []
    for _ in range(num_unique_sequences):
        # random choice of vocab indices
        indices = torch.randint(0, vocab.shape[0], (batch_size, sequence_length))
        # select the tokens
        token_sequence = vocab[indices]
        tokens.append(token_sequence)

    # Stack sequences with repetition pattern (inlined from rrt.stack_sequences)
    sequences_to_stack = tokens * repeat_factor + tokens[:extension] + [tokens[extension][:, :sub_extension]]
    token_ids = torch.cat(sequences_to_stack, dim=-1)

    # Create attention mask
    attention_mask = torch.ones_like(token_ids, dtype=torch.bool)

    # Create decoder input ids
    decoder_input_ids = torch.cat(
        [
            torch.zeros((batch_size, 1), dtype=torch.long),
            tokens[extension][:, sub_extension : sub_extension + 1],
        ],
        dim=-1,
    )

    token_ids = token_ids.to(device)
    attention_mask = attention_mask.to(device)
    decoder_input_ids = decoder_input_ids.to(device)

    return token_ids, attention_mask, decoder_input_ids


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    # single source of truth for device placement
    gpu_count = torch.cuda.device_count()
    if torch.cuda.is_available():
        if hasattr(cfg.eval, "device") and cfg.eval.device is not None:
            target_device = torch.device(cfg.eval.device)
        elif gpu_count > 1:
            target_device = torch.device("cuda:1")
        else:
            target_device = torch.device("cuda:0")
    else:
        target_device = torch.device("cpu")

    if target_device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.95, device=target_device)
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    set_seed(cfg.seed)

    # Define different values to experiment with
    repeat_factors = cfg.induction_scores.repeat_factors
    sequence_lengths = cfg.induction_scores.sequence_lengths

    model_names = cfg.induction_scores.model_names

    total_iterations = len(repeat_factors) * len(sequence_lengths) * len(model_names)
    std_factor = float(cfg.induction_scores.batch_size) ** 0.5

    EOS_TOKEN_ID = torch.tensor(1, dtype=torch.int)

    with tqdm(total=total_iterations, desc="Processing configuations and models") as pbar:
        for repeat_factor in repeat_factors:
            for sequence_length in sequence_lengths:
                print(f"Processing repeat_factor={repeat_factor}, sequence_length={sequence_length}")

                num_unique_sequences = cfg.induction_scores.num_unique_sequences
                extension = cfg.induction_scores.extension
                sub_extension = cfg.induction_scores.sub_extension

                # Generate data with current configuration
                token_ids, attention_mask, decoder_input_ids = generate_rrt_sequences(
                    vocab_range=cfg.induction_scores.vocab_range,
                    batch_size=cfg.induction_scores.batch_size,
                    num_unique_sequences=num_unique_sequences,
                    repeat_factor=repeat_factor,
                    extension=extension,
                    sub_extension=sub_extension,
                    sequence_length=sequence_length,
                    device=target_device,
                )
                eos_value = EOS_TOKEN_ID.to(device=target_device, dtype=token_ids.dtype)
                eos_column = (
                    torch.ones((token_ids.size(0), 1), dtype=token_ids.dtype, device=target_device) * eos_value
                )
                token_ids = torch.cat([token_ids, eos_column], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(eos_column, dtype=attention_mask.dtype)], dim=-1
                )

                t_idx = decoder_input_ids.shape[-1] - 1  # current token in decoder input ids
                s_idx = (
                    sequence_length * ((repeat_factor - 1) * num_unique_sequences + extension) + sub_extension
                )  # most recent previous token in encoder input ids

                # Store results for each model
                config_key = f"rf{repeat_factor}_sl{sequence_length}"

                for model_name in model_names:
                    pbar.set_description(f"Processing {model_name} for RF={repeat_factor}, SL={sequence_length}")

                    pipeline = CircuitLensChronos.from_pretrained(
                        model_name,
                        device_map=str(target_device) if target_device.type == "cuda" else target_device,
                        torch_dtype=torch.bfloat16,
                    )
                    num_layers = pipeline.model.model.config.num_decoder_layers
                    num_heads = pipeline.model.model.config.num_heads

                    heads_to_attribute = [(layer, head) for layer in range(num_layers) for head in range(num_heads)]
                    pipeline.add_head_attribution_hooks(heads_to_attribute)

                    t5_model = pipeline.model.model

                    # Initialize tensor variables before branching
                    mosaic_center_mean = torch.zeros((num_layers, num_heads))
                    mosaic_right_mean = torch.zeros((num_layers, num_heads))
                    correct_token_attribution_mean = torch.zeros((num_layers, num_heads))

                    mosaic_center_std = torch.zeros((num_layers, num_heads))
                    mosaic_right_std = torch.zeros((num_layers, num_heads))
                    correct_token_attribution_std = torch.zeros((num_layers, num_heads))

                    # Original code for processing the full batch
                    outputs = t5_model.generate(
                        input_ids=token_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1,
                        decoder_input_ids=decoder_input_ids,
                        num_return_sequences=1,
                        do_sample=False,
                        use_cache=False,
                        output_attentions=True,
                        output_scores=True,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                    )  # type: ignore

                    # Extract cross attention probabilities
                    # cross_attentions is a list of length layers, each with shape [batch, heads, dec_length, enc_length]
                    cross_attn_probs = outputs.cross_attentions

                    # Extract overall scores for correct tokens (token right of the last occurrence)
                    correct_tokens = token_ids[:, s_idx + 1]
                    token_scores = torch.gather(outputs.scores[0], dim=-1, index=correct_tokens.unsqueeze(1)).squeeze()

                    overall_scores_mean = float(token_scores.to("cpu").mean())
                    overall_scores_std = float(token_scores.to("cpu").std()) / std_factor

                    for layer in range(num_layers):
                        for head in range(num_heads):
                            center_attn = cross_attn_probs[0][layer][:, head, t_idx, s_idx]
                            mosaic_center_mean[layer][head] = float(center_attn.to("cpu").mean())
                            mosaic_center_std[layer][head] = float(center_attn.to("cpu").std()) / std_factor

                            right_attn = cross_attn_probs[0][layer][:, head, t_idx, s_idx + 1]
                            mosaic_right_mean[layer][head] = float(right_attn.to("cpu").mean())
                            mosaic_right_std[layer][head] = float(right_attn.to("cpu").std()) / std_factor

                            # logit attribution for the current head and layer
                            #  NOTE: using cross-attention (CA) head atttribution, double check
                            attributed_logits = pipeline.unembed_residual(
                                pipeline.ca_head_attribution_outputs[layer][head][0][:, -1, :]
                            )
                            correct_token_attribution = torch.gather(
                                attributed_logits,
                                dim=1,
                                index=correct_tokens.unsqueeze(1),
                            )  # attribution for the correct token

                            correct_token_attribution_mean[layer][head] = float(
                                correct_token_attribution.detach().to("cpu").mean()
                            )
                            correct_token_attribution_std[layer][head] = (
                                float(correct_token_attribution.detach().to("cpu").std()) / std_factor
                            )

                    # Clean up after full batch processing
                    del outputs
                    del cross_attn_probs
                    gc.collect()
                    if target_device.type == "cuda":
                        torch.cuda.empty_cache()

                    # Clean model name without the 'amazon/' prefix
                    clean_model_name = model_name.split("/")[-1] if "/" in model_name else model_name

                    # Create directory structure
                    base_dir = os.path.join(
                        "outputs",
                        cfg.induction_scores.rrt_scores_dir,
                        clean_model_name,
                        config_key,
                    )
                    os.makedirs(base_dir, exist_ok=True)

                    # Save individual files
                    center_scores = {
                        "mean": mosaic_center_mean,
                        "std": mosaic_center_std,
                    }
                    right_scores = {"mean": mosaic_right_mean, "std": mosaic_right_std}
                    correct_token_attribution = {
                        "mean": correct_token_attribution_mean,
                        "std": correct_token_attribution_std,
                    }
                    

                    rrt_vars = {
                        "std_factor": std_factor,
                        "overall_scores": (overall_scores_mean, overall_scores_std),
                    }

                    with open(os.path.join(base_dir, "center_scores.pkl"), "wb") as f:
                        pickle.dump(center_scores, f)

                    with open(os.path.join(base_dir, "right_scores.pkl"), "wb") as f:
                        pickle.dump(right_scores, f)

                    with open(os.path.join(base_dir, "correct_token_attribution.pkl"), "wb") as f:
                        pickle.dump(correct_token_attribution, f)

                    with open(os.path.join(base_dir, "rrt_vars.pkl"), "wb") as f:
                        pickle.dump(rrt_vars, f)

                    # Clean up the pipeline
                    pipeline.remove_all_hooks()
                    del pipeline
                    del t5_model
                    gc.collect()
                    if target_device.type == "cuda":
                        torch.cuda.empty_cache()

                    pbar.update(1)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    main()
