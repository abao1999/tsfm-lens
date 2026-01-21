#!/bin/bash
#
# run_ablations_gift-eval_to_heads1pp.sh - Run "heads at 1pp" ablation evaluations on GIFT-Eval
#
# Description:
#   Runs ablation experiments using the "heads at 1 principal component" (heads1pp) strategy.
#   This mode loads precomputed head ablation configs from a JSON file and applies them to
#   bring attention heads to 1pp performance. Optionally skips the last N heads per layer
#   to preserve some capacity, except for protected layers.
#
# Configuration:
#   gpu_index                    - GPU device index to use
#   model_type                   - Model architecture: chronos_bolt, chronos, timesfm, toto
#   rseeds                       - Array of random seeds for reproducibility
#   ablated_components           - Components to ablate (e.g., "[head,mlp]")
#   chosen_layers                - Array of layer indices to include in head ablation
#   chosen_layers_mlp            - Array of layer indices for MLP ablation
#   num_heads_per_layer_to_skip  - Number of heads to skip (keep active) per layer
#   layers_to_keep_at_heads1pp   - Layers where all heads are ablated (no skipping)
#   term                         - GIFT-Eval term filter (short, medium, long, all)
#   max_datasets                 - Maximum number of datasets to evaluate (null for all)
#
# Usage:
#   ./scripts/run_ablations_gift-eval_to_heads1pp.sh
#

set -e
ulimit -n 99999

# =============================================================================
# CONFIGURATION
# =============================================================================
gpu_index=3
term="all"
max_datasets="null"
data_dir="${WORK}/data/gift-eval"

# Ablation grid parameters (bash arrays)
rseeds=(42)
ablated_components="[head,mlp]"

head_selection_strategy="null"

# model_type="timesfm"

# chosen_layers=($(seq 0 19))
# # chosen_layers=(7 8 9 10 11 12 13)
# echo "chosen_layers: ${chosen_layers[*]}"
# chosen_layers_mlp=(10 11)
# echo "chosen_layers_mlp: ${chosen_layers_mlp[*]}"

# num_heads_per_layer_to_skip=1
# echo "num_heads_per_layer_to_skip: ${num_heads_per_layer_to_skip}"
# layers_to_keep_at_heads1pp=(7 8 9 10 11 12 13)
# echo "layers_to_keep_at_heads1pp: ${layers_to_keep_at_heads1pp[*]}"


model_type="chronos_bolt"

chosen_layers=($(seq 1 8))
# chosen_layers=($(seq 2 5))
# chosen_layers=(7 8 9 10 11 12 13)
echo "chosen_layers: ${chosen_layers[*]}"
# chosen_layers_mlp=($(seq 3 8))
# chosen_layers_mlp=($(seq 2 6))
# chosen_layers_mlp=($(seq 1 4))
chosen_layers_mlp=($(seq 3 6))
echo "chosen_layers_mlp: ${chosen_layers_mlp[*]}"

num_heads_per_layer_to_skip=2
echo "num_heads_per_layer_to_skip: ${num_heads_per_layer_to_skip}"
# layers_to_keep_at_heads1pp=()
# layers_to_keep_at_heads1pp=($(seq 1 5))
layers_to_keep_at_heads1pp=($(seq 1 4))
# layers_to_keep_at_heads1pp=($(seq 2 5))
echo "layers_to_keep_at_heads1pp: ${layers_to_keep_at_heads1pp[*]}"

# =============================================================================
# MODEL SETUP
# =============================================================================
declare -A model_names=(
    ["timesfm"]="google/timesfm-2.5-200m-pytorch"
    ["chronos_bolt"]="amazon/chronos-bolt-base"
    ["chronos"]="amazon/chronos-t5-base"
    ["toto"]="Datadog/Toto-Open-Base-1.0"
)

model_name="${model_names[$model_type]}"
if [ -z "$model_name" ]; then
    echo "Error: Invalid model type '${model_type}'"
    echo "Valid options: ${!model_names[*]}"
    exit 1
fi

model_name_str="${model_name//\//-}"

# Model-specific arguments
declare -A model_args_map=(
    ["chronos_bolt"]="chronos_bolt.model_id=${model_name} chronos_bolt.limit_prediction_length=false"
    ["chronos"]="chronos.model_id=${model_name} chronos.limit_prediction_length=false chronos.num_samples=10 chronos.deterministic=false"
    ["timesfm"]="timesfm.model_id=${model_name}"
    ["toto"]="toto.model_id=${model_name} toto.samples_per_batch=20 toto.use_kv_cache=true toto.pad_short_series=false"
)

# Special handling for toto: append samples_per_batch to model_name_str
if [ "$model_type" = "toto" ]; then
    model_name_str="${model_name_str}_samples-20"
fi

read -ra model_args <<< "${model_args_map[$model_type]}"

# =============================================================================
# BASE CONFIGURATION
# =============================================================================
base_args=(
    eval.dataset_name=gift-eval
    eval.data_dir="${data_dir}"
    eval.gift_eval.dataset_names=null
    eval.gift_eval.max_num_datasets="${max_datasets}"
    eval.gift_eval.term="${term}"
    eval.gift_eval.to_univariate=false
    eval.device="cuda:${gpu_index}"
    eval.results_save_dir="${HOME}/tsfm-lens/results"
    ablation.model_name_str="${model_name_str}"
    ablation.model_type="${model_type}"
)

# =============================================================================
# RUN ABLATION GRID
# =============================================================================
echo "============================================"
echo "Ablation Evaluation Configuration"
echo "============================================"
echo "  Model: ${model_name} (${model_type})"
echo "  GPU: cuda:${gpu_index}"
echo "  Term: ${term}"
echo "  Layers: ${!target_ablations[*]}"
echo "  Seeds: ${rseeds[*]}"
echo "  Head selection: ${head_selection_strategy}"
echo "============================================"

for rseed in "${rseeds[@]}"; do
    # Convert bash arrays to Hydra list format [a,b,c,...]
    chosen_layers_hydra="[$(IFS=,; echo "${chosen_layers[*]}")]"
    layers_to_keep_hydra="[$(IFS=,; echo "${layers_to_keep_at_heads1pp[*]}")]"
    chosen_layers_mlp_hydra="[$(IFS=,; echo "${chosen_layers_mlp[*]}")]"

    python scripts/run_ablations_gift-eval.py \
        "${base_args[@]}" \
        "${model_args[@]}" \
        eval.rseed="${rseed}" \
        ablation.ablations_types="${ablated_components}" \
        ablation.head_selection_strategy="${head_selection_strategy}" \
        ablation.to_heads_at_1pp.chosen_layers="${chosen_layers_hydra}" \
        ablation.to_heads_at_1pp.num_heads_per_layer_to_skip="${num_heads_per_layer_to_skip}" \
        ablation.to_heads_at_1pp.layers_to_keep_at_heads1pp="${layers_to_keep_hydra}" \
        ablation.to_heads_at_1pp.chosen_layers_mlp="${chosen_layers_mlp_hydra}" \
        && wait

    echo ">>> Completed: seed=${rseed}"
    
done

echo "============================================"
echo "All evaluations complete!"
echo "============================================"