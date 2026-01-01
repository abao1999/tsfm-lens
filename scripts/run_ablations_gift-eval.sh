#!/bin/bash
#
# run_ablations_gift-eval.sh - Run ablation evaluations on GIFT-Eval benchmark
#
# Description:
#   Runs a grid of ablation experiments by iterating over layers, random seeds,
#   and number of heads to ablate. Each run completes before the next starts.
#
# Configuration:
#   gpu_index              - GPU device index to use
#   model_type             - Model architecture: chronos_bolt, chronos, timesfm, toto
#   rseeds                 - Array of random seeds for reproducibility
#   ablated_components     - Components to ablate (e.g., "[head]")
#   head_selection_strategy- Strategy for selecting heads (e.g., "srank", "srank_reverse")
#   target_ablations       - Associative array mapping layer index to num_heads to ablate
#   term                   - GIFT-Eval term filter (short, medium, long, all)
#   max_datasets           - Maximum number of datasets to evaluate (null for all)
#
# Usage:
#   ./scripts/run_ablations_gift-eval.sh
#

set -e
ulimit -n 99999

# =============================================================================
# CONFIGURATION
# =============================================================================
gpu_index=0
term="all"
max_datasets="null"
data_dir="${WORK}/data/gift-eval"

# Ablation grid parameters (bash arrays)
rseeds=(42)
ablated_components="[head]"

head_selection_strategy="srank"

model_type="timesfm"

# Failed ablations: layer -> space-separated num_heads to re-run
declare -A target_ablations=(
    ["0"]="3 5 7 9 11 13 14 15"
    ["1"]="3 5"
    ["2"]="9 11 13 14"
    ["3"]="14"
    ["4"]="9 10 14"
    ["6"]="9 14"
    ["9"]="4 6 15"
    ["10"]="7 9 11 14 15"
    ["11"]="14"
    ["12"]="7 9 11 13 14 15"
    ["13"]="1 2 3 5 7 11 12 13 15"
    ["14"]="14"
    ["15"]="3 5 9 11 13 14 15"
    ["16"]="7 11 14 15 null"
    ["17"]="3 5 7 9 11"
    ["18"]="11 15"
    ["19"]="1 2 3 4 5 11 13 14"
)

echo "target_ablations: ${target_ablations[*]}"

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
echo "  Components: ${ablated_components}"
echo "  Head selection: ${head_selection_strategy}"
echo "============================================"

for layer in "${!target_ablations[@]}"; do
    layer_spec="[${layer}]"
    read -ra heads <<< "${target_ablations[$layer]}"
    for rseed in "${rseeds[@]}"; do
        for n in "${heads[@]}"; do
            echo ""
            echo ">>> Running: layer=${layer}, heads=${n}, seed=${rseed}"
            echo "--------------------------------------------"

            python scripts/run_ablations_gift-eval.py \
                "${base_args[@]}" \
                "${model_args[@]}" \
                eval.rseed="${rseed}" \
                ablation.ablations_types="${ablated_components}" \
                ablation.ablations_layers_lst="${layer_spec}" \
                ablation.ablate_n_heads_per_layer="${n}" \
                ablation.head_selection_strategy="${head_selection_strategy}" \
            && wait

            echo ">>> Completed: layer=${layer}, heads=${n}, seed=${rseed}"
        done
    done
done


echo ""
echo "============================================"
echo "All evaluations complete!"
echo "============================================"
