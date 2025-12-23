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
#   layers                 - Array of layer specs to ablate. Each element can be:
#                            - Single layer: "[1]"
#                            - Multiple layers together: "[1,2,3]"
#   num_heads              - Array of head counts to ablate per layer
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
gpu_index=1
term="all"
max_datasets="null"
data_dir="${WORK}/data/gift-eval"

# Ablation grid parameters (bash arrays)
rseeds=(42)
ablated_components="[head]"

head_selection_strategy="srank"

# model_type="toto"
# layers=("[1]" "[2]" "[3]" "[4]" "[5]" "[6]" "[7]" "[8]" "[10]")

model_type="chronos_bolt"
# layers=("[0]" "[1]" "[2]" "[3]" "[4]" "[5]" "[6]" "[7]" "[8]" "[9]" "[10]" "[11]")
layers=("[4]" "[5]" "[6]" "[7]" "[9]" "[10]" "[11]")
# layers=("[0]" "[1]" "[2]" "[3]" "[8]")

num_heads=(11 9 7 5 3)

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
echo "  Layers: ${layers[*]}"
echo "  Num heads: ${num_heads[*]}"
echo "  Seeds: ${rseeds[*]}"
echo "  Components: ${ablated_components}"
echo "  Head selection: ${head_selection_strategy}"
echo "============================================"

for layer_spec in "${layers[@]}"; do
    for rseed in "${rseeds[@]}"; do
        for n in "${num_heads[@]}"; do
            echo ""
            echo ">>> Running: layers=${layer_spec}, heads=${n}, seed=${rseed}"
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

            echo ">>> Completed: layers=${layer_spec}, heads=${n}, seed=${rseed}"
        done
    done
done


echo ""
echo "============================================"
echo "All evaluations complete!"
echo "============================================"
