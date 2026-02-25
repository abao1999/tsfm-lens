#!/bin/bash
#
# run_ablations_gift-eval.sh - Run ablation evaluations on GIFT-Eval benchmark
#
# NOTE: this script provides no layer-level control of when to ablate both MLPs and heads.
#        ... until future updates, the user must commit the the same strategy for ablating all layers
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
gpu_index=1
term="all"
max_datasets="null"
data_dir="${WORK}/data/gift-eval"
batch_size=512

# Ablation grid parameters (bash arrays)
rseeds=(42)
ablated_components="[head]"

# TODO: use ablated_components as the switch instead i.e. ablated_components == "null" means do original evaluation
head_selection_strategy="null" # "null" to disable ablations

model_type="chronos2"

# spec: layer -> space-separated num_heads to run
# Generate target_ablations with all layers and all num_heads
declare -A target_ablations
# 1, ..., max_num_heads, null
num_heads_str="$(seq -s ' ' 1 11) null"
layer_lst=(0)
# for layer in {0..11}; do
for layer in "${layer_lst[@]}"; do
    target_ablations[$layer]="$num_heads_str"
done

if [ "$head_selection_strategy" != "null" ]; then
    echo "target_ablations keys: ${!target_ablations[*]}"
    echo "target_ablations: ${target_ablations[*]}"
fi

# =============================================================================
# MODEL SETUP
# =============================================================================
declare -A model_names=(
    ["timesfm"]="google/timesfm-2.5-200m-pytorch"
    ["chronos_bolt"]="amazon/chronos-bolt-base"
    ["chronos"]="amazon/chronos-t5-base"
    ["chronos2"]="amazon/chronos-2"
    ["toto"]="Datadog/Toto-Open-Base-1.0"
    ["moirai"]="Salesforce/moirai-1.1-R-base"
)

model_name="${model_names[$model_type]}"
if [ -z "$model_name" ]; then
    echo "Error: Invalid model type '${model_type}'"
    echo "Valid options: ${!model_names[*]}"
    exit 1
fi

model_name_str="${model_name//\//-}"

toto_num_samples=20
moirai_num_samples=100
chronos_num_samples=20
# Model-specific arguments
declare -A model_args_map=(
    ["chronos_bolt"]="chronos_bolt.model_id=${model_name} chronos_bolt.limit_prediction_length=false"
    ["chronos"]="chronos.model_id=${model_name} chronos.limit_prediction_length=false chronos.num_samples=${chronos_num_samples} chronos.deterministic=false"
    ["timesfm"]="timesfm.model_id=${model_name}"
    ["toto"]="toto.model_id=${model_name} toto.samples_per_batch=${toto_num_samples} toto.use_kv_cache=true toto.pad_short_series=false"
    ["moirai"]="moirai.model_id=${model_name} moirai.patch_size=32 moirai.num_samples=${moirai_num_samples}"
    ["chronos2"]="chronos2.model_id=${model_name}"
)

# Append num_samples to model_name_str for models that use sampling
case "$model_type" in
    toto)    model_name_str="${model_name_str}_samples-${toto_num_samples}" ;;
    chronos) model_name_str="${model_name_str}_samples-${chronos_num_samples}" ;;
    moirai)  model_name_str="${model_name_str}_samples-${moirai_num_samples}" ;;
esac

read -ra model_args <<< "${model_args_map[$model_type]}"

# =============================================================================
# BASE CONFIGURATION
# =============================================================================
base_args=(
    eval.dataset_name=gift-eval
    eval.data_dir="${data_dir}"
    eval.gift_eval.dataset_names=null # NOTE: null is default (all datasets), but for testing/debugging, can specify a set of datasets e.g. ['bizitobs_service']
    eval.gift_eval.max_num_datasets="${max_datasets}"
    eval.gift_eval.term="${term}"
    eval.gift_eval.to_univariate=false
    eval.device="cuda:${gpu_index}"
    eval.results_save_dir="${HOME}/tsfm-lens/results"
    eval.batch_size="${batch_size}"
    ablation.model_name_str="${model_name_str}"
    ablation.model_type="${model_type}"
)
echo "base_args: ${base_args[*]}"

# =============================================================================
# RUN ABLATION GRID
# =============================================================================
if [ "$head_selection_strategy" != "null" ]; then
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

else
    echo "============================================"
    echo "No ablation evaluation configuration provided"
    echo "============================================"
    echo "Running original evaluation"
    for rseed in "${rseeds[@]}"; do
        echo ">>> Running: seed=${rseed}"
        echo "--------------------------------------------"
        python scripts/run_ablations_gift-eval.py \
            "${base_args[@]}" \
            "${model_args[@]}" \
            eval.rseed="${rseed}" \
            ablation.head_selection_strategy="${head_selection_strategy}" \
        && wait
        echo ">>> Completed: seed=${rseed}"
    done
fi

echo ""
echo "============================================"
echo "All evaluations complete!"
echo "============================================"
