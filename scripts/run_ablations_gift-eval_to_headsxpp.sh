#!/bin/bash
#
# run_ablations_gift-eval_to_headsxpp.sh - Run "heads at xpp" ablation evaluations on GIFT-Eval
#
# Description:
#   Runs ablation experiments using the "heads at 1 principal component" (headsxpp) strategy.
#   This mode loads precomputed head ablation configs from a JSON file and applies them to
#   bring attention heads to xpp performance. Optionally skips the last N heads per layer
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
#   layers_to_keep_at_headsxpp   - Layers where all heads are ablated (no skipping)
#   term                         - GIFT-Eval term filter (short, medium, long, all)
#   max_datasets                 - Maximum number of datasets to evaluate (null for all)
#
# Usage:
#   ./scripts/run_ablations_gift-eval_to_headsxpp.sh
#

set -e
ulimit -n 99999

# =============================================================================
# CONFIGURATION
# =============================================================================
gpu_index=2
term="all"
max_datasets="null"
data_dir="${WORK}/data/gift-eval"
batch_size=1024

rseeds=(42)
ablated_components="[head,mlp]"

head_selection_strategy="headsxpp"
threshold_pct=1.0

model_type="moirai"

chosen_layers=(1 2 3 4 5)
echo "chosen_layers: ${chosen_layers[*]}"
chosen_layers_mlp=()
echo "chosen_layers_mlp: ${chosen_layers_mlp[*]}"

num_heads_per_layer_to_skip=0
echo "num_heads_per_layer_to_skip: ${num_heads_per_layer_to_skip}"
layers_to_keep_at_headsxpp=()
echo "layers_to_keep_at_headsxpp: ${layers_to_keep_at_headsxpp[*]}"

# =============================================================================
# MODEL SETUP
# =============================================================================
declare -A model_names=(
    ["timesfm"]="google/timesfm-2.5-200m-pytorch"
    ["chronos_bolt"]="amazon/chronos-bolt-base"
    ["chronos"]="amazon/chronos-t5-base"
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
    layers_to_keep_hydra="[$(IFS=,; echo "${layers_to_keep_at_headsxpp[*]}")]"
    chosen_layers_mlp_hydra="[$(IFS=,; echo "${chosen_layers_mlp[*]}")]"

    python scripts/run_ablations_gift-eval.py \
        "${base_args[@]}" \
        "${model_args[@]}" \
        eval.rseed="${rseed}" \
        ablation.ablations_types="${ablated_components}" \
        ablation.head_selection_strategy="${head_selection_strategy}" \
        ablation.to_heads_at_xpp.threshold_pct="${threshold_pct}" \
        ablation.to_heads_at_xpp.chosen_layers="${chosen_layers_hydra}" \
        ablation.to_heads_at_xpp.num_heads_per_layer_to_skip="${num_heads_per_layer_to_skip}" \
        ablation.to_heads_at_xpp.layers_to_keep_at_headsxpp="${layers_to_keep_hydra}" \
        ablation.to_heads_at_xpp.chosen_layers_mlp="${chosen_layers_mlp_hydra}" \
        && wait

    echo ">>> Completed: seed=${rseed}"
    
done

echo "============================================"
echo "All evaluations complete!"
echo "============================================"