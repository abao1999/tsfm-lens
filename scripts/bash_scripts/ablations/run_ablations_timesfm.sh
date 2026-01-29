#!/bin/bash

ulimit -n 99999

# Source shared utility functions
source "$(dirname "$0")/ablation_utils.sh"

# Parse command-line arguments
# Required arguments
dataset_name=$1
num_test_instances=$2
ablation_types_input=$3
n_consecutive_layers_input=$4
ablate_n_heads_per_layer=$5
gpu_index=$6
term=$7
prediction_length=$8

# Hardcoded window_start_time
window_start_time=2512

# Check required arguments
if [ -z "$dataset_name" ] || [ -z "$num_test_instances" ] || [ -z "$ablation_types_input" ] || [ -z "$n_consecutive_layers_input" ] || [ -z "$ablate_n_heads_per_layer" ] || [ -z "$gpu_index" ]; then
    echo "Usage: $0 <dataset_name> <num_test_instances> <ablation_types_lst> <n_consecutive_layers_lst> <ablate_n_heads_per_layer> <gpu_index> [term]"
    echo ""
    echo "Required arguments:"
    echo "  dataset_name                      - Name of the dataset (e.g., gift-eval)"
    echo "  num_test_instances                - Number of test instances"
    echo "  ablation_types_lst                - Space-separated ablation types (e.g., 'head head,mlp' for [['head'], ['head','mlp']])"
    echo "  n_consecutive_layers_lst          - Space-separated layer counts (e.g., '1 2 4 6' or '12')"
    echo "  ablate_n_heads_per_layer          - Number of heads to ablate per layer (use 'null' for all heads)"
    echo "  gpu_index                         - GPU index to use (e.g., 1 for cuda:1)"
    echo "  term                              - Term to use (i.e. short, medium, long)"
    echo "  prediction_length                 - Prediction length to use (we recommend multiple of 64 because of the evaluation setup)"
    echo ""
    exit 1
fi

data_dir=$WORK/data/$dataset_name

# Parse ablation_types_lst from space-separated input
# Input format: "head head,mlp mlp" -> creates array: (['head'] ['head', 'mlp'] ['mlp'])
IFS=' ' read -r -a ablation_types_lst <<< "$ablation_types_input"

# Parse n_consecutive_layers_to_ablate_lst from space-separated input
# Input format: "1 2 4 6" or "12" -> creates array: (1 2 4 6) or (12)
IFS=' ' read -r -a n_consecutive_layers_to_ablate_lst <<< "$n_consecutive_layers_input"

echo "data_dir: $data_dir"

timesfm_model_name=2.5-200m-pytorch

model_dirname="timesfm"
# num_layers: 20, num_heads: 16

model_name="google/timesfm-${timesfm_model_name}"
model_name_str=${model_name//\//-}
echo "model_dirname: $model_dirname"
echo "model_name: $model_name"
echo "dataset_name: $dataset_name"
echo "num_test_instances: $num_test_instances"
echo "ablation_types_lst: ${ablation_types_lst[@]}"
echo "n_consecutive_layers_to_ablate_lst: ${n_consecutive_layers_to_ablate_lst[@]}"
echo "ablate_n_heads_per_layer: $ablate_n_heads_per_layer"
echo "gpu_index: $gpu_index"
echo "term: $term"
echo "window_start_time: $window_start_time"

# for ablation_types in "${ablation_types_lst[@]}"; do
for ablation_types in "${ablation_types_lst[@]}"; do
    for layer_group_size in "${n_consecutive_layers_to_ablate_lst[@]}"; do
        # Generate ablation configuration
        generate_ablation_config 20 "$layer_group_size" "$ablation_types" "$ablate_n_heads_per_layer" "$dataset_name" "$window_start_time" "$num_test_instances" "$term"
        
        echo "ablation_types: $ablation_types"
        echo "layer_group_size: $layer_group_size"
        echo "ablations_layer_lst: $ABLATIONS_LAYER_LST"
        echo "run_name: $RUN_NAME"

        python scripts/run_ablations.py \
            ablation.ablations_types="${ablation_types}" \
            ablation.ablations_layers_lst=${ABLATIONS_LAYER_LST} \
            ablation.ablate_n_heads_per_layer=${ablate_n_heads_per_layer} \
            ablation.model_type=timesfm \
            ablation.model_name_str=${model_name_str} \
            timesfm.model_id=${model_name} \
            timesfm.context_length=512 \
            eval.prediction_length=${prediction_length} \
            eval.dataset_name=$dataset_name \
            eval.data_dir=$data_dir \
            eval.dysts.system_names=null \
            eval.dysts.num_subdirs=1 \
            eval.gift_eval.dataset_names=null \
            eval.gift_eval.max_num_datasets=null \
            eval.gift_eval.term=${term} \
            eval.gift_eval.to_univariate=false \
            eval.num_test_instances=${num_test_instances} \
            eval.parallel_sample_reduction=median \
            eval.window_style=fixed_start \
            eval.window_start_time=${window_start_time} \
            eval.batch_size=32 \
            eval.results_save_dir=$WORK/results \
            eval.metrics_save_dir=$WORK/ablations_results/${model_dirname}/${model_name_str}/${RUN_NAME} \
            eval.metrics_fname=metrics_${RUN_NAME} \
            eval.device=cuda:${gpu_index} \
            eval.rseed=123 \
        
    done
done