#!/bin/bash

ulimit -n 99999

# Parse command-line arguments with defaults
gpu_index=${1:-0}
term=${2:-short}
max_datasets=${3:-null}
model_type=${4:-null}
layers_to_ablate=${5:-null}
ablated_components=${6:-null}
ablate_n_heads_per_layer=${7:-null}
head_selection_strategy=${8:-srank}
rseed=${9:-123}

# Set data directory
data_dir=${WORK}/data/gift-eval

echo "Running evaluation with:"
echo "  GPU: cuda:${gpu_index}"
echo "  Term: ${term}"
echo "  Max datasets: ${max_datasets}"
echo "  Model type: ${model_type}"
echo "  Data directory: ${data_dir}"
echo "  Layers to ablate: ${layers_to_ablate}"
echo "  Ablated components: ${ablated_components}"
echo "  Ablate n heads per layer: ${ablate_n_heads_per_layer}"
echo "  Head selection strategy: ${head_selection_strategy}"
echo "  Rseed: ${rseed}"
echo "--------------------------------"

if [ "$model_type" = "timesfm" ]; then
    model_name=google/timesfm-2.5-200m-pytorch
elif [ "$model_type" = "chronos_bolt" ]; then
    model_name=amazon/chronos-bolt-base
elif [ "$model_type" = "chronos" ]; then
    model_name=amazon/chronos-t5-base
elif [ "$model_type" = "toto" ]; then
    model_name=Datadog/Toto-Open-Base-1.0
else
    echo "Invalid model type: ${model_type}"
    exit 1
fi


model_name_str=${model_name//\//-}
echo "  Model name: ${model_name}"
echo "  Model name string: ${model_name_str}"

chronos_bolt_args=(
    chronos_bolt.model_id=${model_name}
    chronos_bolt.limit_prediction_length=false
)

chronos_args=(
    chronos.model_id=${model_name}
    chronos.limit_prediction_length=false
    chronos.num_samples=10
    chronos.deterministic=false
)

timesfm_args=(
    timesfm.model_id=${model_name}
)

toto_args=(
    toto.model_id=${model_name}
    toto.samples_per_batch=20
    toto.use_kv_cache=true
    toto.pad_short_series=false
)

# Append samples_per_batch to model_name_str for toto models
if [ "$model_type" = "toto" ]; then
    # Extract samples_per_batch from toto_args
    for arg in "${toto_args[@]}"; do
        if [[ $arg == toto.samples_per_batch=* ]]; then
            samples_per_batch="${arg#*=}"
            model_name_str="${model_name_str}_samples-${samples_per_batch}"
            echo "  Updated model name string: ${model_name_str}"
            break
        fi
    done
fi


# Dynamically select model args based on model type
declare -n model_args="${model_type}_args"

echo "model args: ${model_args[@]}"
echo "--------------------------------"

# '["m4_weekly", "bizitobs_l2c/H"]'

# Base configuration
base_args=(
    eval.dataset_name=gift-eval
    eval.data_dir=${data_dir}
    eval.gift_eval.dataset_names=null
    eval.gift_eval.max_num_datasets=${max_datasets}
    eval.gift_eval.term=${term}
    eval.gift_eval.to_univariate=false
    eval.device=cuda:${gpu_index}
    eval.rseed=${rseed}
    eval.results_save_dir=${HOME}/tsfm-lens/results
    ablation.model_name_str=${model_name_str}
    ablation.model_type=${model_type}
)

# Add ablation configuration if requested
if [ "$layers_to_ablate" != "null" ]; then
    echo "Setting up ablations"
    ablation_args=(
        ablation.ablations_types="${ablated_components}"
        ablation.ablations_layers_lst=${layers_to_ablate}
        ablation.ablate_n_heads_per_layer=${ablate_n_heads_per_layer}
        ablation.head_selection_strategy=${head_selection_strategy}
    )
    echo "Ablations configured: ${ablation_args[@]}"
else
    echo "No ablations configured"
    ablation_args=(
        ablation.ablations_layers_lst=null
    )
fi

# Run the evaluation script
python scripts/eval_ablations_gift_combined.py \
    "${base_args[@]}" \
    "${ablation_args[@]}" \
    "${model_args[@]}"

echo "Evaluation complete!"

