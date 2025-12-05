#!/bin/bash

# Shared utility functions for ablation scripts

# Function to generate layer ablation list and run name
generate_ablation_config() {
    local num_layers=$1
    local layer_group_size=$2
    local ablation_types=$3
    local ablate_n_heads_per_layer=$4
    local dataset_name=$5
    local window_start_time=$6
    local num_test_instances=${7:-""}
    local term=${8:-""}               # Optional parameter for gift-eval
    
    # Generate layer ablation list
    local ablations_layer_lst="["
    for ((i=0; i<=num_layers-layer_group_size; i++)); do
        ablations_layer_lst+="["
        for ((j=0; j<layer_group_size; j++)); do
            ablations_layer_lst+="$((i+j))"
            [ $j -lt $((layer_group_size-1)) ] && ablations_layer_lst+=","
        done
        ablations_layer_lst+="]"
        [ $i -lt $((num_layers-layer_group_size)) ] && ablations_layer_lst+=","
    done
    ablations_layer_lst+="]"
    
    # local ablation_types_str=$(echo "${ablation_types[@]}" | tr -d "[]," | tr " " "-")
    local ablation_types_str=$(echo "${ablation_types[@]}" | tr -d "[]" | tr ", " "-")
    local ablate_n_heads_per_layer_str
    if [ "$ablate_n_heads_per_layer" = null ]; then
        ablate_n_heads_per_layer_str="all"
    else
        ablate_n_heads_per_layer_str="${ablate_n_heads_per_layer}"
    fi
    local run_name
    
    local test_instances=${num_test_instances:-1}
    run_name="${dataset_name}_nwindows-${test_instances}_za-${ablation_types_str}_layergroup-${layer_group_size}_nheads-${ablate_n_heads_per_layer_str}"
    
    if [ "$dataset_name" != "gift-eval" ]; then
        run_name="start-${window_start_time}_${run_name}"
    else
        run_name="${term}_${run_name}"
    fi
    
    # Return values via global variables (bash limitation)
    ABLATIONS_LAYER_LST="$ablations_layer_lst"
    RUN_NAME="$run_name"
}
