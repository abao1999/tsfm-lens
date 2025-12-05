#!/bin/bash
gpu_index=2
model_type="chronos_bolt"
rseeds=(42)
ablated_components="[head]"
head_selection_strategy="srank_reverse"

ablated_layers="[2]"
for rseed in "${rseeds[@]}"; do
    echo "Running evaluation with rseed: ${rseed}"
    echo "ablated_layers: ${ablated_layers}"
    echo "ablated_components: ${ablated_components}"
    echo "head_selection_strategy: ${head_selection_strategy}"
    echo "gpu_index: ${gpu_index}"
    echo "--------------------------------"
    ./scripts/bash_scripts/gift-eval/make_ablations_run.sh ${gpu_index} all null ${model_type} "${ablated_layers}" "${ablated_components}" 10 ${head_selection_strategy} ${rseed} && wait && \
    ./scripts/bash_scripts/gift-eval/make_ablations_run.sh ${gpu_index} all null ${model_type} "${ablated_layers}" "${ablated_components}" 8 ${head_selection_strategy} ${rseed} && wait && \
    ./scripts/bash_scripts/gift-eval/make_ablations_run.sh ${gpu_index} all null ${model_type} "${ablated_layers}" "${ablated_components}" 6 ${head_selection_strategy} ${rseed} && wait && \
    ./scripts/bash_scripts/gift-eval/make_ablations_run.sh ${gpu_index} all null ${model_type} "${ablated_layers}" "${ablated_components}" 4 ${head_selection_strategy} ${rseed} && wait && \
    ./scripts/bash_scripts/gift-eval/make_ablations_run.sh ${gpu_index} all null ${model_type} "${ablated_layers}" "${ablated_components}" 2 ${head_selection_strategy} ${rseed} && wait && \
    ./scripts/bash_scripts/gift-eval/make_ablations_run.sh ${gpu_index} all null ${model_type} "${ablated_layers}" "${ablated_components}" 1 ${head_selection_strategy} ${rseed} && wait

done

ablated_layers="[1]"
for rseed in "${rseeds[@]}"; do
    echo "Running evaluation with rseed: ${rseed}"
    echo "ablated_layers: ${ablated_layers}"
    echo "ablated_components: ${ablated_components}"
    echo "head_selection_strategy: ${head_selection_strategy}"
    echo "gpu_index: ${gpu_index}"
    echo "--------------------------------"
    ./scripts/bash_scripts/gift-eval/make_ablations_run.sh ${gpu_index} all null ${model_type} "${ablated_layers}" "${ablated_components}" 10 ${head_selection_strategy} ${rseed} && wait && \
    ./scripts/bash_scripts/gift-eval/make_ablations_run.sh ${gpu_index} all null ${model_type} "${ablated_layers}" "${ablated_components}" 8 ${head_selection_strategy} ${rseed} && wait && \
    ./scripts/bash_scripts/gift-eval/make_ablations_run.sh ${gpu_index} all null ${model_type} "${ablated_layers}" "${ablated_components}" 6 ${head_selection_strategy} ${rseed} && wait && \
    ./scripts/bash_scripts/gift-eval/make_ablations_run.sh ${gpu_index} all null ${model_type} "${ablated_layers}" "${ablated_components}" 4 ${head_selection_strategy} ${rseed} && wait && \
    ./scripts/bash_scripts/gift-eval/make_ablations_run.sh ${gpu_index} all null ${model_type} "${ablated_layers}" "${ablated_components}" 2 ${head_selection_strategy} ${rseed} && wait && \
    ./scripts/bash_scripts/gift-eval/make_ablations_run.sh ${gpu_index} all null ${model_type} "${ablated_layers}" "${ablated_components}" 1 ${head_selection_strategy} ${rseed} && wait

done