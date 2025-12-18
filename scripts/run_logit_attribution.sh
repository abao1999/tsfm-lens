#!/bin/bash

ulimit -n 99999

dataset_name=base40
data_dir=$WORK/data/$dataset_name
echo "data_dir: $data_dir"

chronos_model_size=base

use_deterministic=false

model_dirname="chronos"
if [ "$use_deterministic" = false ]; then
    model_dirname="chronos_nondeterministic"
fi

model_name="amazon/chronos-t5-${chronos_model_size}"
echo "model_dirname: $model_dirname"
echo "model_name: $model_name"

python scripts/logit_attribution.py \
    chronos.model_id=${model_name} \
    chronos.deterministic=$use_deterministic \
    chronos.num_samples=5 \
    chronos.limit_prediction_length=false \
    chronos.context_length=512 \
    eval.prediction_length=512 \
    eval.data_dir=$data_dir \
    eval.dysts.system_names=null \
    eval.dysts.num_subdirs=null \
    eval.num_test_instances=1 \
    eval.parallel_sample_reduction=median \
    eval.window_style=sampled \
    eval.batch_size=4 \
    eval.metrics_save_dir=$WORK/logit_attribution_results/${model_dirname}/${model_name} \
    eval.metrics_fname=metrics \
    eval.device=cuda:0 \
    eval.rseed=42 \
    "$@"