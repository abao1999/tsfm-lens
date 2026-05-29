#!/bin/bash
#
# run_ablations.sh — Drive scripts/run_ablations.py across all 6 TSFMs.
# One python invocation per (n_heads, rseed); ALL layers are batched inside
# the call via `ablations_layers_lst=[[0],[1],...,[L-1]]`.
#
# Usage: ./scripts/run_ablations.sh [-h|--help] [-n|--dry-run] [model_type]
#

set -e
ulimit -n 99999

# =============================================================================
# CONFIGURATION (edit me)
# =============================================================================
model_type="chronos2"          # chronos | chronos_bolt | chronos2 | timesfm | toto | moirai
gpu_index=2
dataset_name="gift-eval"
term="all"
data_dir="${STOR}/data/${dataset_name}"
num_test_instances=10
prediction_length=512
window_start_time=2512 # NOTE: not used for GIFT-Eval

# Ablation grid (one python call per entry of head_counts × rseeds)
rseeds=(99)
ablated_components="[head]"        # alternatives: "[head,mlp]" "[mlp]"
head_counts=(1 null)               # `null` = ablate all heads in each layer
# Full sweep example:
#   head_counts=($(seq 1 11) null)

# =============================================================================
# PER-MODEL TABLE
# =============================================================================
declare -A model_id=(
    [chronos]="amazon/chronos-t5-base"
    [chronos_bolt]="amazon/chronos-bolt-base"
    [chronos2]="amazon/chronos-2"
    [timesfm]="google/timesfm-2.5-200m-pytorch"
    [toto]="Datadog/Toto-Open-Base-1.0"
    [moirai]="Salesforce/moirai-1.1-R-base"
)
declare -A model_num_layers=(
    [chronos]=12 [chronos_bolt]=12 [chronos2]=12
    [timesfm]=20 [toto]=12        [moirai]=12
)
declare -A model_batch_size=(
    [chronos]=32 [chronos_bolt]=32 [timesfm]=32
    [chronos2]=16 [toto]=16        [moirai]=16
)
chronos_num_samples=10
toto_num_samples=10
moirai_num_samples=100
declare -A model_args=(
    [chronos]="chronos.model_id=${model_id[chronos]} chronos.deterministic=false chronos.num_samples=${chronos_num_samples} chronos.limit_prediction_length=false chronos.context_length=512"
    [chronos_bolt]="chronos_bolt.model_id=${model_id[chronos_bolt]} chronos_bolt.limit_prediction_length=false chronos_bolt.context_length=512"
    [chronos2]="chronos2.model_id=${model_id[chronos2]} chronos2.context_length=8192"
    [timesfm]="timesfm.model_id=${model_id[timesfm]} timesfm.context_length=512"
    [toto]="toto.model_id=${model_id[toto]} toto.num_samples=${toto_num_samples} toto.samples_per_batch=${toto_num_samples} toto.context_length=512 toto.use_kv_cache=true"
    [moirai]="moirai.model_id=${model_id[moirai]} moirai.num_samples=${moirai_num_samples} moirai.patch_size=32"
)

# =============================================================================
# ARG PARSING
# =============================================================================
dry_run=0
print_help() {
    cat <<EOF
Usage: $(basename "$0") [-h|--help] [-n|--dry-run] [model_type]

Arguments:
  model_type           one of: ${!model_id[*]}
                       (default: ${model_type})

Options:
  -h, --help           Show this help and exit
  -n, --dry-run        Print the python command for each grid iteration; do not launch

Current defaults (edit the CONFIGURATION block to change):
  gpu_index            ${gpu_index}
  dataset_name         ${dataset_name}
  term                 ${term}
  data_dir             ${data_dir}
  num_test_instances   ${num_test_instances}
  prediction_length    ${prediction_length}
  window_start_time    ${window_start_time}
  rseeds               ${rseeds[*]}
  ablated_components   ${ablated_components}
  head_counts          ${head_counts[*]}
EOF
}
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)    print_help; exit 0 ;;
        -n|--dry-run) dry_run=1; shift ;;
        -*)           echo "Unknown option: $1" >&2; print_help; exit 2 ;;
        *)            model_type="$1"; shift ;;
    esac
done

if [ -z "${model_id[$model_type]+x}" ]; then
    echo "Error: unknown model_type '${model_type}'. Valid: ${!model_id[*]}" >&2
    exit 2
fi

# =============================================================================
# DERIVE
# =============================================================================
model_name="${model_id[$model_type]}"
model_name_str="${model_name//\//-}"
num_layers="${model_num_layers[$model_type]}"
batch_size="${model_batch_size[$model_type]}"
read -ra args_for_model <<< "${model_args[$model_type]}"

# layer_spec = "[[0],[1],...,[L-1]]" — all single-layer ablation sets in ONE call
layer_spec="["
for ((i=0; i<num_layers; i++)); do
    layer_spec+="[${i}]"
    [ $i -lt $((num_layers-1)) ] && layer_spec+=","
done
layer_spec+="]"

ablated_components_str=$(echo "${ablated_components}" | tr -d "[]" | tr "," "-")
total=$(( ${#rseeds[@]} * ${#head_counts[@]} ))

echo "Model: ${model_type} (${model_name}) | GPU: cuda:${gpu_index} | ${dataset_name}/${term}"
echo "Grid: ${num_layers} layers (batched) × ${#head_counts[@]} head_counts × ${#rseeds[@]} seed(s) = ${total} run(s)"
[ "$dry_run" = 1 ] && echo "(dry-run — no python will be launched)"

# =============================================================================
# BASE ARGS
# =============================================================================
base_args=(
    eval.dataset_name="${dataset_name}"
    eval.data_dir="${data_dir}"
    eval.dysts.system_names=null
    eval.dysts.num_subdirs=1
    eval.gift_eval.dataset_names=null
    eval.gift_eval.max_num_datasets=null
    eval.gift_eval.term="${term}"
    eval.gift_eval.to_univariate=false
    eval.num_test_instances="${num_test_instances}"
    eval.parallel_sample_reduction=median
    eval.window_style=fixed_start
    eval.window_start_time="${window_start_time}"
    eval.batch_size="${batch_size}"
    eval.prediction_length="${prediction_length}"
    eval.results_save_dir="${HOME}/tsfm-lens/results"
    eval.device="cuda:${gpu_index}"
    ablation.model_name_str="${model_name_str}"
    ablation.model_type="${model_type}"
)

# =============================================================================
# RUN GRID
# =============================================================================
i=0
for rseed in "${rseeds[@]}"; do
    for n in "${head_counts[@]}"; do
        i=$((i+1))
        nheads_str="$n"; [ "$n" = "null" ] && nheads_str="all"
        run_name="${term}_${dataset_name}_nwindows-${num_test_instances}_za-${ablated_components_str}_nheads-${nheads_str}"
        metrics_save_dir="${STOR}/ablations_results/${model_type}/${model_name_str}/${run_name}"

        echo "[${i}/${total}] heads=${n}, seed=${rseed}"

        cmd=(python scripts/run_ablations.py
            "${base_args[@]}"
            "${args_for_model[@]}"
            eval.rseed="${rseed}"
            eval.metrics_save_dir="${metrics_save_dir}"
            eval.metrics_fname="metrics_${run_name}"
            ablation.ablations_types="${ablated_components}"
            ablation.ablations_layers_lst="${layer_spec}"
            ablation.ablate_n_heads_per_layer="${n}"
        )

        if [ "$dry_run" = 1 ]; then
            printf '  %s\n' "${cmd[*]}"
        else
            "${cmd[@]}"
        fi
    done
done

echo "Done."
