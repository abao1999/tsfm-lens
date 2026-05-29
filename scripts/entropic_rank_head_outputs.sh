#!/bin/bash
#
# entropic_rank_head_outputs.sh — Drive scripts/entropic_rank_head_outputs.py
# for a single TSFM. Selects the model via the top-level `models_to_run` Hydra
# override; all other knobs are exposed as edit-me variables below.
#
# Usage: ./scripts/entropic_rank_head_outputs.sh [-h|--help] [-n|--dry-run] [model_type] [extra hydra overrides...]
#

set -e
ulimit -n 99999

# =============================================================================
# CONFIGURATION (edit me)
# =============================================================================
MODEL=${MODEL:-"chronos"}              # chronos | chronos_bolt | chronos2 | timesfm | toto | moirai
GPU=${GPU:-0}
torch_dtype="bfloat16"
rseed=1234

# Data selection (entropic_rank.*)
data_dir="/ssd2/data/base40"
output_dir="/ssd2/entropic_rank_results"
system_names=null                 # e.g. "[lorenz,rossler]" or null for all subdirs
num_subdirs=null
num_samples_per_subdir=null

# Window / sampling (entropic_rank.*)
context_length=512
prediction_length=512
num_samples=20
deterministic=false
epsilon=1e-8

# Caps on iteration (entropic_rank.*)
max_series_per_system=null
max_windows_per_series=null
max_windows_total=null

# Eval-side window selection (eval.*)
num_test_instances=4
window_style=sampled              # sampled | rolling | single | fixed_start
window_stride=1
window_start_time=null

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
chronos_num_samples=20
toto_num_samples=10
moirai_num_samples=100
declare -A model_args=(
    [chronos]="chronos.model_id=${model_id[chronos]} chronos.deterministic=${deterministic} chronos.num_samples=${chronos_num_samples} chronos.limit_prediction_length=false chronos.context_length=${context_length}"
    [chronos_bolt]="chronos_bolt.model_id=${model_id[chronos_bolt]} chronos_bolt.limit_prediction_length=false chronos_bolt.context_length=${context_length}"
    [chronos2]="chronos2.model_id=${model_id[chronos2]} chronos2.context_length=${context_length}"
    [timesfm]="timesfm.model_id=${model_id[timesfm]} timesfm.context_length=${context_length}"
    [toto]="toto.model_id=${model_id[toto]} toto.num_samples=${toto_num_samples} toto.samples_per_batch=${toto_num_samples} toto.context_length=${context_length} toto.use_kv_cache=true"
    [moirai]="moirai.model_id=${model_id[moirai]} moirai.num_samples=${moirai_num_samples} moirai.patch_size=32"
)

# =============================================================================
# ARG PARSING
# =============================================================================
dry_run=0
print_help() {
    cat <<EOF
Usage: $(basename "$0") [-h|--help] [-n|--dry-run] [MODEL] [extra hydra overrides...]

Arguments:
  MODEL                one of: ${!model_id[*]}
                       (default: ${MODEL})
  extra overrides      any trailing args are forwarded to Hydra verbatim,
                       e.g. entropic_rank.max_windows_total=4 eval.rseed=7

Options:
  -h, --help           Show this help and exit
  -n, --dry-run        Print the python command; do not launch
EOF
}
extra_args=()
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)    print_help; exit 0 ;;
        -n|--dry-run) dry_run=1; shift ;;
        --)           shift; extra_args+=("$@"); break ;;
        -*)           echo "Unknown option: $1" >&2; print_help; exit 2 ;;
        *)
            if [[ "$1" == *"="* ]]; then
                extra_args+=("$1"); shift
            elif [ -n "${model_id[$1]+x}" ]; then
                model_type="$1"; shift
            else
                echo "Error: unknown model_type '$1'. Valid: ${!model_id[*]}" >&2
                exit 2
            fi
            ;;
    esac
done

if [ -z "${model_id[$MODEL]+x}" ]; then
    echo "Error: unknown MODEL '${MODEL}'. Valid: ${!model_id[*]}" >&2
    exit 2
fi

# =============================================================================
# DERIVE
# =============================================================================
read -ra args_for_model <<< "${model_args[$MODEL]}"

echo "Model: ${MODEL} (${model_id[$MODEL]}) | GPU: cuda:${GPU}"
echo "Output: ${output_dir}/${MODEL}"
[ "$dry_run" = 1 ] && echo "(dry-run — no python will be launched)"

# =============================================================================
# BASE ARGS
# =============================================================================
base_args=(
    "models_to_run=[${MODEL}]"
    entropic_rank.output_dir="${output_dir}"
    entropic_rank.data_dir="${data_dir}"
    entropic_rank.system_names="${system_names}"
    entropic_rank.num_subdirs="${num_subdirs}"
    entropic_rank.num_samples_per_subdir="${num_samples_per_subdir}"
    entropic_rank.context_length="${context_length}"
    entropic_rank.prediction_length="${prediction_length}"
    entropic_rank.num_samples="${num_samples}"
    entropic_rank.deterministic="${deterministic}"
    entropic_rank.epsilon="${epsilon}"
    entropic_rank.max_series_per_system="${max_series_per_system}"
    entropic_rank.max_windows_per_series="${max_windows_per_series}"
    entropic_rank.max_windows_total="${max_windows_total}"
    eval.device="cuda:${GPU}"
    eval.torch_dtype="${torch_dtype}"
    eval.rseed="${rseed}"
    eval.num_test_instances="${num_test_instances}"
    eval.window_style="${window_style}"
    eval.window_stride="${window_stride}"
    eval.window_start_time="${window_start_time}"
)

cmd=(python scripts/entropic_rank_head_outputs.py
    "${base_args[@]}"
    "${args_for_model[@]}"
    "${extra_args[@]}"
)

if [ "$dry_run" = 1 ]; then
    printf '  %s\n' "${cmd[*]}"
else
    "${cmd[@]}"
fi

echo "Done."
