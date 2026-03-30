# # dataset_name num_test_instances ablation_types n_consecutive_layers ablate_n_heads_per_layer gpu_idx term

# # Chronos (12 layers, 12 heads per layer)
# ./scripts/bash_scripts/ablations/run_ablations_chronos.sh gift-eval 10 "head [head,mlp]" "4 12" 3 1 short && wait && \
# ./scripts/bash_scripts/ablations/run_ablations_chronos.sh gift-eval 10 "head [head,mlp]" "4 12" 6 1 short && wait && \
# ./scripts/bash_scripts/ablations/run_ablations_chronos.sh gift-eval 10 "head [head,mlp]" "4 12" 9 1 short && wait && \
# ./scripts/bash_scripts/ablations/run_ablations_chronos.sh gift-eval 10 "head [head,mlp]" "4 12" 3 1 long && wait && \
# ./scripts/bash_scripts/ablations/run_ablations_chronos.sh gift-eval 10 "head [head,mlp]" "4 12" 6 1 long && wait && \
# ./scripts/bash_scripts/ablations/run_ablations_chronos.sh gift-eval 10 "head [head,mlp]" "4 12" 9 1 long && wait

# # # Chronos Bolt (12 layers, 12 heads per layer)
# ./scripts/bash_scripts/ablations/run_ablations_bolt.sh gift-eval 10 "head [head,mlp]" "4 12" 3 1 short && wait && \
# ./scripts/bash_scripts/ablations/run_ablations_bolt.sh gift-eval 10 "head [head,mlp]" "4 12" 6 1 short && wait && \
# ./scripts/bash_scripts/ablations/run_ablations_bolt.sh gift-eval 10 "head [head,mlp]" "4 12" 9 1 short && wait && \
# ./scripts/bash_scripts/ablations/run_ablations_bolt.sh gift-eval 10 "head [head,mlp]" "4 12" 3 1 long && wait && \
# ./scripts/bash_scripts/ablations/run_ablations_bolt.sh gift-eval 10 "head [head,mlp]" "4 12" 6 1 long && wait && \
# ./scripts/bash_scripts/ablations/run_ablations_bolt.sh gift-eval 10 "head [head,mlp]" "4 12" 9 1 long && wait

# # # # TimesFM (20 layers, 16 heads per layer)
# ./scripts/bash_scripts/ablations/run_ablations_timesfm.sh gift-eval 10 "head mlp [head,mlp]" "3 5 6 7" null 0 short && wait && \
# ./scripts/bash_scripts/ablations/run_ablations_timesfm.sh gift-eval 10 "head mlp [head,mlp]" "3 5 6 7" null 0 long && wait

# Chronos-2 (12 layers, 12 heads per layer)
./scripts/bash_scripts/ablations/run_ablations_chronos2.sh gift-eval 10 "head mlp [head,mlp]" "3" null 1 short && wait && \
./scripts/bash_scripts/ablations/run_ablations_chronos2.sh gift-eval 10 "head mlp [head,mlp]" "3" null 1 long && wait


# Moirai (12 layers, 12 heads per layer)
# ./scripts/bash_scripts/ablations/run_ablations_moirai.sh gift-eval 10 "head mlp [head,mlp]" "1 2" null 1 short
# ./scripts/bash_scripts/ablations/run_ablations_moirai.sh gift-eval 10 "head mlp [head,mlp]" "1 2" null 1 long && wait