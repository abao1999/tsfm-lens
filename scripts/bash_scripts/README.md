## Documentation for Bash Scripts

The `ablations` directory contains bash scripts and bash script utils for running the per-context-window ablation evaluations of the TSFMs in our study. The corresponding python script is one level up, in `scripts/run_ablations.py`. This implementation is legacy, but still has useful functionality not implemented in our current `scripts/run_ablations_gift-eval.py`. Notably, we compute metrics for each context window, and we also compute the structural similarity metrics e.g. Spearman distance between the model predictions under ablations and the original model predictions.

### TODOs
+ Tidy up the scripts and unify them across TSFMs
