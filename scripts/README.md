## Documentation for Scripts

This `scripts` directory contains the scripts for our study. All scripts import their main functionality from the main library files within `tsfm_lens` one level up. We have implemented:

Ablations Evaluation
+ `run_ablations_gift-eval.py` and the corresponding bash script `run_ablations_gift-eval.sh` implement the large-scale evaluation of the TSFMs, under customizable ablations of components, on the entire Gift-Eval test split, computing the full range of metrics used by Gift-Eval, as provided by GluonTS, from `gluonts.ev.metrics`. We are working on extending the functionality to more TSFMs, in addition to quality-of-life upgrades.

+ `run_ablations.py` and the corresponding bash scripts (TODO: organize better) within `bash_scripts/ablations` implement the evaluation of the TSFMs, under customizable ablations of components, on Gift-Eval test split or on the `dysts` synthetic data. Notably, the metrics computed for these evaluations include metrics for structural similarity between the model predictions under ablations and the original model predictions. And crucially, all metrics computed by this script are per-context-window, i.e. there is no aggregation within a dataset. In this way, we can get a more fine-grained view of how the ablations affect the structure of the forecasts, relative to the original forecast structure.

+ `logit_attribution.py` and the corresponding bash script `logit_attribution.sh` implement the scaled-up direct logit attribution measurements on the residual stream of the Chronos decoder. By scaling up, we refer to the collection of metrics from measurements on the residual stream of hundreds of forecasting tasks. We measure e.g. the average entropy of the residual stream's logit maps, the number of distinct peaks that correspond to lines of thought, etc.

+ `entropic_rank_head_outputs.py` implements the scaled-up measurement of the pairwise head outputs activation vector similarity, across hundreds of forecasting tasks. This functionality is implemented for the various TSFMs in our study, and in particular, we have it for both the cross-attention and the self-attention heads of Chronos (an encoder-decoder model).

### TODOs
+ Complete the documentation for the other scripts, which are still works in progress.
