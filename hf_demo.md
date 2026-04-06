# HF Demo

This adds a Gradio-based Hugging Face demo scaffold at [hf_demo/app.py](/stor/home/ab75733/tsfm-lens/hf_demo/app.py). The app reuses the existing `tsfm_lens` pipeline wrappers and the local `GiftEvalDataset` loader rather than re-implementing the ablation logic from the notebooks.

## What The Demo Supports

- Model selection across the TSFMs already used in these ablation notebooks:
  `Chronos`, `Chronos-Bolt`, `Chronos-2`, `TimesFM 2.5`, `Toto`, `Moirai 1.1 R Base`
- Gift-Eval controls:
  dataset, term, sample index, selected dimensions, context length
- Ablation controls:
  per-layer head specification and MLP layer specification
- Output:
  one plot with context, ground truth, baseline forecast, and ablated forecast
  plus a short run summary including MAE before and after ablation

## Files

- [app.py](/stor/home/ab75733/tsfm-lens/hf_demo/app.py): Gradio app entrypoint
- [requirements.txt](/stor/home/ab75733/tsfm-lens/hf_demo/requirements.txt): extra UI dependencies for the demo layer
- [runtime.txt](/stor/home/ab75733/tsfm-lens/hf_demo/runtime.txt): Python version hint for Spaces

## Local Run

From the repo root:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install toto-ts
uv pip install -e .[tsfms]
uv pip install gluonts==0.15.1
uv pip install -r hf_demo/requirements.txt
cd external/timesfm && uv pip install -e . && cd ../..
python hf_demo/app.py
```

The app now uses `$WORK/data/gift-eval` explicitly. Make sure `WORK` is set before launch:

```bash
export WORK=/path/to/workdir
```

## Hugging Face Spaces Deployment

The cleanest path is to deploy from the full repo, not from `hf_demo/` alone, because the app imports:

- `tsfm_lens/...`
- local `assets/`
- the local TimesFM submodule setup already used by this repo

Recommended setup:

1. Push the full repo to the Space.
2. Set the Space app file to `hf_demo/app.py`.
3. Copy or merge [requirements.txt](/stor/home/ab75733/tsfm-lens/hf_demo/requirements.txt) into the Space root `requirements.txt`.
4. Make sure the repo package itself is installed in the Space build.
5. Mount or copy the Gift-Eval dataset into `$WORK/data/gift-eval` and make sure `WORK` is set in the Space environment.

If you want the Space to be truly standalone, you will need to vendor or publish the repo package and any local assets it depends on.

## UI Conventions

- Dimensions accept comma and range syntax:
  `0`
  `0,1,2`
  `0-3`
- MLP layers use the same syntax.
- Head ablations use one `layer: heads` rule per line:

```text
3: 0,1,2
7: all
9: 4-7
```

## Notes

- `TimesFM 2.5` is kept on `cuda:0` by default when CUDA is available, matching the repo note that other device placements have been flaky.
- The app caches loaded pipelines in-process so repeated runs do not reload the weights each time.
- The current scaffold compares baseline and ablated point forecasts. If you want notebook-style quantile or sample visualizations next, extend `run_forecast(...)` in [app.py](/stor/home/ab75733/tsfm-lens/hf_demo/app.py).
