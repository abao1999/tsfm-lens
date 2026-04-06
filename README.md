# TSFM Lens

### Setup

Dependencies can be installed directly with pip from the pyproject.toml, but for more fine grained control, install [`uv`](https://docs.astral.sh/uv/getting-started/installation/). NOTE that there is a dependency conflict with toto-ts, which is why we don't just do `uv sync`. Instead, we recommend:

Step 1:
```bash
uv venv --python 3.11
source .venv/bin/activate
```

Step 2:
`uv pip install toto-ts`

Step 3:
`uv pip install -e .[tsfms]`

Step 4:
`uv pip install gluonts==0.15.1`

Crucially, we benchmark on `Gift-Eval`, which requires `pandas==2.0.0` and also `gluonts==0.15.1`. The latter requirement gets overridden when installing Moirai (`uni2ts`) from the tsfms group. For this reason, Step 4 enables running the ablation evaluations on `Gift-Eval`. The installation of Moirai also downgrades other dependencies, notably downgrading CUDA version 12.6 to version 12.1, but this does not present an issue for us.

To fetch and initialize the timesfm submodule:
```
git config submodule.recurse true
git submodule update --init --recursive
git submodule status
```

Lastly, install the submodule in editable mode, with:
```
cd external/timesfm
uv pip install -e .
```


To automatically strip notebook outputs before commit, install pre-commit once:
```
uv tool install pre-commit
pre-commit install
```
This uses `.pre-commit-config.yaml` (nbstripout) to clear `*.ipynb` outputs on commit.

### TODOs:
+ TimesFM 2.5 doesn't work when not cuda:0 i.e. when gpu index is not 0. The predictions flatline.
+ Fix the bug with TimesFM 2.5 inference that appeared after we installed deps for Moirai
+ Make the circuitlens `add_ablation_hooks_explicit` function signature more intuitive
+ For QOL, add a field to each model's CircuitLens class to store the model width (embedding dimension)