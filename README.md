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

Crucially, we benchmark on `Gift-Eval`, which requires `pandas==2.0.0`.

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

