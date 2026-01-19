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


To automatically strip notebook outputs on git push, make a file `.git/hooks/pre-commit` and write:
```
#!/bin/bash
files=$(git diff --cached --name-only --diff-filter=ACM | grep '\.ipynb$')
if [[ -n "$files" ]]; then
    echo "Cleaning notebook outputs before commit..."
    echo "$files" | xargs -I {} jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True --inplace {}
    git add $files
fi
```

Then, `chmod +x .git/hooks/pre-commit`

TODOs:
- TimesFM 2.5 doesn't work when not cuda:0 i.e. when gpu index is not 0. The predictions flatline.

