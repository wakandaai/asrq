# Recreating the asrq conda environment

The environment is Python 3.12.14 with torch 2.14.0+cu126, NeMo 3.0.0 and transformers 5.17.0. Two of its
packages are not on PyPI as used here — `asrq` itself and the humming fork — so they are installed from the
checkout after the rest.

Two lock files, exported from the working machine:

- `dev/env/environment.yml` — `conda env export --no-builds`: every conda package with its version, and a `pip:`
  section for the rest. Reproduces the environment most closely, but only on the same platform (linux-64).
- `dev/env/requirements.txt` — `pip freeze` without the local editable installs. Use it when creating the
  environment by hand, or on another platform.

## The straightforward way, same platform

```bash
git clone <this repo> asrq && cd asrq
git submodule update --init third_party/open_asr_leaderboard third_party/humming

conda env create -n asrq -f dev/env/environment.yml
conda activate asrq

pip install -e . --no-deps
pip install -e third_party/humming --no-deps
```

`--no-deps` on both because the lock file already pins everything; without it pip may resolve a different
torch.

## By hand, or on another platform

```bash
conda create -n asrq python=3.12.14
conda activate asrq

# torch first, from the CUDA 12.6 index, so nothing else pulls a different build
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126

pip install -r dev/env/requirements.txt
pip install -e . --no-deps
pip install -e third_party/humming --no-deps
```

`requirements.txt` still carries the `+cu126` local version on torch, so on a machine with a different CUDA
version, install torch from the matching index first and let the rest resolve around it.

## What the lock files leave out

- **`asrq`** — installed editable from the checkout, so the CLI imports the working tree.
- **`humming-kernels`** — the fork at `third_party/humming`, branch `hadamard-sign-seed`, which carries the
  seeded Hadamard signs. Installing the PyPI `humming-kernels` instead gives a kernel whose signs do not match
  the ones folded into the weights. Its kernels JIT-compile on first use, which needs `ninja` and a CUDA
  toolkit.
- **The optional Hadamard kernels** (`.[fast-hadamard]`, `.[hadacore]`) build from source and need
  `--no-build-isolation`; `requirements.txt` records the commit of whichever one is installed here.
- **The submodule commits**, which git tracks: `third_party/humming` at `a8296473`, `third_party/open_asr_leaderboard`
  at `cb614857`. `git submodule update --init` restores exactly those.

## Data and caches, which are not part of the environment

- **Calibration sets** are built per model, see `python -m asrq.calibration.build --help`, and land in
  `outputs/calibration/`.
- **Model and dataset caches** live in `~/.cache/huggingface`. Copying that directory saves re-downloading;
  `HF_HOME` points it elsewhere if the disk is small.
- **Rotations** in `outputs/rotation/` are checkpoints, not environment: copy them to reuse a search.

## Check the environment works

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # 2.14.0+cu126 True
pytest -m "not integration" tests                                               # about two minutes
pytest -m integration tests/test_parakeet_ctc.py                                # needs a GPU and the models
```

The integration tests download the models on first run.

## Refreshing the lock files

From the machine whose environment is the reference:

```bash
conda env export --no-builds | grep -v "^prefix:" > dev/env/environment.yml
pip freeze | grep -vE "^-e |^asrq|^humming-kernels|file:///" > dev/env/requirements.txt
```
