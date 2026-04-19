# LiveCodeBench

Handler + vendored evaluator for the [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) dataset.

## Overview

`LiveCodeBench` provides competitive-programming problems (LeetCode-style) with starter code, test cases, and reference solutions. In `PDB`, [handler.py](handler.py) implements the `LiveCodeBenchHandler` class; its `verify_unit_test` method shells out to `lcb_runner.runner.custom_evaluator`, which is installed from the vendored source under [install/](install/).

The install tree is self-contained: `handler.py` runs the evaluator inside `install/.venv/` via `python -m lcb_runner.runner.custom_evaluator`. The parent shell's environment is never touched — no `source .venv/bin/activate` or conda activate required. Each subprocess opens and closes the venv for its own lifetime.

## Install (uv)

```bash
cd dataset/livecodebench/install
uv venv --python 3.10
uv sync
```

Expect this to pull ~10 GB of deps (torch, vllm, triton, nvidia-*). `uv lock` resolves cleanly against the current hatchling `pyproject.toml` — no manual lockfile regeneration is required.

Verify the install (**must** run from `install/` — see the cwd note in Troubleshooting):

```bash
./.venv/bin/python -m lcb_runner.runner.custom_evaluator --help
```

## Layout

```
dataset/livecodebench/
├── README.md                    # this file
├── handler.py                   # LiveCodeBenchHandler — imports, no runtime env changes
└── install/                     # self-contained uv project (vendored from eval_env/)
    ├── pyproject.toml           # uv-native hatchling build
    ├── uv.lock
    ├── LICENSE
    ├── README.md                # upstream LiveCodeBench docs
    ├── ERRATA.md
    ├── assets/ data/ output/    # benchmark fixtures + scratch output
    ├── lcb_sky.yml              # SkyPilot config (not used by PDB)
    ├── .venv/                   # created by `uv venv` (git-ignored)
    └── lcb_runner/              # the Python package
        ├── runner/custom_evaluator.py
        ├── benchmarks/
        ├── evaluation/
        ├── prompts/
        └── utils/
```

Previously this tree lived at `eval_env/LiveCodeBench/` and was activated interactively via `eval_env/env.sh`. Both have been removed — `handler.py` now resolves the venv at `dataset/livecodebench/install/.venv/` automatically.

## Verification

After installing, from the repo root, confirm the handler resolves its venv with **no** venv activated:

```bash
python -c "from dataset.livecodebench.handler import LiveCodeBenchHandler; h = LiveCodeBenchHandler(); print('venv:', h.venv_python, h.venv_python.exists())"
```

## Troubleshooting

- **`RuntimeError: uv venv missing at .../install/.venv/bin/python`** — run `cd dataset/livecodebench/install && uv sync`.
- **`FileNotFoundError: lcb_runner/prompts/few_shot_examples/generation/func.json`** when importing `lcb_runner.prompts` — the upstream `code_generation.py:169` opens this file via a **cwd-relative** path, so any process that imports `lcb_runner` must have its working directory set to `install/`. `handler.py` already sets `cwd=self.install_dir` for the subprocess; only manual `python -m …` invocations need to `cd dataset/livecodebench/install` first.
- **`uv sync` runs out of disk** — a full install is ~11 GB (torch + vllm + nvidia-* + triton). Free space before retrying.
- **Torch / vllm install fails** — these are heavy deps; ensure you have a compatible CUDA toolkit, or trim the `vllm` / `torch` entries in `pyproject.toml` if you only need CPU evaluation.
- **`lcb_runner.runner` module not found** — the top-level `lcb_runner/` directory is a namespace package (no `__init__.py`). Hatchling handles this fine via `[tool.hatch.build.targets.wheel] packages = ["lcb_runner"]`, but if you see import errors inside the venv, run `uv pip install -e .` from `install/` to force an editable reinstall.
