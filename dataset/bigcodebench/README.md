# BigCodeBench

Handler + vendored evaluator for the [BigCodeBench](https://github.com/bigcode-project/bigcodebench) dataset.

## Overview

`BigCodeBench` provides coding tasks with structured prompts, canonical solutions, and rich test suites. In `PDB`, [handler.py](handler.py) implements the `BigCodeBenchHandler` class; its `verify_unit_test` method shells out to `bigcodebench.evaluate`, which is installed from the vendored copy under [install/](install/).

The install tree is self-contained: `handler.py` runs the evaluator inside `install/.venv/` via `python -m bigcodebench.evaluate`. The parent shell's environment is never touched — no `conda activate` or `source .venv/bin/activate` required. Each subprocess opens and closes the venv for its own lifetime.

## Install (uv)

```bash
cd dataset/bigcodebench/install
uv venv --python 3.10
uv sync --extra eval
```

`--extra eval` installs the sandbox dependencies (Django, Flask, TensorFlow, scikit-learn, pandas, OpenCV, etc.) that BigCodeBench's test harness needs to execute candidate solutions. Version specs in `[project.optional-dependencies] eval` are **ranges, not strict pins** — they were relaxed from the upstream `Requirements/requirements-eval.txt` to a known-working set (the legacy `Evaluate` conda env on the original dev host).

The core dependencies already include `transformers`, `datasets`, `gradio-client`, `e2b`, `rich`, etc. — they're needed at module-load time by `bigcodebench.evaluate` and friends, so `uv sync` (without `--extra eval`) still pulls a non-trivial stack (torch, tokenizers, …). If you genuinely don't need to run any tests, there's no reason to skip `--extra eval`.

Verify the install:

```bash
./.venv/bin/python -m bigcodebench.evaluate --help
```

## Layout

```
dataset/bigcodebench/
├── README.md                    # this file
├── handler.py                   # BigCodeBenchHandler — imports, no runtime env changes
└── install/                     # self-contained uv project (vendored bigcodebench v0.2.5)
    ├── pyproject.toml           # uv-native; mirrors upstream entry points
    ├── Requirements/            # original pinned requirement files, kept for reference
    ├── LICENSE
    ├── .venv/                   # created by `uv venv` (git-ignored)
    └── bigcodebench/            # the Python package (copied from site-packages v0.2.5)
```

The `bigcodebench/` package was copied from the authoritative `Evaluate` conda env's site-packages, which already contained all local patches (e.g. the `int(os.getenv("BIGCODEBENCH_TIMEOUT_PER_TASK", ...))` cast in `eval/__init__.py` and `gen/util/__init__.py`, XDG_CONFIG_HOME isolation, and the `hf-inference` backend).

## Verification

After installing, from the repo root, confirm the handler resolves its venv with **no** conda env activated:

```bash
python -c "from dataset.bigcodebench.handler import BigCodeBenchHandler; h = BigCodeBenchHandler(); print('venv:', h.venv_python, h.venv_python.exists())"
```

## Troubleshooting

- **`RuntimeError: uv venv missing at .../install/.venv/bin/python`** — run `cd dataset/bigcodebench/install && uv sync --extra eval`.
- **`ModuleNotFoundError` inside sandboxed tests** — you skipped `--extra eval`. Rerun `uv sync --extra eval`.
- **`uv sync` fails with a protobuf version conflict** — `tensorflow==2.11` overstates a `protobuf<3.20` upper bound in its pip metadata that doesn't reflect what actually works at runtime. `pyproject.toml` already has `[tool.uv] override-dependencies = ["protobuf>=5.0,<6.0"]` to unblock this. If you bump tensorflow or drop the override, the resolver will fail again.
- **TensorFlow prints CUDA `libcudart.so.11.0` / `libnvinfer.so.7` load warnings** — cosmetic; they mean no GPU is attached. TF falls back to CPU and the evaluator works fine.
- **Timeouts** — bumped via `BIGCODEBENCH_TIMEOUT_PER_TASK` (set by `handler.py` per call) or the handler's `timeout=` kwarg.
