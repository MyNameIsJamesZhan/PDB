# Adding a New Dataset to PDB

## Overview

Each dataset in PDB is represented by a **handler class** that implements five
dataset-specific operations. This guide walks you through adding support for a new dataset.

## Architecture

```
dataset/
    __init__.py          # Registry — maps dataset name strings to handler instances
    base.py              # DatasetHandler ABC — defines the interface all handlers share
    bigcodebench/
        README.md        # per-dataset install + usage
        handler.py       # BigCodeBenchHandler
        install/         # self-contained uv project with vendored evaluator
            pyproject.toml
            <package>/
            .venv/       # created by `uv venv` (git-ignored)
    livecodebench/
        ...
    README.md            # this file
```

Callers throughout the codebase use `get_handler(dataset_name)` to obtain a handler, then
call its methods without needing to know which dataset is active. Each handler runs its
evaluator inside `dataset/<name>/install/.venv/` via `python -m <module>` — the parent
shell's environment is never touched, and there's nothing to activate or deactivate.

## Steps to Add a New Dataset

### 1. Create the dataset directory

```bash
mkdir -p dataset/<your_dataset_name>
touch dataset/<your_dataset_name>/__init__.py
```

### 2. Implement the handler class

Create `dataset/<your_dataset_name>/handler.py`:

```python
from dataset.base import DatasetHandler


class YourDatasetHandler(DatasetHandler):

    def preprocess(self, raw_data):
        """
        Transform raw data into PDB's standardized format.

        Must return a list of dicts, each with at least:
            "task_id": str
            "gt_solution": str
            "task_prompt": str
        Optionally:
            "test": str (unit test code)
        """
        processed_data = []
        for example in raw_data:
            processed_data.append({
                "task_id": example["id"],
                "gt_solution": example["solution"],
                "task_prompt": example["prompt"],
            })
        return processed_data

    def verify_unit_test(self, verify_file, gt_file=None,
                         timeout_per_task=20, timeout=1800):
        """
        Run unit tests using your dataset's evaluation harness.

        Must return: (fail_ids, correct_ids, fail_feedback)
            - fail_ids: list[str] of task IDs that failed
            - correct_ids: list[str] of task IDs that passed
            - fail_feedback: str or list with failure details
        """
        ...

    def build_verify_unit_test(self, log_file_prefix, results,
                               sol_field="solution"):
        """
        Build verification file(s) consumed by verify_unit_test().

        Returns: file path (str) or None if nothing to verify.
        """
        ...

    def save_formatted_gt(self, log_file_prefix, data):
        """
        Save ground truth in the format your evaluator expects.

        Returns: file path (str) or None if not needed.
        """
        ...
```

### 3. Decide on `mark_editable_lines`

The base class provides a **concrete default** `mark_editable_lines` that works for
Python code with starter-code frozen lines. It uses keyword-based heuristics to determine
which lines can be edited or deleted during bug injection.

- **If this default works for your dataset**: do nothing (it's inherited automatically).
- **If your dataset needs different rules**: override `mark_editable_lines(self, data)` in
  your handler.

### 4. Register the handler

In `dataset/__init__.py`, add your import and registry entry:

```python
from dataset.your_dataset_name.handler import YourDatasetHandler

# Add to _REGISTRY:
_REGISTRY["your_dataset_name"] = YourDatasetHandler()
```

### 5. Vendor the evaluator under `install/`

Each dataset gets its own self-contained uv project so `verify_unit_test` can shell out
without depending on a globally-installed CLI or a pre-activated conda env.

```bash
mkdir -p dataset/<your_dataset_name>/install
cd dataset/<your_dataset_name>/install
```

Drop the evaluator's Python package(s) and a `pyproject.toml` here. Use hatchling for
the build backend (matches the existing datasets):

```toml
[project]
name = "<your_evaluator>"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    # pin the eval harness's runtime deps
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["<your_evaluator_package>"]
```

Then create the venv and install:

```bash
uv venv --python 3.10
uv sync
```

Inside `handler.verify_unit_test`, build subprocess commands with the inherited
`self.venv_cmd(module, *args)` helper from `DatasetHandler`. It returns a list like
`["<install>/.venv/bin/python", "-m", module, *args]` and raises a clear error if the
venv hasn't been created yet:

```python
subprocess.run(
    self.venv_cmd("your_evaluator.cli", "--samples", verify_file),
    cwd=self.install_dir,
    check=True,
    timeout=timeout,
)
```

`self.install_dir` and `self.venv_python` are also available on the base class. The
subprocess opens its own venv for its lifetime and exits cleanly — no activate /
deactivate plumbing required.

Finally, write a `dataset/<your_dataset_name>/README.md` with Overview / Install (uv) /
Layout / Verification / Troubleshooting sections (mirror `dataset/bigcodebench/README.md`
or `dataset/livecodebench/README.md`).

### 6. Add your data files

Place raw data files in `data/<your_dataset_name>/`.

### 7. Test

Run preprocessing to verify everything works end to end:

```bash
python src/preprocess.py --dataset_name your_dataset_name \
    --input_file your_data.json
```

## Method Reference

| Method | Purpose | Returns |
|---|---|---|
| `preprocess(raw_data)` | Raw data -> standardized format | `list[dict]` |
| `mark_editable_lines(data)` | Annotate which lines can be edited | `None` (mutates in-place) |
| `build_verify_unit_test(prefix, results, sol_field)` | Build test harness input file | file path or `None` |
| `verify_unit_test(verify_file, gt_file, ...)` | Run unit tests | `(fail_ids, correct_ids, feedback)` |
| `save_formatted_gt(prefix, data)` | Save ground truth for evaluator | file path or `None` |

## Existing Datasets

| Dataset | Status | Evaluation Tool | Install |
|---|---|---|---|
| `bigcodebench` | Complete | `bigcodebench.evaluate` (vendored v0.2.5) | `dataset/bigcodebench/install/` — `uv sync --extra eval` |
| `livecodebench` | Complete | `lcb_runner.runner.custom_evaluator` | `dataset/livecodebench/install/` — `uv sync` |

See each dataset's own `README.md` for full install + verification instructions.
