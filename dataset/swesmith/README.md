# SWE-smith

Handler + vendored dependencies for the [SWE-smith](https://github.com/SWE-bench/SWE-smith) dataset.

## Overview

`SWE-smith` provides Docker-based Python repository environments derived from real GitHub issues. It is the source of the **228 real-world repository bug examples** that make up the SWE-smith partition of the released **PDB-Wild** evaluation set (the other 256 examples are synthesized multi-line bugs over BigCodeBench / LiveCodeBench).

In `PDB`, [handler.py](handler.py) implements the `SWESmithHandler` class. It is wired into `bash scripts/{simple,run}_debug_eval.sh wild`:

- The evaluator's `verify_unit_test` call applies the model's predicted fix as a unified diff and runs it inside the repo's Docker container, checking whether the originally-failing `FAIL_TO_PASS` tests now pass and the `PASS_TO_PASS` tests still pass.
- `verify_unit_test` always runs in fix-eval mode (any `gt_file` argument is intentionally ignored); SWE-smith's harness validates fixes against the test sets baked into each instance's image, so an external GT file is unnecessary.

The handler imports `swesmith` and `swebench` directly from the source trees under [install/](install/) — no pip install required, no venv to activate. `swebench` is a hard dependency of `swesmith` itself (used inside `swesmith.harness.valid`).

Bug-generation for SWE-smith is driven by [`src/gen_swesmith_data.py`](../../src/gen_swesmith_data.py) and [`src/create_swesmith_data.py`](../../src/create_swesmith_data.py) — the standard `bug_generation.py --dataset_name swesmith` path is intentionally not exercised, since the SWE-smith pipeline starts from real-world repository patches rather than line-level synthesis.

## Install

Both source trees are vendored in [install/](install/) and committed to this repo — no cloning required. Docker images are pulled automatically at runtime; no Docker Hub login is required. All 120 eligible repos use public `x86_64` images under the `swebench/` org.

## Layout

```
dataset/swesmith/
├── README.md                    # this file
├── handler.py                   # SWESmithHandler — bug-gen and fix-eval modes
├── data/
│   ├── pdb_swe.json             # fix-eval instance metadata (from create_swesmith_data.py)
│   └── pdb_swe_input.json       # raw file extracts for bug injection (from gen_swesmith_data.py)
└── install/                     # vendored source trees (no pip install needed)
    ├── README.md                # install notes
    ├── SWE-smith/               # vendored SWE-smith source
    │   └── swesmith/
    └── SWE-bench/               # vendored SWE-bench source (hard dep of swesmith)
        └── swebench/
```

## Verification

Confirm the handler resolves its install dir and loads the index:

```bash
python -c "
from dataset.swesmith.handler import SWESmithHandler, _INSTALL_DIR
print('install dir:', _INSTALL_DIR)
print('SWE-smith exists:', (_INSTALL_DIR / 'SWE-smith').exists())
print('SWE-bench  exists:', (_INSTALL_DIR / 'SWE-bench').exists())
h = SWESmithHandler()
"

## Troubleshooting

- **`ImportError: swesmith/swebench not importable`** — both source trees must exist under `install/`; check that the repo was cloned with full history (not a shallow clone that omitted large files).
- **`[SWESmithHandler] WARNING: data file not found`** — `data/pdb_swe.json` is missing. Generate it with `python src/create_swesmith_data.py` or copy the canonical dataset.
- **`platform mismatch` / container won't start on Apple Silicon** — Docker Hub only has `x86_64` images. The handler forces `arch = "x86_64"` via the swesmith registry before any container operation; ensure Docker Desktop has Rosetta emulation enabled.
- **Pre-gold baseline times out** — some repos have slow test suites. The baseline log is cached under `logs/run_validation/<repo>/`; once written it is reused on subsequent runs.
