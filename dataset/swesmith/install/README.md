# SWE-smith / SWE-bench vendored harness

These are source-only checkouts of SWE-bench and SWE-smith used by the
`swesmith` dataset handler for container-based unit-test scoring. They are added
to `sys.path` by `dataset/swesmith/handler.py`; their Python deps are **not**
installed by `uv sync`.

## Setup (Delta x86 only)

```bash
cd <repo>/PDB && uv sync
bash <repo>/scripts/swesmith/setup_env.sh   # editable installs SWE-bench + SWE-smith[validate] + docker
```

This makes `swesmith.harness.valid.run_validation` and
`swebench.harness.grading.get_valid_report` importable. The `docker` client lib
is required only because `swesmith.harness.utils` imports it at module top — the
Apptainer backend (`dataset/swesmith/exec_backend.py`) never contacts a Docker
daemon.

## Scoring backend

SWE-smith images are x86-64 only, so scoring must run on **Delta**, never
DeltaAI/ARM. Delta has only Apptainer, so the handler monkeypatches
`swesmith.harness.valid.run_patch_in_container` with the Apptainer backend when
`PDB_SWE_BACKEND=apptainer` (the default). The vendored trees here are left
pristine for upstream syncs — do not edit them.

Env knobs: `PDB_SWE_BACKEND` ∈ {apptainer,docker,none}, `PDB_SWE_SIF_DIR`
(SIF cache on Delta scratch — prestage with `scripts/swesmith/prestage_sifs.sh`),
`PDB_SWE_FAKEROOT` ∈ {1,0}.
