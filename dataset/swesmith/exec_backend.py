"""Apptainer execution backend for SWE-smith validation on Delta (x86-64).

Delta provides only Apptainer (no Docker/podman), so this module supplies a
drop-in replacement for ``swesmith.harness.utils.run_patch_in_container`` that
runs the per-instance image with Apptainer instead of the Docker SDK. The
handler monkeypatches ``swesmith.harness.valid.run_patch_in_container`` with
``apptainer_run_patch_in_container`` when ``PDB_SWE_BACKEND=apptainer``; every
layer above it (``run_validation``, ``get_valid_report``) is reused unchanged
because this function reproduces the vendored function's *side-effects*:

  * write the combined test output to ``<log_dir>/<run_id>/<instance_id>/test_output.txt``
    (``LOG_TEST_OUTPUT``) bracketed by the TEST_OUTPUT_START/END markers, and
  * return ``(logger, timed_out)``.

The SWE-smith images are x86-64 only — this backend must run on Delta, never on
DeltaAI/ARM (the handler guards that separately).

Writable strategy (``PDB_SWE_FAKEROOT``):
  * ``1`` (default): ``--fakeroot --writable-tmpfs`` — we are root inside the
    container (root-mapped namespace on Delta) and the per-exec tmpfs overlay is
    discarded afterwards. Verified to work with the real swesmith images.
  * ``0``: ``--writable-tmpfs`` only — use when fakeroot is unavailable AND the
    image's /testbed is writable by the invoking uid (rare). When neither holds,
    build a per-repo writable sandbox out-of-band and point the backend at it.

NOTE: ``--writable-tmpfs`` overlays are per-``exec`` and ephemeral, so the patch
apply and the test run happen in a single ``apptainer exec`` invocation.
"""
import os
import shlex
import subprocess
from pathlib import Path

from unidiff import PatchSet

from swebench.harness.constants import (
    DOCKER_WORKDIR,
    KEY_INSTANCE_ID,
    LOG_INSTANCE,
    LOG_TEST_OUTPUT,
    TESTS_TIMEOUT,
)
from swebench.harness.docker_build import setup_logger
from swesmith.constants import TEST_OUTPUT_START, TEST_OUTPUT_END
from swesmith.profiles import registry

# Same ordered fallbacks the Docker harness uses (swesmith.harness.utils).
GIT_APPLY_CMDS = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fuzz=5 -p1 -i",
]

_APPLY_FAILED_RC = 3  # sentinel exit code: patch did not apply -> treat as fail


def _sif_dir() -> Path:
    return Path(os.environ.get("PDB_SWE_SIF_DIR", "/tmp/pdb_swe_sif"))


def _fakeroot() -> bool:
    return os.environ.get("PDB_SWE_FAKEROOT", "1").strip() == "1"


def sif_path_for(image_name: str) -> Path:
    """Resolve (and lazily pull) the SIF for a Docker image name."""
    sif = _sif_dir() / (image_name.replace("/", "__").replace(":", "__") + ".sif")
    if not sif.exists():
        sif.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["apptainer", "pull", str(sif), f"docker://{image_name}"], check=True
        )
    return sif


def _build_inner_script(patch, commit, is_gold) -> str:
    """Bash run inside the container: optional checkout, optional patch, eval."""
    lines = ["set -o pipefail", f"cd {shlex.quote(DOCKER_WORKDIR)}"]
    if commit is not None:
        lines += ["git fetch || true", f"git checkout {shlex.quote(commit)}"]
    if patch:
        changed = " ".join(shlex.quote(f.path) for f in PatchSet(patch))
        if changed:
            lines.append(f"git checkout -- {changed} || true")
        reverse = " --reverse" if is_gold else ""
        attempts = " || ".join(f"{c}{reverse} /pdb_io/patch.diff" for c in GIT_APPLY_CMDS)
        # If no apply variant succeeds, bail with a sentinel so the caller can
        # mark the instance failed instead of running tests on unpatched code.
        lines.append(f"if ! ( {attempts} ); then echo PDB_APPLY_FAILED; exit {_APPLY_FAILED_RC}; fi")
    lines.append("bash /pdb_io/eval.sh")
    return "\n".join(lines)


def apptainer_run_patch_in_container(
    instance: dict,
    run_id: str,
    log_dir,
    timeout: int,
    patch: str | None = None,
    commit: str | None = None,
    f2p_only: bool = False,
    is_gold: bool = False,
):
    """Apptainer drop-in for ``run_patch_in_container``; returns (logger, timed_out).

    Mirrors the vendored function's on-disk contract so ``run_validation`` and
    ``get_valid_report`` work unchanged.
    """
    instance_id = instance[KEY_INSTANCE_ID]
    rp = registry.get_from_inst(instance)

    log_dir = Path(log_dir) / run_id / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        f"swesmith.apptainer.{run_id}.{instance_id}", log_dir / LOG_INSTANCE
    )

    # eval.sh: the repo's FAIL_TO_PASS/PASS_TO_PASS test command, bracketed by
    # the markers get_valid_report's log parser looks for.
    test_command, _ = rp.get_test_cmd(instance, f2p_only=f2p_only)
    (log_dir / "eval.sh").write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "set -uxo pipefail",
                f"cd {DOCKER_WORKDIR}",
                f": '{TEST_OUTPUT_START}'",
                test_command,
                f": '{TEST_OUTPUT_END}'",
            ]
        )
        + "\n"
    )
    if patch:
        (log_dir / "patch.diff").write_text(patch)

    inner = _build_inner_script(patch, commit, is_gold)

    cmd = ["apptainer", "exec", "--no-home", "--cleanenv"]
    cmd += ["--fakeroot", "--writable-tmpfs"] if _fakeroot() else ["--writable-tmpfs"]
    cmd += [
        "--pwd",
        DOCKER_WORKDIR,
        "--bind",
        f"{log_dir}:/pdb_io",
        str(sif_path_for(rp.image_name)),
        "bash",
        "-c",
        inner,
    ]

    logger.info(f"Running instance {instance_id} via apptainer (fakeroot={_fakeroot()})")
    timed_out = False
    # Merge stderr into stdout so the `set -x` trace (which carries the
    # TEST_OUTPUT_START/END markers on stderr) stays interleaved with pytest's
    # stdout — get_valid_report extracts the test section *between* those
    # markers, so the original ordering must be preserved (Docker's exec_run
    # returns a single combined stream; capturing the two pipes separately and
    # concatenating would push the markers past all test output).
    try:
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        timed_out = True
        out = e.stdout or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        combined = f"{out}\n\n{TESTS_TIMEOUT}: {timeout} seconds exceeded"
        (log_dir / LOG_TEST_OUTPUT).write_text(combined)
        logger.info(f"Timed out for {instance_id} after {timeout}s")
        return logger, timed_out

    combined = proc.stdout or ""
    # Patch failed to apply: do NOT write the test-output log. run_validation
    # then sees the missing pre-gold output and returns status "fail" — the same
    # outcome the Docker harness produces when _apply_patch raises.
    if proc.returncode == _APPLY_FAILED_RC or "PDB_APPLY_FAILED" in combined:
        logger.info(f"Patch failed to apply for {instance_id}; marking failed.")
        (log_dir / "apply_failure.log").write_text(combined)
        return logger, timed_out

    (log_dir / LOG_TEST_OUTPUT).write_text(combined)
    logger.info(f"Test output for {instance_id} written ({len(combined)} chars)")
    return logger, timed_out
