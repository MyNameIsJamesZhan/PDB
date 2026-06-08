"""
DatasetHandler for Docker-based evaluation.

Fix-eval mode (default): loads instance metadata from data/pdb_swe.json at init.
Bug-gen mode: validates injected bugs via Docker pre/post test comparison.

Mode is inferred from arguments:
  build_verify_unit_test: sol_field == "buggy_code"  → bug-gen
  verify_unit_test:       gt_file is not None         → bug-gen
  save_formatted_gt:      any entry has "gt_solution" → bug-gen
"""
import difflib
import json
import os
import platform
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm

# SWE-smith and SWE-bench live under dataset/swesmith/install/
_INSTALL_DIR = Path(__file__).parent / "install"
for _name in ("SWE-smith", "SWE-bench"):
    _p = str(_INSTALL_DIR / _name)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataset.base import DatasetHandler
from utils import get_indentation

_DEFAULT_DATA_FILE = Path(__file__).parent / "data" / "pdb_swe.json"
_INSTALL_MSG = (
    f"Ensure {_INSTALL_DIR}/SWE-smith and {_INSTALL_DIR}/SWE-bench exist "
    "(see dataset/swesmith/install/README.md)"
)

# Container backend selector. Real unit-test scoring runs the repo's tests
# inside a per-instance image: `docker` (x86 with Docker) or `apptainer`
# (Delta x86, the only container runtime there). `none` => structural-only,
# in which case the evaluator must not reach the verify_* paths at all.
_DEFAULT_BACKEND = "apptainer"


def _container_backend() -> str:
    return os.environ.get("PDB_SWE_BACKEND", _DEFAULT_BACKEND).strip().lower()


def _require_container_backend() -> str:
    """Guard before any container-scored verification.

    The SWE-smith images are x86-64 only, so container scoring must run on
    Delta (x86), never DeltaAI/ARM. Raises a clear, actionable error when the
    backend is missing or the architecture is incompatible.
    """
    backend = _container_backend()
    if backend in ("", "none"):
        raise RuntimeError(
            "SWE-smith unit-test scoring needs a container backend, but "
            "PDB_SWE_BACKEND=none (structural-only). Set PDB_SWE_BACKEND="
            "apptainer (Delta x86) or docker, or run the evaluator's "
            "structural-only path which skips unit scoring."
        )
    if platform.machine().lower() in ("aarch64", "arm64") and backend in ("apptainer", "docker"):
        raise RuntimeError(
            "SWE-smith container scoring must run on Delta (x86-64); the "
            f"current host is {platform.machine()}. Run scoring on Delta, or "
            "set PDB_SWE_BACKEND=none for structural-only on ARM."
        )
    return backend


def _compute_fix_patch(buggy_code: str, corrected_code: str, file_path: str) -> str:
    """Compute a unified diff patch from buggy_code to corrected_code for git apply."""
    buggy_lines = buggy_code.splitlines(keepends=True)
    corrected_lines = corrected_code.splitlines(keepends=True)
    if buggy_lines and not buggy_lines[-1].endswith("\n"):
        buggy_lines[-1] += "\n"
    if corrected_lines and not corrected_lines[-1].endswith("\n"):
        corrected_lines[-1] += "\n"
    return "".join(difflib.unified_diff(
        buggy_lines, corrected_lines,
        fromfile=f"a/{file_path}", tofile=f"b/{file_path}",
    ))


def _set_arch(repos, registry) -> None:
    """Force x86_64 on all repos — Docker Hub only has x86_64 images."""
    for repo in repos:
        try:
            registry.get(repo).arch = "x86_64"
        except Exception:
            pass


def _run_pregold(repos, registry, run_patch_in_container,
                 KEY_INSTANCE_ID, REF_SUFFIX, LOG_DIR_RUN_VALIDATION, close_logger) -> None:
    """Run the clean baseline once per repo; results are cached to disk."""
    for repo in repos:
        try:
            rp = registry.get(repo)
            ref_inst = f"{rp.repo_name}{REF_SUFFIX}"
            ref_dir = LOG_DIR_RUN_VALIDATION / repo / ref_inst
            if rp.min_pregold or ref_dir.exists():
                continue
            (LOG_DIR_RUN_VALIDATION / repo).mkdir(parents=True, exist_ok=True)
            print(f"[SWESmithHandler] Running pre-gold baseline for {repo}...")
            logger, timed_out = run_patch_in_container(
                {KEY_INSTANCE_ID: ref_inst}, repo,
                LOG_DIR_RUN_VALIDATION, rp.timeout_ref,
            )
            close_logger(logger)
            if timed_out:
                print(f"[SWESmithHandler] Pre-gold timed out for {repo}, skipping.")
        except Exception as e:
            print(f"[SWESmithHandler] Pre-gold failed for {repo}: {e}")


class SWESmithHandler(DatasetHandler):
    """DatasetHandler for Docker-based evaluation. See module docstring for modes."""

    install_subdir = "install"

    def __init__(self, data_file: str | Path | None = None):
        """Load instance metadata index from data_file (default: data/pdb_swe.json)."""
        self._data_index: dict[str, dict] = {}
        target = Path(data_file) if data_file else _DEFAULT_DATA_FILE
        if target.exists():
            data = json.load(open(target))
            self._data_index = {item["task_id"]: item for item in data}
            print(f"[SWESmithHandler] loaded {len(self._data_index)} instances from {target}")
        else:
            print(f"[SWESmithHandler] WARNING: data file not found at {target}. "
                  "Run create_swesmith_data.py to generate it.")

    # ── DatasetHandler interface ──────────────────────────────────────────────

    def mark_editable_lines(self, data):
        """Mark editable/deletable lines; frozen_lines=0 (no starter-code prefix)."""
        if not data:
            return
        for d in data:
            code_lines = d["gt_solution"].splitlines()
            d["frozen_lines"] = 0
            d["gt_length"] = len(code_lines)
            d["editable_lines"] = []
            d["deletable_lines"] = []

            for i, line in enumerate(code_lines):
                if not line.strip():
                    continue
                stripped = line.strip()
                editable = not any(kw in stripped for kw in self.NO_CHANGE_KEYWORDS)
                deletable = editable and not any(kw in stripped for kw in self.NO_DELETE_KEYWORDS)

                if editable and deletable:
                    if stripped.endswith(":"):
                        deletable = False
                    elif i > 0 and code_lines[i - 1].strip().endswith(":"):
                        if i == len(code_lines) - 1:
                            deletable = False
                        else:
                            prev_indent = get_indentation(code_lines[i - 1])
                            next_indent = get_indentation(code_lines[i + 1])
                            if prev_indent == next_indent:
                                deletable = False
                    elif (0 < i < len(code_lines) - 1
                          and not code_lines[i - 1].strip()
                          and not code_lines[i + 1].strip()):
                        deletable = False

                if editable:
                    d["editable_lines"].append((i + 1, line))
                if editable and deletable:
                    d["deletable_lines"].append((i + 1, line))

    def preprocess(self, raw_data: list[dict]) -> list[dict]:
        """Pass-through with format validation; expects 'buggy_code' or 'gt_solution'."""
        if not raw_data:
            return raw_data
        if "buggy_code" not in raw_data[0] and "gt_solution" not in raw_data[0]:
            raise ValueError(
                "SWESmithHandler.preprocess() expects 'buggy_code' (fix-eval) "
                "or 'gt_solution' (bug-gen). Got neither — check your input file."
            )
        return raw_data

    def build_verify_unit_test(
        self,
        log_file_prefix: str,
        results: list[dict],
        sol_field: str = "solution",
    ) -> str | None:
        """Write verify JSONL consumed by verify_unit_test()."""
        verify_file = f"{log_file_prefix}.jsonl"
        Path(verify_file).parent.mkdir(parents=True, exist_ok=True)
        if sol_field == "buggy_code":
            return self._build_verify_bug_gen(verify_file, results)
        return self._build_verify_fix_eval(verify_file, results, sol_field)

    def _build_verify_bug_gen(self, verify_file: str, results: list[dict]) -> str | None:
        """Write JSONL for bug-gen Docker validation."""
        parent_lookup: dict[str, dict] = {
            item["task_id"]: {
                "gt_solution": item.get("gt_solution", ""),
                "target_file": item.get("target_file", "solution.py"),
                "repo": item.get("repo", ""),
                "image_name": item.get("image_name", ""),
            }
            for item in results
            if "__atom__" not in item.get("task_id", "")
        }

        written = 0
        with open(verify_file, "w") as f:
            for item in results:
                task_id = item.get("task_id", "")
                if "__atom__" in task_id:
                    meta = parent_lookup.get(task_id.split("__atom__")[0], {})
                else:
                    meta = parent_lookup.get(task_id, {
                        "gt_solution": item.get("gt_solution", ""),
                        "target_file": item.get("target_file", "solution.py"),
                        "repo": item.get("repo", ""),
                        "image_name": item.get("image_name", ""),
                    })
                f.write(json.dumps({
                    "task_id": task_id,
                    "buggy_code": item.get("buggy_code", ""),
                    "gt_solution": meta.get("gt_solution", ""),
                    "target_file": meta.get("target_file", "solution.py"),
                    "repo": meta.get("repo", ""),
                    "image_name": meta.get("image_name", ""),
                }) + "\n")
                written += 1

        return verify_file if written > 0 else None

    def _build_verify_fix_eval(
        self, verify_file: str, results: list[dict], sol_field: str
    ) -> str | None:
        """Write JSONL for fix-eval Docker evaluation."""
        written = 0
        with open(verify_file, "w") as f:
            for item in results:
                task_id = item.get("task_id", "")
                meta = self._data_index.get(task_id, {})
                f.write(json.dumps({
                    "task_id": task_id,
                    "solution": item.get(sol_field, ""),
                    "buggy_code": meta.get("buggy_code", "") or item.get("buggy_code", ""),
                    "target_file": meta.get("target_file", "") or item.get("target_file", "solution.py"),
                    "repo": meta.get("repo", "") or item.get("repo", ""),
                    "image_name": meta.get("image_name", "") or item.get("image_name", ""),
                    "FAIL_TO_PASS": meta.get("FAIL_TO_PASS", []) or item.get("FAIL_TO_PASS", []),
                    "PASS_TO_PASS": meta.get("PASS_TO_PASS", []) or item.get("PASS_TO_PASS", []),
                    "patch": meta.get("patch", "") or item.get("patch", ""),
                }) + "\n")
                written += 1
        return verify_file if written > 0 else None

    def verify_unit_test(
        self,
        verify_file: str,
        gt_file: str | None = None,
        timeout: int = 3600,
        n_workers: int = 4,
    ) -> tuple[list[str], list[str], dict[str, str]]:
        """Run Docker evaluation; always treats this as fix-eval mode.

        NOTE: [design thought] The PDB evaluator passes a `formatted_gt` path
        so it can run the GT through the same unit-test runner used for the
        BCB / LCB sandboxes. SWE-smith's Docker harness instead validates
        fixes against the FAIL_TO_PASS / PASS_TO_PASS test sets baked into
        each instance's image, so we don't need an external GT file. We
        force `gt_file=None` unconditionally here so the evaluator stays
        uniform across handlers.

        Bug-generation for SWE-smith is driven by `src/gen_swesmith_data.py`
        and `src/create_swesmith_data.py` rather than the standard
        `bug_generation.py` path, so the bug-gen branch in this handler is
        intentionally not exercised from `verify_unit_test`.
        """
        _require_container_backend()
        entries = [json.loads(l) for l in open(verify_file) if l.strip()]
        return self._verify_fix_eval(entries, timeout, ckpt_prefix=verify_file)

    def verify_cross_fix(
        self, verify_file: str, timeout: int = 1800
    ) -> tuple[list[str], list[str], dict[str, str]]:
        """Public entry point for cross-file fix evaluation."""
        _require_container_backend()
        entries = [json.loads(l) for l in open(verify_file) if l.strip()]
        return self._verify_cross_fix_eval(entries, timeout)

    def _resolve_run_patch(self):
        """Select the container exec backend and return its run_patch_in_container.

        For PDB_SWE_BACKEND=apptainer we also rebind
        ``swesmith.harness.valid.run_patch_in_container`` so the unchanged
        ``run_validation`` routes through Apptainer. For docker we return the
        vendored implementation untouched. Raises on `none`/ARM via the guard.
        """
        backend = _require_container_backend()
        from swesmith.harness.utils import run_patch_in_container as _docker_impl
        if backend == "apptainer":
            from dataset.swesmith.exec_backend import apptainer_run_patch_in_container as _impl
            import swesmith.harness.valid as _valid
            _valid.run_patch_in_container = _impl
            return _impl
        return _docker_impl

    def _verify_fix_eval(
        self, entries: list[dict], timeout: int, ckpt_prefix: str = ""
    ) -> tuple[list[str], list[str], dict[str, str]]:
        """Apply GT→model patch via run_validation; status "0_f2p" → correct."""
        try:
            from swesmith.harness.valid import run_validation
            from swesmith.harness.utils import run_patch_in_container
            from swesmith.constants import KEY_PATCH, REF_SUFFIX, LOG_DIR_RUN_VALIDATION
            from swesmith.profiles import registry
            from swebench.harness.constants import KEY_INSTANCE_ID
            from swebench.harness.docker_build import close_logger
        except ImportError as e:
            raise ImportError(f"swesmith/swebench not importable: {e}\n{_INSTALL_MSG}") from e

        run_patch_in_container = self._resolve_run_patch()
        repos = {e.get("repo", "") for e in entries if e.get("repo")}
        _set_arch(repos, registry)
        # Establish the clean (gold) baseline per repo so get_valid_report has a
        # reference to diff against; without it every instance would score 0_f2p.
        _run_pregold(repos, registry, run_patch_in_container,
                     KEY_INSTANCE_ID, REF_SUFFIX, LOG_DIR_RUN_VALIDATION, close_logger)

        ckpt_file = Path(ckpt_prefix + "_fix_eval_ckpt.json") if ckpt_prefix else Path("/tmp/fix_eval_ckpt.json")
        ckpt: dict = {}
        if ckpt_file.exists():
            try:
                ckpt = json.load(open(ckpt_file))
                print(f"[fix-eval] Resuming from checkpoint: {len(ckpt)} / {len(entries)} done")
            except Exception:
                ckpt = {}

        fail_ids, correct_ids, fail_feedback = [], [], {}
        for task_id, result in ckpt.items():
            if result["status"] == "correct":
                correct_ids.append(task_id)
            else:
                fail_ids.append(task_id)
                fail_feedback[task_id] = result.get("feedback", "")

        for entry in tqdm.tqdm([e for e in entries if e["task_id"] not in ckpt],
                                desc="SWE-smith fix-eval"):
            task_id = entry["task_id"]
            repo = entry.get("repo", "")
            target_file = entry.get("target_file", "solution.py")
            model_solution = entry.get("solution", "")
            meta = self._data_index.get(task_id, {})
            gt_solution = meta.get("gt_solution", "") or entry.get("gt_solution", "")

            if not gt_solution or not model_solution or not repo:
                fail_ids.append(task_id)
                fail_feedback[task_id] = "missing gt_solution/solution/repo"
                ckpt[task_id] = {"status": "fail", "feedback": fail_feedback[task_id]}
                json.dump(ckpt, open(ckpt_file, "w"))
                continue

            patch = _compute_fix_patch(gt_solution, model_solution, target_file)
            if not patch:
                correct_ids.append(task_id)
                ckpt[task_id] = {"status": "correct"}
                json.dump(ckpt, open(ckpt_file, "w"))
                continue

            instance = {KEY_INSTANCE_ID: task_id, "repo": repo, KEY_PATCH: patch}
            try:
                result = run_validation(instance)
                status = result.get("status", "")
                if status == "0_f2p":
                    correct_ids.append(task_id)
                    ckpt[task_id] = {"status": "correct"}
                else:
                    fail_ids.append(task_id)
                    fail_feedback[task_id] = f"status={status}"
                    ckpt[task_id] = {"status": "fail", "feedback": f"status={status}"}
            except Exception as e:
                fail_ids.append(task_id)
                fail_feedback[task_id] = str(e)
                ckpt[task_id] = {"status": "fail", "feedback": str(e)}
            json.dump(ckpt, open(ckpt_file, "w"))

        if ckpt_file.exists() and len(ckpt) >= len(entries):
            ckpt_file.unlink()

        return fail_ids, correct_ids, fail_feedback

    def _verify_cross_fix_eval(
        self, entries: list[dict], timeout: int
    ) -> tuple[list[str], list[str], dict[str, str]]:
        """Apply GT→model patch per file via run_validation; status != "1+_f2p" → correct."""
        try:
            from swesmith.harness.valid import run_validation
            from swesmith.harness.utils import run_patch_in_container
            from swesmith.constants import KEY_PATCH, REF_SUFFIX, LOG_DIR_RUN_VALIDATION
            from swesmith.profiles import registry
            from swebench.harness.constants import KEY_INSTANCE_ID
            from swebench.harness.docker_build import close_logger
        except ImportError as e:
            raise ImportError(f"swesmith/swebench not importable: {e}\n{_INSTALL_MSG}") from e

        run_patch_in_container = self._resolve_run_patch()
        repos = {e.get("repo", "") for e in entries if e.get("repo")}
        _set_arch(repos, registry)
        _run_pregold(repos, registry, run_patch_in_container,
                     KEY_INSTANCE_ID, REF_SUFFIX, LOG_DIR_RUN_VALIDATION, close_logger)

        fail_ids, correct_ids, fail_feedback = [], [], {}
        for entry in tqdm.tqdm(entries, desc="SWE-smith cross fix-eval"):
            task_id = entry.get("task_id", "")
            gt_solution = entry.get("gt_solution", "")
            corrected = entry.get("solution", "")
            target_file = entry.get("target_file", "solution.py")
            repo = entry.get("repo", "")

            if not repo or not gt_solution:
                fail_ids.append(task_id)
                fail_feedback[task_id] = "missing repo/gt_solution"
                continue
            if not corrected:
                fail_ids.append(task_id)
                fail_feedback[task_id] = "empty solution"
                continue

            patch = _compute_fix_patch(gt_solution, corrected, target_file)
            if not patch:
                correct_ids.append(task_id)
                continue

            instance = {KEY_INSTANCE_ID: task_id, "repo": repo, KEY_PATCH: patch}
            try:
                result = run_validation(instance)
                if result.get("status", "fail") != "1+_f2p":
                    correct_ids.append(task_id)
                else:
                    fail_ids.append(task_id)
                    fail_feedback[task_id] = "status=1+_f2p"
            except Exception as e:
                fail_ids.append(task_id)
                fail_feedback[task_id] = str(e)

        return fail_ids, correct_ids, fail_feedback

    def _verify_bug_gen(
        self, entries: list[dict], timeout: int, n_workers: int = 4,
    ) -> tuple[list[str], list[str], dict[str, str]]:
        """
        Validate that each entry's buggy_code causes test failures.

        fail_ids    → buggy code fails tests (valid bug)
        correct_ids → buggy code passes tests (invalid or trivially fixable)
        Dropped entries (Docker error) appear in neither list.
        """
        _require_container_backend()
        try:
            from swesmith.harness.valid import run_validation
            from swesmith.harness.utils import run_patch_in_container
            from swesmith.constants import KEY_PATCH, REF_SUFFIX, LOG_DIR_RUN_VALIDATION
            from swesmith.profiles import registry
            from swebench.harness.constants import KEY_INSTANCE_ID, LOG_REPORT, FAIL_TO_PASS
            from swebench.harness.docker_build import close_logger
        except ImportError as e:
            raise ImportError(f"swesmith/swebench not importable: {e}\n{_INSTALL_MSG}") from e

        run_patch_in_container = self._resolve_run_patch()
        repos = {e.get("repo", "") for e in entries if e.get("repo")}
        _set_arch(repos, registry)
        _run_pregold(repos, registry, run_patch_in_container,
                     KEY_INSTANCE_ID, REF_SUFFIX, LOG_DIR_RUN_VALIDATION, close_logger)

        def _validate_one(entry):
            task_id = entry.get("task_id", "")
            buggy_code = entry.get("buggy_code", "")
            gt_solution = entry.get("gt_solution", "")
            target_file = entry.get("target_file", "solution.py")
            repo = entry.get("repo", "")

            if not repo or not gt_solution or not buggy_code:
                return "drop", task_id, "missing repo/gt_solution/buggy_code"

            bug_patch = _compute_fix_patch(gt_solution, buggy_code, target_file)
            if not bug_patch:
                return "correct", task_id, None

            instance = {KEY_INSTANCE_ID: task_id, "repo": repo, KEY_PATCH: bug_patch}
            try:
                result = run_validation(instance)
                status = result.get("status", "fail")
                if status == "1+_f2p":
                    f2p = None
                    report_path = LOG_DIR_RUN_VALIDATION / repo / task_id / LOG_REPORT
                    if report_path.exists():
                        try:
                            report = json.loads(report_path.read_text())
                            f2p = {
                                "FAIL_TO_PASS": report.get("FAIL_TO_PASS", []),
                                "PASS_TO_PASS": report.get("PASS_TO_PASS", []),
                            }
                        except Exception:
                            pass
                    return "fail", task_id, f2p
                if status == "0_f2p":
                    return "correct", task_id, None
                return "drop", task_id, f"status={status}"
            except Exception as e:
                return "drop", task_id, str(e)

        workers = min(n_workers, len(entries)) if entries else 1
        print(f"[SWESmithHandler] Validating {len(entries)} bugs with {workers} parallel workers...")
        fail_ids, correct_ids, fail_feedback = [], [], {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_validate_one, e): e for e in entries}
            for fut in tqdm.tqdm(as_completed(futures), total=len(entries),
                                 desc="SWE-smith bug-gen validation"):
                outcome, task_id, info = fut.result()
                if outcome == "fail":
                    fail_ids.append(task_id)
                    if info:
                        fail_feedback[task_id] = info
                elif outcome == "correct":
                    correct_ids.append(task_id)
                elif info:
                    fail_feedback[task_id] = info

        return fail_ids, correct_ids, fail_feedback

    def save_formatted_gt(
        self,
        log_file_prefix: str,
        data: list[dict],
    ) -> str | None:
        """Write GT JSONL for bug-gen mode; return None in fix-eval mode."""
        has_gt = any(
            item.get("gt_solution") and "__atom__" not in item.get("task_id", "")
            for item in data
        )
        if not has_gt:
            return None

        gt_file = f"{log_file_prefix}.jsonl"
        Path(gt_file).parent.mkdir(parents=True, exist_ok=True)
        with open(gt_file, "w") as f:
            for item in data:
                if item.get("gt_solution") and "__atom__" not in item.get("task_id", ""):
                    f.write(json.dumps({
                        "task_id": item["task_id"],
                        "gt_solution": item["gt_solution"],
                        "target_file": item.get("target_file", "solution.py"),
                    }) + "\n")
        return gt_file
