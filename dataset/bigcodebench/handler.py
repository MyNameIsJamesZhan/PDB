import copy
import json
import os
import subprocess
from pathlib import Path

from dataset.base import DatasetHandler


class BigCodeBenchHandler(DatasetHandler):
    """
    Handler for the BigCodeBench dataset.

    BigCodeBench provides coding tasks with structured prompts, canonical solutions,
    and test suites. Evaluation runs via the `bigcodebench.evaluate` CLI tool.
    """

    # Resolved relative to this file so consumers don't need to chdir to the
    # PDB repo root before invoking the handler. Override via env var if the
    # data lives elsewhere (e.g., a downstream training repo with its own layout).
    GT_DATA_PATH = os.environ.get(
        "BCB_FULL_DATA_PATH",
        str(Path(__file__).parent / "data" / "full_data.json"),
    )

    def preprocess(self, raw_data):
        """
        Transform raw BigCodeBench data into standardized format.

        Accepts either:
          - dict format (original BigCodeBench JSON keyed by task_id)
          - list format (already partially processed, each item has task_id field)
        """
        processed_data = []
        if isinstance(raw_data, dict):
            for task_id, example in raw_data.items():
                processed_data.append({
                    "task_id": task_id,
                    "gt_solution": example["code_prompt"] + "\n" + example["canonical_solution"],
                    "task_prompt": example["instruct_prompt"],
                    "test": example["test"] if "test" in example else None,
                })
        elif isinstance(raw_data, list):
            for example in raw_data:
                processed_data.append({
                    "task_id": example["task_id"],
                    "gt_solution": example["gt_solution"],
                    "task_prompt": example["task_prompt"],
                    "test": example["test"] if "test" in example else None,
                })
        else:
            raise NotImplementedError
        return processed_data

    def verify_unit_test(self, verify_file, gt_file=None, timeout_per_task=30, timeout=1800, calibrated=True, parallel=-1):
        """
        Run unit tests using the bigcodebench.evaluate CLI.

        The input file should be in JSONL format:
            {"task_id": "123xxx", "solution": "The solution to the task"}

        NOTE: BigCodeBench provides its own evaluation harness as a CLI
        tool. We shell out to it rather than importing its internals, because the harness
        manages sandboxing, timeouts, and result collection internally.
        """
        if gt_file is not None:
            assert Path(verify_file).parent == Path(gt_file).parent
            os.environ["BIGCODEBENCH_OVERRIDE_PATH"] = Path(gt_file).name
            selected_ids = None
        else:
            os.environ.pop("BIGCODEBENCH_OVERRIDE_PATH", None)
            selected_ids = ",".join([json.loads(s)["task_id"] for s in open(verify_file).readlines()])

        os.environ["BIGCODEBENCH_TIMEOUT_PER_TASK"] = str(timeout_per_task)
        workdir = Path(verify_file).parent
        base_name = Path(verify_file).with_suffix("").name
        candidates = [
            workdir / f"{base_name}_eval_results.json",
            workdir / f"{base_name}_pass_at_k.json",
        ]
        try:
            candidates[0].unlink()
            print(f"Removed existing file: {candidates[0]}")
            candidates[1].unlink()
            print(f"Removed existing file: {candidates[1]}")
        except FileNotFoundError:
            print(f"New verifying files: {candidates[0]} and {candidates[1]}")

        eval_args = [
            "--execution", "local",
            "--split", "instruct",
            "--subset", "full",
            "--samples", Path(verify_file).name,
        ]
        if selected_ids is not None:
            eval_args += ["--selective_evaluate", selected_ids]
        eval_args.append("--no_gt")
        # calibrated=True prepends code_prompt + "pass" to each solution (BCB's
        # completion-style default). Wrong for full-file/debug solutions — it doubles
        # the function def. Pass calibrated=False to run the solution as-is.
        eval_args += ["--calibrated", str(calibrated)]
        # parallel<1 -> BCB uses cpu_count()//2 workers; with unbounded BLAS threads
        # this oversubscribes cores and times out tasks. Set explicitly (with
        # OMP_NUM_THREADS=1 in the env) for deterministic, timeout-free scoring.
        if parallel and parallel > 0:
            eval_args += ["--parallel", str(parallel)]

        # Cap BLAS/OpenMP to 1 thread per worker for the scoring subprocess ONLY.
        # The BCB harness forks cpu_count()//2 workers; with unbounded BLAS threads
        # (numpy/sklearn/matplotlib) those oversubscribe the cores and stall tasks past
        # the per-task wall-clock limit -> spurious `timeout`, scored as failures
        # (observed: ~40-46% of tasks, load-dependent). Scoped via the subprocess env
        # so training/generation threading is untouched.
        score_env = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }

        try:
            subprocess.run(
                self.venv_cmd("bigcodebench.evaluate", *eval_args),
                cwd=workdir,
                check=True,
                timeout=timeout,
                env=score_env,
            )
        except subprocess.CalledProcessError as e:
            print("Command failed with an error.")
            print(f"Return Code: {e.returncode}")
        except subprocess.TimeoutExpired as e:
            print("Command timed out!")
        except TypeError:
            print("Error: A command argument was not a string. Check your variables.")

        if candidates[0].exists():
            with open(candidates[0], "r") as f:
                data = json.load(f)

            eval_dict = data.get("eval", {})

            fail_ids, correct_ids, fail_feedback = [], [], []
            for task_id, perfs in eval_dict.items():
                status = perfs[0].get("status", "fail")
                # NOTE: [design thought] Only "pass" counts as correct. Everything else
                # (fail, timeout, error) is a test failure — timeout in particular means
                # the code hung or was too slow, which is still a bug.
                if status == "pass":
                    correct_ids.append(task_id)
                else:
                    fail_ids.append(task_id)
                    fail_feedback.append(self._bcb_feedback(perfs[0].get("details")))
            return fail_ids, correct_ids, fail_feedback
        else:
            raise FileNotFoundError(f"Cannot locate evaluation results for {base_name}")

    @classmethod
    def _bcb_feedback(cls, details) -> str:
        """Turn BCB's per-test ``{test_name: traceback}`` details into uniform
        (input, expected, got) triples. BCB tests are unittest assertion methods,
        so ``input`` is the test-case name and expected/got come from the
        ``AssertionError: <got> != <expected>`` line (falling back to the raised
        exception line for non-assertion failures)."""
        if isinstance(details, dict) and details:
            return cls.format_failed_cases(
                [cls._bcb_case(name, str(trace)) for name, trace in details.items()]
            )
        if details:  # timeout/error with a non-dict detail blob
            return cls.format_failed_cases([{"got": str(details)}])
        return ""

    @staticmethod
    def _bcb_case(test_name: str, trace: str) -> dict:
        case = {"input": test_name, "expected": None, "got": None}
        idx = trace.find("AssertionError:")
        if idx != -1:
            line = trace[idx + len("AssertionError:"):].split("\n", 1)[0].strip()
            if " != " in line:
                got, expected = line.split(" != ", 1)
                case["got"], case["expected"] = got.strip(), expected.strip()
            else:
                case["got"] = line or "AssertionError"
        else:
            non_empty = [ln for ln in trace.splitlines() if ln.strip()]
            case["got"] = non_empty[-1].strip() if non_empty else "error"
        return case

    def build_worker_request(self, verify_file, gt_file=None, timeout_per_task=20,
                              timeout=1800, compact_feedback=False):
        """Shape the JSON request that the BCB persistent worker expects.

        The wire format passes verify_file + gt_file PATHS (not contents) —
        the manager has already written them via build_verify_unit_test +
        save_formatted_gt, and the worker reads them off the shared filesystem.
        Mirrors verify_unit_test()'s gt_file requirement: BCB always needs a
        gt_file because eval depends on the canonical task metadata.

        compact_feedback=True drops per-test traceback / stdout / stderr from
        fail_feedback (keeps the field as empty strings for protocol compat).
        Caller should set this in the RL reward path — the details are never
        read there, and emitting them can push the JSON response past
        asyncio's 64 KiB StreamReader limit and corrupt the flush.

        See PDB/dataset/bigcodebench/install/worker_loop.py for the receiving end.
        """
        if gt_file is None:
            raise ValueError("BCB worker requires gt_file (build via save_formatted_gt)")
        if Path(verify_file).parent != Path(gt_file).parent:
            raise ValueError(
                f"verify_file and gt_file must share a parent directory; "
                f"got {Path(verify_file).parent} vs {Path(gt_file).parent}"
            )
        return {
            "op": "score",
            "verify_file": str(verify_file),
            "gt_file": str(gt_file),
            "timeout_per_task": timeout_per_task,
            "compact_feedback": compact_feedback,
        }

    def parse_worker_response(self, resp):
        """Unpack the worker's score response into (fail_ids, correct_ids, fail_feedback).

        Identical shape to verify_unit_test() so callers can swap the two.
        """
        return resp["fail_ids"], resp["correct_ids"], resp["fail_feedback"]

    def build_verify_unit_test(self, log_file_prefix, results, sol_field="solution"):
        """
        Build a JSONL verification file for bigcodebench.evaluate.

        Each line: {"task_id": "...", "solution": "..."}
        """
        verify_file = log_file_prefix + ".jsonl"
        with open(verify_file, "w") as f:
            wrote_any = False
            for entry in results:
                if entry[sol_field] is not None:
                    json.dump({
                        "task_id": entry["task_id"],
                        "solution": entry[sol_field]
                    }, f)
                    f.write("\n")
                    wrote_any = True
        if wrote_any:
            return verify_file
        else:
            print("No submissions to evaluate.")
            return None

    def save_formatted_gt(self, log_file_prefix, data):
        """
        Save ground truth in BigCodeBench's expected format.

        NOTE: BigCodeBench's evaluator needs the full original task
        metadata (not just the solution). We load the original GT file, find the matching
        entry by task_id, and write it out. For composed bug task_ids (e.g., "BigCodeBench/123_0"),
        we strip suffixes until we find the base task_id in the original data.
        """
        original_gt_data = json.load(open(self.GT_DATA_PATH))
        gt_data = []
        for d in data:
            task_id = d["task_id"]
            while task_id not in original_gt_data:
                task_id = task_id.rsplit("_", 1)[0]
            selected = copy.deepcopy(original_gt_data[task_id])
            selected["task_id"] = d["task_id"]
            gt_data.append(selected)
        out_path = f"{log_file_prefix}.jsonl"
        with open(out_path, "w") as f:
            f.write("\n".join([json.dumps(d) for d in gt_data]))
        return out_path
