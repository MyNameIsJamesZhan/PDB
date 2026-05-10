#!/usr/bin/env python
"""Persistent LiveCodeBench evaluator worker.

Spawned once per training job by the PreciseCoder reward manager (see
rllm/rewards/eval_workers/worker.py), this script keeps a hot in-process
copy of the LCB benchmark and answers per-step `score` requests on
stdin/stdout. It eliminates the ~150 s per-call cold start (heavy imports +
`load_code_generation_dataset`) that the legacy `subprocess.run(venv_cmd(...))`
path paid every reward call.

Wire format (JSON lines, one per request, UTF-8):

  Startup (worker → manager):
    READY\\n

  Score request (manager → worker):
    {"req_id": "<uuid>", "op": "score",
     "verify_file": "/abs/path/to/samples.json",
     "map_file":    "/abs/path/to/samples_map.json"}\\n

  Score response (worker → manager):
    {"req_id": "<uuid>", "ok": true,
     "fail_ids": [...], "correct_ids": [...], "fail_feedback": [...],
     "elapsed_s": 41.7}\\n

The manager writes verify_file + map_file via
LiveCodeBenchHandler.build_verify_unit_test, then sends paths — same on-disk
shape as today's subprocess path. Worker just bypasses the subprocess by
calling `lcb_runner.runner.custom_evaluator.run()` in-process with the
cached `benchmark`.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


@contextlib.contextmanager
def _stdout_to_stderr():
    """Redirect everything written to sys.stdout to sys.stderr for the
    duration of the block. Critical: LCB's evaluator and its dependencies
    print progress / "Loaded N problems" to stdout, which would contaminate
    our JSON response stream. We send those to the worker's log file via
    stderr instead. Also dups fd 1 so C-level writes don't escape either.
    """
    old_stdout = sys.stdout
    saved_fd = os.dup(1)
    try:
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        yield
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        sys.stdout = old_stdout

# Pay the heavy imports once at startup. After this point, sys.modules holds
# everything LCB needs, so per-request invocations of `run()` only do the
# actual evaluation work — no module-load tax.
from lcb_runner.runner.custom_evaluator import run as run_custom_evaluator
from lcb_runner.runner.scenario_router import build_prompt_benchmark
from lcb_runner.utils.scenarios import Scenario


def _make_args(custom_output_file: str | None = None) -> argparse.Namespace:
    """Build the args namespace LCB's run()/build_prompt_benchmark expect.

    Mirrors get_args()'s defaults for every field run() and
    build_prompt_benchmark touch — release_version, scenario, dates, etc.
    Only `custom_output_file` varies per request.
    """
    return argparse.Namespace(
        scenario=Scenario.codegeneration,
        not_fast=False,
        release_version="release_latest",
        start_date=None,
        end_date=None,
        cot_code_execution=False,
        custom_output_file=custom_output_file,
        custom_output_save_name=None,
        num_process_evaluate=12,
        timeout=6,
    )


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _emit_error(req_id: str, exc: BaseException) -> None:
    _emit({
        "req_id": req_id,
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    })


def _parse_eval(eval_path: Path,
                ordered_qids: list[str],
                qid_to_full_ids: dict[str, list[str]]
                ) -> tuple[list[str], list[str]]:
    """Inline LiveCodeBenchHandler.verify_unit_test()'s rich-output parse."""
    with open(eval_path) as f:
        eval_data = json.load(f)

    if not (isinstance(eval_data, list) and len(eval_data) > 1 and isinstance(eval_data[1], dict)):
        raise ValueError("Unexpected LiveCodeBench rich output format; missing per-index results")
    per_index = eval_data[1]

    sorted_qids = sorted(ordered_qids)
    qid_to_results = {}
    for idx, qid in enumerate(sorted_qids):
        key = str(idx)
        if key in per_index:
            qid_to_results[qid] = per_index[key]

    fail_ids: list[str] = []
    correct_ids: list[str] = []
    for qid in ordered_qids:
        if qid not in qid_to_results:
            continue
        candidate_results = qid_to_results[qid]
        if not isinstance(candidate_results, list) or len(candidate_results) == 0:
            continue
        full_ids = qid_to_full_ids.get(qid, [qid] * len(candidate_results))
        num_to_map = min(len(candidate_results), len(full_ids))
        for j in range(num_to_map):
            tests = candidate_results[j]
            passed = False
            if isinstance(tests, list) and len(tests) > 0:
                if all(isinstance(e, bool) for e in tests):
                    passed = all(tests)
                else:
                    has_error = any(isinstance(e, (int, float)) and e < 0 for e in tests)
                    all_true_bools = all((e is True) for e in tests if isinstance(e, bool))
                    passed = (not has_error) and all_true_bools
            elif isinstance(tests, bool):
                passed = tests
            (correct_ids if passed else fail_ids).append(full_ids[j])

        if len(full_ids) > num_to_map:
            for j in range(num_to_map, len(full_ids)):
                fail_ids.append(full_ids[j])

    return fail_ids, correct_ids


def _handle_score(req: dict, benchmark: list) -> dict:
    verify_file = Path(req["verify_file"])
    map_file = Path(req.get("map_file", verify_file.with_name(verify_file.stem + "_map.json")))

    if not verify_file.exists():
        raise FileNotFoundError(f"verify_file does not exist: {verify_file}")

    with open(verify_file) as f:
        verify_input = json.load(f)
    ordered_qids = [d.get("question_id") for d in verify_input]
    qid_to_full_ids = {}
    if map_file.exists():
        with open(map_file) as f:
            qid_to_full_ids = json.load(f)

    if not ordered_qids:
        return {"fail_ids": [], "correct_ids": [], "fail_feedback": [], "elapsed_s": 0.0}

    args = _make_args(custom_output_file=str(verify_file))
    t0 = time.monotonic()
    with _stdout_to_stderr():
        run_custom_evaluator(args, benchmark=benchmark)
    elapsed = time.monotonic() - t0

    eval_path = verify_file.with_name(verify_file.stem + "_output_eval.json")
    if not eval_path.exists():
        raise FileNotFoundError(
            f"LCB evaluator did not produce {eval_path}"
        )
    fail_ids, correct_ids = _parse_eval(eval_path, ordered_qids, qid_to_full_ids)

    return {
        "fail_ids": fail_ids,
        "correct_ids": correct_ids,
        "fail_feedback": [""] * len(fail_ids),
        "elapsed_s": elapsed,
    }


def main() -> int:
    # Mirror the legacy `cwd=self.install_dir` from
    # LiveCodeBenchHandler.verify_unit_test() — some lcb_runner internals may
    # reference paths relative to the install dir (e.g., cached datasets).
    install_dir = Path(__file__).resolve().parent
    os.chdir(install_dir)

    print("[lcb_worker] loading benchmark", file=sys.stderr, flush=True)
    t0 = time.monotonic()
    with _stdout_to_stderr():
        benchmark, _ = build_prompt_benchmark(_make_args())
    print(f"[lcb_worker] loaded {len(benchmark)} problems in {time.monotonic() - t0:.1f}s",
          file=sys.stderr, flush=True)

    sys.stdout.write("READY\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _emit({"req_id": "?", "ok": False,
                   "error": f"JSONDecodeError: {e}", "traceback": ""})
            continue

        req_id = req.get("req_id", "?")
        op = req.get("op")
        try:
            if op == "ping":
                _emit({"req_id": req_id, "ok": True, "pong": True})
            elif op == "score":
                result = _handle_score(req, benchmark)
                result.update({"req_id": req_id, "ok": True})
                _emit(result)
            elif op == "shutdown":
                _emit({"req_id": req_id, "ok": True, "shutdown": True})
                return 0
            else:
                _emit({"req_id": req_id, "ok": False,
                       "error": f"unknown op: {op!r}", "traceback": ""})
        except Exception as exc:
            print(f"[lcb_worker] req {req_id} crashed: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            _emit_error(req_id, exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
