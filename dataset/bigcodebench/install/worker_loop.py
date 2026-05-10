#!/usr/bin/env python
"""Persistent BigCodeBench evaluator worker.

Spawned once per training job by the PreciseCoder reward manager (see
rllm/rewards/eval_workers/worker.py), this script keeps a hot in-process
copy of the BCB problem set and answers per-step `score` requests on
stdin/stdout. It eliminates the ~30 s per-call cold start (heavy imports +
`get_bigcodebench` load) that the legacy `subprocess.run(venv_cmd(...))`
path paid every reward call.

Wire format (JSON lines, one per request, UTF-8):

  Startup (worker → manager):
    READY\\n

  Score request (manager → worker):
    {"req_id": "<uuid>", "op": "score",
     "verify_file": "/abs/path/to/samples.jsonl",
     "gt_file":     "/abs/path/to/gt.jsonl",
     "timeout_per_task": 20}\\n

  Score response (worker → manager):
    {"req_id": "<uuid>", "ok": true,
     "fail_ids": [...], "correct_ids": [...], "fail_feedback": [...],
     "elapsed_s": 41.7}\\n

  Ping (manager → worker), used in tests:
    {"req_id": "<uuid>", "op": "ping"}\\n
    {"req_id": "<uuid>", "ok": true, "pong": true}\\n

  On worker-side exception:
    {"req_id": "<uuid>", "ok": false, "error": "...", "traceback": "..."}\\n

The manager writes verify_file + gt_file via the handler's existing
`build_verify_unit_test`/`save_formatted_gt`, then sends paths — same
on-disk shape as today's subprocess path. Worker just bypasses the
subprocess by importing `bigcodebench.evaluate` and calling `evaluate()`
in-process with the cached `problems` dict.

stderr is unbuffered text — the manager redirects it to a per-worker log
file rather than reading it, so a long traceback can never deadlock the
stdin/stdout protocol.
"""
from __future__ import annotations

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
    duration of the block. Critical: BCB's `evaluate()` prints "Reading
    samples..." and tqdm progress bars to stdout, which would contaminate
    our JSON response stream. We send those to the worker's log file via
    stderr instead. Also dups fd 1 so `print(..., file=sys.__stdout__)` and
    C-level writes (e.g., from libraries) don't escape either.
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
# everything BCB needs, so per-request invocations of `evaluate()` only do
# the actual evaluation work — no module-load tax.
from bigcodebench.evaluate import evaluate
from bigcodebench.data import get_bigcodebench
from bigcodebench.data.utils import stream_jsonl

# CLI defaults that BigCodeBenchHandler.verify_unit_test always passes:
# --execution local --split instruct --subset full --no_gt
_SPLIT = "instruct"
_SUBSET = "full"


def _emit(obj: dict) -> None:
    """Write one JSON line to stdout and flush."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _emit_error(req_id: str, exc: BaseException) -> None:
    _emit({
        "req_id": req_id,
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    })


def _handle_score(req: dict, problems: dict) -> dict:
    """Run one batch through `evaluate()` in-process.

    Mirrors BigCodeBenchHandler.verify_unit_test() exactly: same env vars
    (BIGCODEBENCH_TIMEOUT_PER_TASK), same cwd (parent of verify_file), same
    eval results JSON parse. Two differences from the legacy subprocess path:
      1. `evaluate()` is called in-process with a per-request `problems`
         dict loaded directly from the manager-written gt_file (the JSONL
         that save_formatted_gt produced). The legacy path achieved the
         same effect via BIGCODEBENCH_OVERRIDE_PATH, but that env var is
         captured at module-import time — useless in a long-lived worker.
      2. Heavy imports + HF dataset library state are reused across
         requests. The cached `problems` dict (the *full* BCB dataset
         loaded at startup) is intentionally unused per-call — we trust
         the manager's gt_file to be authoritative for this batch's task
         ids, just as BCB's own override path does.
    """
    verify_file = Path(req["verify_file"])
    gt_file = Path(req["gt_file"])
    timeout_per_task = int(req.get("timeout_per_task", 20))

    if verify_file.parent != gt_file.parent:
        raise ValueError(
            f"verify_file and gt_file must share a parent directory; "
            f"got {verify_file.parent} vs {gt_file.parent}"
        )

    # `save_formatted_gt` wrote one JSONL line per sample task_id (with the
    # SYNTHETIC suffix preserved, base task metadata copied from the full
    # dataset). Load that here so problems_subset's keys match exactly the
    # task_ids in verify_file — no need to filter or copy.
    problems_subset = {p["task_id"]: p for p in stream_jsonl(str(gt_file))}
    if not problems_subset:
        raise ValueError(f"gt_file is empty: {gt_file}")

    workdir = verify_file.parent
    base_name = verify_file.with_suffix("").name
    results_path = workdir / f"{base_name}_eval_results.json"
    pass_at_k_path = workdir / f"{base_name}_pass_at_k.json"
    # `evaluate()` short-circuits if the results JSON already exists from a
    # previous run; remove it so we always re-evaluate the new submissions.
    for stale in (results_path, pass_at_k_path):
        try:
            stale.unlink()
        except FileNotFoundError:
            pass

    os.environ["BIGCODEBENCH_TIMEOUT_PER_TASK"] = str(timeout_per_task)
    prev_cwd = os.getcwd()
    os.chdir(workdir)
    t0 = time.monotonic()
    try:
        with _stdout_to_stderr():
            evaluate(
                split=_SPLIT,
                subset=_SUBSET,
                samples=verify_file.name,
                execution="local",
                no_gt=True,
                problems=problems_subset,
            )
    finally:
        os.chdir(prev_cwd)
    elapsed = time.monotonic() - t0

    if not results_path.exists():
        raise FileNotFoundError(
            f"BCB evaluator did not produce {results_path}"
        )
    with open(results_path) as f:
        data = json.load(f)

    eval_dict = data.get("eval", {})
    fail_ids: list[str] = []
    correct_ids: list[str] = []
    fail_feedback: list[str] = []
    for task_id, perfs in eval_dict.items():
        # Mirrors handler.verify_unit_test's pass/fail logic verbatim.
        status = perfs[0].get("status", "fail")
        if status == "pass":
            correct_ids.append(task_id)
        else:
            fail_ids.append(task_id)
            fail_feedback.append(json.dumps(perfs[0].get("details", ""), indent=2))

    return {
        "fail_ids": fail_ids,
        "correct_ids": correct_ids,
        "fail_feedback": fail_feedback,
        "elapsed_s": elapsed,
    }


def main() -> int:
    # Pre-load the BCB problem set once. This is the ~30 s cost we are
    # amortising — paid here at worker startup, free for every score call
    # afterwards. Wrap in _stdout_to_stderr so any chatty load print/tqdm
    # can't leak into the JSON response stream.
    print(f"[bcb_worker] loading subset={_SUBSET}", file=sys.stderr, flush=True)
    t0 = time.monotonic()
    with _stdout_to_stderr():
        problems = get_bigcodebench(subset=_SUBSET)
    print(f"[bcb_worker] loaded {len(problems)} problems in {time.monotonic() - t0:.1f}s",
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
                result = _handle_score(req, problems)
                result.update({"req_id": req_id, "ok": True})
                _emit(result)
            elif op == "shutdown":
                _emit({"req_id": req_id, "ok": True, "shutdown": True})
                return 0
            else:
                _emit({"req_id": req_id, "ok": False,
                       "error": f"unknown op: {op!r}", "traceback": ""})
        except Exception as exc:
            print(f"[bcb_worker] req {req_id} crashed: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            _emit_error(req_id, exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
