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

import atexit
import contextlib
import json
import multiprocessing
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
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
# everything BCB needs, so per-request invocations only do the actual
# evaluation work — no module-load tax.
#
# We import `untrusted_check` from `bigcodebench.eval` (NOT
# `bigcodebench.evaluate`). The latter has module-load side effects:
# `os.setsid()`, SIGTERM `_kill_process_group` handler, and an atexit hook
# (evaluate.py:33-78). Those make a long-lived persistent-pool worker hard
# to clean up. `bigcodebench.eval` has no such side effects.
from bigcodebench.data import get_bigcodebench, load_solutions
from bigcodebench.data.utils import stream_jsonl
from bigcodebench.eval import untrusted_check

# Legacy path (PDB_PERSISTENT_POOL!=1) still calls the high-level
# evaluate() which builds a fresh ProcessPoolExecutor every flush. Kept
# behind a gate so we can A/B test against the new persistent-pool path
# without ripping the old code out.
_USE_PERSISTENT_POOL = os.environ.get("PDB_PERSISTENT_POOL", "0") == "1"
if not _USE_PERSISTENT_POOL:
    from bigcodebench.evaluate import evaluate  # noqa: E402

# CLI defaults that BigCodeBenchHandler.verify_unit_test always passes:
# --execution local --split instruct --subset full --no_gt
_SPLIT = "instruct"
_SUBSET = "full"

# Persistent ProcessPoolExecutor — built once in main() if
# PDB_PERSISTENT_POOL=1. None if gated off.
_POOL: ProcessPoolExecutor | None = None
# Counter for periodic pool recycling. Python 3.10 doesn't support
# `max_tasks_per_child`, so we recycle the whole pool every N flushes to
# mitigate heap fragmentation.
_POOL_FLUSH_COUNTER = 0
_POOL_RECYCLE_EVERY = int(os.environ.get("PDB_BCB_POOL_RECYCLE_EVERY", "50"))


def _make_pool() -> ProcessPoolExecutor:
    # Respect cgroup CPU limit via sched_getaffinity rather than the raw
    # hardware thread count from cpu_count(). On Grace Hopper nodes
    # cpu_count() reports 288 threads but Slurm typically allocates ~16,
    # so the old default cpu_count()//2 = 144 over-subscribed by ~9x and
    # added scheduler contention that drowned the persistent-pool win.
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = multiprocessing.cpu_count()
    # Leave 2 cores for the parent + LCB pool; cap at 12 so we don't push
    # the per-task Process count past what 16 CPUs can actually run.
    default = max(1, min(affinity - 2, 12))
    n = int(os.environ.get("PDB_BCB_POOL_SIZE", str(default)))
    print(f"[bcb_worker] spawning persistent ProcessPoolExecutor max_workers={n} "
          f"(affinity={affinity})",
          file=sys.stderr, flush=True)
    t0 = time.monotonic()
    pool = ProcessPoolExecutor(max_workers=n)
    # ProcessPoolExecutor.__init__ does NOT fork workers — they're spawned
    # lazily on first submit(). Force the fork now by running a no-op
    # future per worker, so the first real flush doesn't pay the fork
    # cost. Submit 2*n so every slot is touched (futures are racy across
    # workers; oversubscribing the warm-up is the simplest reliable way).
    warm_futures = [pool.submit(os.getpid) for _ in range(n * 2)]
    warm_pids = {f.result() for f in warm_futures}
    print(f"[bcb_worker] pool warmed in {time.monotonic() - t0:.1f}s "
          f"(forked {len(warm_pids)} workers)",
          file=sys.stderr, flush=True)
    return pool


def _ensure_pool() -> ProcessPoolExecutor:
    """Lazy-construct the pool and recycle it every _POOL_RECYCLE_EVERY
    flushes. Recycling caps long-run RSS drift from Python heap
    fragmentation.
    """
    global _POOL, _POOL_FLUSH_COUNTER
    if _POOL is None:
        _POOL = _make_pool()
    elif _POOL_FLUSH_COUNTER >= _POOL_RECYCLE_EVERY:
        print(f"[bcb_worker] recycling pool after {_POOL_FLUSH_COUNTER} flushes",
              file=sys.stderr, flush=True)
        _POOL.shutdown(wait=True, cancel_futures=False)
        _POOL = _make_pool()
        _POOL_FLUSH_COUNTER = 0
    return _POOL


def _shutdown_pool() -> None:
    global _POOL
    if _POOL is not None:
        try:
            _POOL.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        _POOL = None


atexit.register(_shutdown_pool)


def _local_check_correctness(
    completion_id: int,
    problem: dict,
    solution: str,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    identifier,
    min_time_limit: float,
    gt_time_limit: float,
    timeout_per_task: int,
) -> dict:
    """Pool-worker entrypoint. Wraps `untrusted_check` and returns the
    same dict shape that `bigcodebench.evaluate.check_correctness` does.

    `BIGCODEBENCH_TIMEOUT_PER_TASK` is read inside `untrusted_check`
    (eval/__init__.py:182). Pool children were forked BEFORE the master
    set it in `_handle_score`, so they have the master's env-at-fork
    value. Set it here in the worker so the per-request value sticks.
    """
    os.environ["BIGCODEBENCH_TIMEOUT_PER_TASK"] = str(int(timeout_per_task))
    stat, details = untrusted_check(
        solution,
        problem["test"],
        problem["entry_point"],
        max_as_limit,
        max_data_limit,
        max_stack_limit,
        min_time_limit,
        gt_time_limit,
    )
    return {
        "completion_id": completion_id,
        "task_id": problem["task_id"],
        "_identifier": identifier,
        "solution": solution,
        "base": (stat, details),
    }


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


def _parse_results_dict(eval_dict: dict) -> tuple[list[str], list[str], list[str]]:
    """Walk the `evaluate()`-shape results dict and split into
    (fail_ids, correct_ids, fail_feedback). Shared by both legacy and
    persistent-pool paths.
    """
    fail_ids: list[str] = []
    correct_ids: list[str] = []
    fail_feedback: list[str] = []
    for task_id, perfs in eval_dict.items():
        status = perfs[0].get("status", "fail")
        if status == "pass":
            correct_ids.append(task_id)
        else:
            fail_ids.append(task_id)
            fail_feedback.append(json.dumps(perfs[0].get("details", ""), indent=2))
    return fail_ids, correct_ids, fail_feedback


def _handle_score_pool(req: dict) -> dict:
    """Persistent-pool path: dispatch each sample as its own future on
    a long-lived ProcessPoolExecutor. Eliminates the per-flush pool
    spawn that `evaluate()` does inside its `with ProcessPoolExecutor`
    block.

    Per-task isolation is unchanged: `untrusted_check` still spawns a
    fresh `multiprocessing.Process` per candidate for sandboxing (the
    pool worker only dispatches).
    """
    verify_file = Path(req["verify_file"])
    gt_file = Path(req["gt_file"])
    timeout_per_task = int(req.get("timeout_per_task", 20))

    if verify_file.parent != gt_file.parent:
        raise ValueError(
            f"verify_file and gt_file must share a parent directory; "
            f"got {verify_file.parent} vs {gt_file.parent}"
        )

    problems_subset = {p["task_id"]: p for p in stream_jsonl(str(gt_file))}
    if not problems_subset:
        raise ValueError(f"gt_file is empty: {gt_file}")

    workdir = verify_file.parent
    base_name = verify_file.with_suffix("").name
    results_path = workdir / f"{base_name}_eval_results.json"
    pass_at_k_path = workdir / f"{base_name}_pass_at_k.json"
    for stale in (results_path, pass_at_k_path):
        try:
            stale.unlink()
        except FileNotFoundError:
            pass

    # Defaults from bigcodebench.evaluate.evaluate signature. We never
    # ran with custom values, so hard-code them. calibrated=True matches
    # the legacy handler path (handler.py never passes --calibrated).
    max_as_limit = 30 * 1024
    max_data_limit = 30 * 1024
    max_stack_limit = 10
    min_time_limit = 10.0
    # no_gt=True branch in evaluate.py:383 falls back to gt_time_limit=20.
    gt_time_limit = 20.0

    t_read_start = time.monotonic()
    pool = _ensure_pool()
    t_pool = time.monotonic()

    t0 = time.monotonic()

    completion_id: Counter[str] = Counter()
    futures = []
    # Re-mirror evaluate.py's solution-assembly logic exactly so behavior
    # is byte-identical to the legacy path (see evaluate.py:363-384).
    with _stdout_to_stderr():
        for sample in load_solutions(str(verify_file)):
            tid = sample["task_id"]
            if tid not in problems_subset:
                continue
            if "solution" in sample:
                solution = sample["solution"]
            else:
                solution = problems_subset[tid]["complete_prompt"] + sample["completion"]
            # calibrated=True default — see evaluate.py:368-369.
            solution = problems_subset[tid]["code_prompt"] + "\n    pass\n" + solution
            args = (
                completion_id[tid],
                problems_subset[tid],
                solution,
                max_as_limit,
                max_data_limit,
                max_stack_limit,
                sample["_identifier"],
                min_time_limit,
                gt_time_limit,
                timeout_per_task,
            )
            futures.append(pool.submit(_local_check_correctness, *args))
            completion_id[tid] += 1
    t_submit = time.monotonic()

    eval_results: dict[str, list[dict]] = defaultdict(list)
    for fut in as_completed(futures):
        r = fut.result()
        eval_results[r["task_id"]].append(r)
    t_wait = time.monotonic()

    # Build the same {"date":..., "eval": {tid: [...]}} dict that
    # `evaluate()` writes — downstream `_parse_results_dict` and any
    # external consumer of the results file expect this exact shape.
    eval_dict: dict[str, list[dict]] = {}
    for tid, rs in eval_results.items():
        rs.sort(key=lambda x: x["completion_id"])
        eval_dict[tid] = [
            {
                "task_id": tid,
                "solution": r["solution"],
                "status": r["base"][0],
                "details": r["base"][1],
            }
            for r in rs
        ]
    results = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "eval": eval_dict,
    }
    t_aggregate = time.monotonic()

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    t_write = time.monotonic()

    elapsed = time.monotonic() - t0

    print(f"[bcb_worker] flush phases n_futures={len(futures)}: "
          f"pool={t_pool - t_read_start:.2f}s "
          f"submit={t_submit - t_pool:.2f}s "
          f"wait={t_wait - t_submit:.2f}s "
          f"agg={t_aggregate - t_wait:.2f}s "
          f"write={t_write - t_aggregate:.2f}s "
          f"total={elapsed:.2f}s",
          file=sys.stderr, flush=True)

    global _POOL_FLUSH_COUNTER
    _POOL_FLUSH_COUNTER += 1

    fail_ids, correct_ids, fail_feedback = _parse_results_dict(eval_dict)
    return {
        "fail_ids": fail_ids,
        "correct_ids": correct_ids,
        "fail_feedback": fail_feedback,
        "elapsed_s": elapsed,
    }


def _handle_score_legacy(req: dict, problems: dict) -> dict:
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
    fail_ids, correct_ids, fail_feedback = _parse_results_dict(eval_dict)

    return {
        "fail_ids": fail_ids,
        "correct_ids": correct_ids,
        "fail_feedback": fail_feedback,
        "elapsed_s": elapsed,
    }


def _handle_score(req: dict, problems: dict) -> dict:
    """Route to the persistent-pool path or the legacy `evaluate()` path
    based on PDB_PERSISTENT_POOL.
    """
    if _USE_PERSISTENT_POOL:
        return _handle_score_pool(req)
    return _handle_score_legacy(req, problems)


def main() -> int:
    # Pre-load the BCB problem set once. This is the ~30 s cost we are
    # amortising — paid here at worker startup, free for every score call
    # afterwards. Wrap in _stdout_to_stderr so any chatty load print/tqdm
    # can't leak into the JSON response stream.
    print(f"[bcb_worker] loading subset={_SUBSET} "
          f"(persistent_pool={_USE_PERSISTENT_POOL})",
          file=sys.stderr, flush=True)
    t0 = time.monotonic()
    with _stdout_to_stderr():
        problems = get_bigcodebench(subset=_SUBSET)
    print(f"[bcb_worker] loaded {len(problems)} problems in {time.monotonic() - t0:.1f}s",
          file=sys.stderr, flush=True)

    # Eagerly build the persistent pool BEFORE emitting READY, so the
    # one-time pool-spawn cost is folded into cold start instead of
    # showing up on the first flush. Pool children fork from the master
    # at this point — they inherit the loaded `problems` dict via COW
    # but never touch it (each request loads its own problems_subset
    # from gt_file), so the pages stay shared.
    if _USE_PERSISTENT_POOL:
        _ensure_pool()

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
                _shutdown_pool()
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
