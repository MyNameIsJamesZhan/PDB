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
import hashlib
import json
import multiprocessing
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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
# Stall watchdog: if no candidate future completes within this many seconds
# while work is still pending, treat the pool as wedged (a sandbox grandchild
# deadlocked outside its per-task timeout) and recycle it instead of blocking
# the training step indefinitely. BCB per-task budget is ~20s, so the window is
# wider than LCB's; healthy flushes still complete a future every few seconds.
_POOL_STALL_S = float(os.environ.get("PDB_BCB_POOL_STALL_S", "180"))

# Step-scoped execution cache. GRPO rollouts duplicate ~60% of candidate
# solutions within a training step, and (test, solution) fully determines the
# sandbox outcome, so duplicates are served from cache instead of re-executed.
# Cleared whenever the request's `step` field changes; stall-fabricated
# results never reach the insert path. PDB_CACHE_TIMEOUTS=0 additionally
# skips caching per-candidate timeouts (they can flip under node load).
_CACHE_ON = os.environ.get("PDB_EXEC_CACHE", "1") == "1"
_CACHE_TIMEOUTS = os.environ.get("PDB_CACHE_TIMEOUTS", "1") == "1"
_EXEC_CACHE: dict[tuple[str, str], tuple] = {}
_CACHE_STEP: object = None


# Top-level modules by frequency across the 1140 BCB tasks (code_prompt + test),
# heavy ones only; stdlib is instant. Override with PDB_BCB_PRELOAD_MODULES=a,b,c.
_PRELOAD_MODULES = (
    "numpy", "pandas", "matplotlib", "matplotlib.pyplot", "sklearn",
    "sklearn.linear_model", "sklearn.preprocessing", "sklearn.model_selection",
    "sklearn.cluster", "sklearn.metrics", "scipy", "scipy.stats", "scipy.optimize",
    "seaborn", "statsmodels.api", "statsmodels.formula.api", "requests", "faker",
    "nltk", "pytz", "bs4", "PIL", "dateutil", "cv2", "sympy", "networkx",
)


def _preload_heavy_libs() -> None:
    import importlib
    mods = os.environ.get("PDB_BCB_PRELOAD_MODULES")
    mods = mods.split(",") if mods else _PRELOAD_MODULES
    t0 = time.monotonic()
    loaded, failed = [], []
    for m in mods:
        m = m.strip()
        if not m:
            continue
        try:
            importlib.import_module(m)
            loaded.append(m)
        except Exception as e:  # missing optional dep: the test would fail the same way
            failed.append(f"{m}({type(e).__name__})")
    print(f"[bcb_worker] preloaded {len(loaded)} modules in {time.monotonic() - t0:.1f}s"
          + (f"; skipped {failed}" if failed else ""), file=sys.stderr, flush=True)


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


def _force_recycle_pool() -> None:
    """Drop a wedged pool and SIGKILL its workers so the next flush rebuilds.
    Used when the stall watchdog fires: shutdown(wait=False) won't stop a
    worker stuck on an unkillable sandbox grandchild, so we kill the worker
    procs directly. Stragglers reparent and die on their own timeout — a
    transient leak, not a hang."""
    global _POOL, _POOL_FLUSH_COUNTER
    pool = _POOL
    _POOL = None
    _POOL_FLUSH_COUNTER = 0
    if pool is None:
        return
    procs = list(getattr(pool, "_processes", {}).values())
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass


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


def _parse_results_dict(eval_dict: dict, compact: bool = False) -> tuple[list[str], list[str], list[str]]:
    """Walk the `evaluate()`-shape results dict and split into
    (fail_ids, correct_ids, fail_feedback). Shared by both legacy and
    persistent-pool paths. With compact=True, fail_feedback is a list of
    empty strings — RL reward path never reads details and dumping
    per-test tracebacks can push the JSON response past asyncio's 64 KiB
    StreamReader limit, corrupting the whole flush's reward signal.
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
            fail_feedback.append("" if compact
                                 else json.dumps(perfs[0].get("details", ""), indent=2))
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

    global _CACHE_STEP
    step = req.get("step")
    if _CACHE_ON and step != _CACHE_STEP:
        _EXEC_CACHE.clear()
        _CACHE_STEP = step

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

    # Upstream's default of 30 GB per candidate × 8 pool workers can OOM the
    # node when an adversarial candidate allocates aggressively. Cap at 4 GiB
    # — comfortably above any real BCB task's working set, but bounded.
    max_as_limit = 4 * 1024
    max_data_limit = 4 * 1024
    max_stack_limit = 10
    min_time_limit = 10.0
    # no_gt=True branch in evaluate.py:383 falls back to gt_time_limit=20.
    gt_time_limit = 20.0

    t_read_start = time.monotonic()
    pool = _ensure_pool()
    t_pool = time.monotonic()

    t0 = time.monotonic()

    completion_id: Counter[str] = Counter()
    futures: dict = {}
    eval_results: dict[str, list[dict]] = defaultdict(list)
    test_hash: dict[str, str] = {}
    # In-request duplicates coalesce onto the first submission's future and
    # get the result fanned out on completion (GRPO rollouts of one task all
    # arrive in the same unit-pass request, so cache inserts alone are too
    # late to catch them).
    pending_by_key: dict[tuple[str, str], list[dict]] = {}
    cache_hits = 0
    coalesced = 0
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
            cid = completion_id[tid]
            cache_key = None
            if _CACHE_ON:
                th = test_hash.get(tid)
                if th is None:
                    th = hashlib.sha256(
                        (problems_subset[tid]["test"] + "\x00"
                         + problems_subset[tid]["entry_point"]).encode()
                    ).hexdigest()
                    test_hash[tid] = th
                cache_key = (th, hashlib.sha256(solution.encode()).hexdigest())
                cached = _EXEC_CACHE.get(cache_key)
                if cached is not None:
                    eval_results[tid].append({
                        "completion_id": cid,
                        "task_id": tid,
                        "_identifier": sample["_identifier"],
                        "solution": solution,
                        "base": cached,
                    })
                    completion_id[tid] += 1
                    cache_hits += 1
                    continue
                shadows = pending_by_key.get(cache_key)
                if shadows is not None:
                    shadows.append({
                        "completion_id": cid,
                        "task_id": tid,
                        "_identifier": sample["_identifier"],
                        "solution": solution,
                    })
                    completion_id[tid] += 1
                    coalesced += 1
                    continue
                pending_by_key[cache_key] = []
            args = (
                cid,
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
            fut = pool.submit(_local_check_correctness, *args)
            # Keep enough to fabricate a failure result if this future wedges.
            futures[fut] = {
                "completion_id": cid,
                "task_id": tid,
                "_identifier": sample["_identifier"],
                "solution": solution,
                "cache_key": cache_key,
            }
            completion_id[tid] += 1
    t_submit = time.monotonic()

    pending = set(futures)
    wedged = False
    while pending:
        done, pending = wait(pending, timeout=_POOL_STALL_S,
                             return_when=FIRST_COMPLETED)
        if not done:
            wedged = True
            break
        for fut in done:
            r = fut.result()
            eval_results[r["task_id"]].append(r)
            key = futures[fut]["cache_key"]
            if key is not None:
                if _CACHE_TIMEOUTS or r["base"][0] != "timeout":
                    _EXEC_CACHE[key] = r["base"]
                for shadow in pending_by_key.get(key, ()):
                    eval_results[shadow["task_id"]].append({**shadow, "base": r["base"]})
    if wedged:
        print(f"[bcb_worker] POOL STALL: no progress for {_POOL_STALL_S:.0f}s "
              f"with {len(pending)}/{len(futures)} futures pending — failing "
              f"stragglers and recycling pool", file=sys.stderr, flush=True)
        for fut in pending:
            info = futures[fut]
            fut.cancel()
            eval_results[info["task_id"]].append({
                "completion_id": info["completion_id"],
                "task_id": info["task_id"],
                "_identifier": info["_identifier"],
                "solution": info["solution"],
                "base": ("fail", "pool_stall_timeout"),
            })
            if info["cache_key"] is not None:
                for shadow in pending_by_key.get(info["cache_key"], ()):
                    eval_results[shadow["task_id"]].append(
                        {**shadow, "base": ("fail", "pool_stall_timeout")})
        _force_recycle_pool()
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
          f"total={elapsed:.2f}s "
          f"cache step={step} hits={cache_hits} coalesced={coalesced} "
          f"misses={len(futures)} size={len(_EXEC_CACHE)}",
          file=sys.stderr, flush=True)

    global _POOL_FLUSH_COUNTER
    _POOL_FLUSH_COUNTER += 1

    fail_ids, correct_ids, fail_feedback = _parse_results_dict(
        eval_dict, compact=req.get("compact_feedback", False))
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
    fail_ids, correct_ids, fail_feedback = _parse_results_dict(
        eval_dict, compact=req.get("compact_feedback", False))

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
    # Pre-import the heavy libraries BCB tasks use before forking the pool, so
    # every forked test process inherits them through sys.modules instead of
    # re-importing from disk. On a cold shared filesystem (Anvil GPFS) a fresh
    # `import sklearn/scipy/statsmodels/matplotlib` costs 20-90 s -- more than
    # the 20 s per-candidate budget -- and correct solutions score `timeout`
    # (measured: 11/11 val GT timed out on a cold node, 5/11 still warm; the
    # tests themselves run in ~3 s). Module state is copied at fork, so each
    # test still starts from the same pristine snapshot. PDB_BCB_PRELOAD=0 opts out.
    if os.environ.get("PDB_BCB_PRELOAD", "1") == "1":
        _preload_heavy_libs()
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
