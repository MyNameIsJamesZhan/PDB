#!/usr/bin/env python
"""Persistent LiveCodeBench evaluator worker.

A long-lived process that loads the LCB benchmark and heavy imports once,
then answers `score` requests on stdin/stdout, avoiding the per-call start-up
cost of running the evaluator as a fresh subprocess.

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

The manager writes verify_file and map_file with
LiveCodeBenchHandler.build_verify_unit_test and sends their paths.
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import hashlib
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path


@contextlib.contextmanager
def _stdout_to_stderr():
    """Send stdout (including C-level writes on fd 1) to stderr for the block, so
    evaluator progress output cannot corrupt the JSON response stream."""
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

# Heavy imports happen once at start-up.
from lcb_runner.runner.scenario_router import build_prompt_benchmark
from lcb_runner.utils.scenarios import Scenario

# The persistent-pool path calls the per-task function directly instead of
# run_custom_evaluator, which spawns its own pool.
from lcb_runner.evaluation.compute_code_generation_metrics import (
    evaluate_generations_by_problem,
)
from lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results

# PDB_PERSISTENT_POOL!=1 uses run_custom_evaluator, which builds a fresh pool
# per request.
_USE_PERSISTENT_POOL = os.environ.get("PDB_PERSISTENT_POOL", "0") == "1"
if not _USE_PERSISTENT_POOL:
    from lcb_runner.runner.custom_evaluator import run as run_custom_evaluator  # noqa: E402

# Persistent process pool (PDB_PERSISTENT_POOL=1); built in main().
_POOL: ProcessPoolExecutor | None = None
# Python 3.10 has no `max_tasks_per_child`, so the whole pool is recycled every
# N flushes to limit heap fragmentation.
_POOL_FLUSH_COUNTER = 0
_POOL_RECYCLE_EVERY = int(os.environ.get("PDB_LCB_POOL_RECYCLE_EVERY", "50"))
# Per-candidate timeout (seconds).
_DEFAULT_TIMEOUT = int(os.environ.get("PDB_LCB_TIMEOUT", "3"))
# Stall watchdog: if no candidate finishes within this many seconds while work is
# pending, the pool is treated as wedged (a sandbox process deadlocked outside its
# per-task timeout) and recycled instead of blocking indefinitely.
_POOL_STALL_S = float(os.environ.get("PDB_LCB_POOL_STALL_S", "90"))

# Step-scoped execution cache: (evaluation sample, code) determines the outcome,
# and RL rollouts often repeat solutions within a step, so duplicates are served
# from cache. Cleared when the request's `step` changes; results fabricated after
# a stall are never cached. PDB_CACHE_TIMEOUTS=0 also skips caching timeouts,
# which can flip under load.
_CACHE_ON = os.environ.get("PDB_EXEC_CACHE", "1") == "1"
_CACHE_TIMEOUTS = os.environ.get("PDB_CACHE_TIMEOUTS", "1") == "1"
_EXEC_CACHE: dict[tuple[str, str], tuple] = {}
_CACHE_STEP: object = None


def _is_timeout_result(meta) -> bool:
    try:
        return "Time Limit Exceeded" in json.dumps(meta)
    except Exception:
        return False


def _pool_child_init() -> None:
    # Redirect fd 1 to fd 2 so descendants can't corrupt the JSON IPC stream.
    os.dup2(2, 1)


def _make_pool() -> ProcessPoolExecutor:
    # Default 8: the pool stays alive for the whole run, so keep it modest.
    n = int(os.environ.get("PDB_LCB_POOL_SIZE", "8"))
    print(f"[lcb_worker] spawning persistent ProcessPoolExecutor max_workers={n}",
          file=sys.stderr, flush=True)
    t0 = time.monotonic()
    pool = ProcessPoolExecutor(max_workers=n, initializer=_pool_child_init)
    # Workers fork lazily on first submit; warm them up now (2*n no-op tasks so
    # every slot is likely touched) so the first request doesn't pay for it.
    warm_futures = [pool.submit(os.getpid) for _ in range(n * 2)]
    warm_pids = {f.result() for f in warm_futures}
    print(f"[lcb_worker] pool warmed in {time.monotonic() - t0:.1f}s "
          f"(forked {len(warm_pids)} workers)",
          file=sys.stderr, flush=True)
    return pool


def _ensure_pool() -> ProcessPoolExecutor:
    """Lazy-construct the pool and recycle every _POOL_RECYCLE_EVERY
    flushes. Recycling caps long-run RSS drift from Python heap
    fragmentation.
    """
    global _POOL, _POOL_FLUSH_COUNTER
    if _POOL is None:
        _POOL = _make_pool()
    elif _POOL_FLUSH_COUNTER >= _POOL_RECYCLE_EVERY:
        print(f"[lcb_worker] recycling pool after {_POOL_FLUSH_COUNTER} flushes",
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
        timeout=_DEFAULT_TIMEOUT,
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


def _handle_score_pool(req: dict, benchmark: list) -> dict:
    """Persistent-pool path: dispatch each candidate as its own future
    on a long-lived ProcessPoolExecutor. Eliminates the per-flush pool
    spawn that `evaluate_generations()` does inside its
    `with ProcessPoolExecutor` block.

    Per-task isolation is unchanged: the per-task function
    `evaluate_generations_by_problem` still spawns a fresh
    `multiprocessing.Process` per candidate (via `check_correctness` in
    compute_code_generation_metrics.py) for sandboxing.

    Reproduces the same `*_output_eval.json` shape that
    `run_custom_evaluator` writes — `[metrics, results_dict,
    final_metadata]` — so `_parse_eval` is unchanged.
    """
    verify_file = Path(req["verify_file"])
    map_file = Path(req.get("map_file", verify_file.with_name(verify_file.stem + "_map.json")))

    if not verify_file.exists():
        raise FileNotFoundError(f"verify_file does not exist: {verify_file}")

    global _CACHE_STEP
    step = req.get("step")
    if _CACHE_ON and step != _CACHE_STEP:
        _EXEC_CACHE.clear()
        _CACHE_STEP = step

    with open(verify_file) as f:
        verify_input = json.load(f)
    ordered_qids = [d.get("question_id") for d in verify_input]
    qid_to_full_ids = {}
    if map_file.exists():
        with open(map_file) as f:
            qid_to_full_ids = json.load(f)

    if not ordered_qids:
        return {"fail_ids": [], "correct_ids": [], "fail_feedback": [], "elapsed_s": 0.0}

    # Pair candidates with benchmark instances, sorted by question_id: `_parse_eval`
    # indexes results by position in that order, as upstream does.
    out_by_id = {str(d["question_id"]): d["code_list"] for d in verify_input}
    benchmark_by_id = {str(inst.question_id): inst for inst in benchmark}
    common_qids = sorted(
        qid for qid in (str(q) for q in ordered_qids)
        if qid in out_by_id and qid in benchmark_by_id
    )
    if not common_qids:
        # Nothing to score: still write an (empty) eval file for _parse_eval.
        eval_path = verify_file.with_name(verify_file.stem + "_output_eval.json")
        with open(eval_path, "w") as f:
            json.dump([{}, {}, []], f)
        return {"fail_ids": [], "correct_ids": [], "fail_feedback": [], "elapsed_s": 0.0}

    t_setup_start = time.monotonic()
    pool = _ensure_pool()
    t_pool = time.monotonic()

    # One future per (problem index, candidate), linearized as codegen_metrics does
    # so aggregation matches upstream.
    samples_linear = []
    generations_linear = []
    remap_index = []  # future_idx -> sorted_problem_idx
    generations_list_per_problem = []  # for the metadata length assertion
    problem_hashes = []  # sorted_problem_idx -> sha256 of the evaluation sample
    for sorted_idx, qid in enumerate(common_qids):
        instance = benchmark_by_id[qid]
        code_list = out_by_id[qid]
        generations_list_per_problem.append(code_list)
        with _stdout_to_stderr():
            sample = instance.get_evaluation_sample()
        if _CACHE_ON:
            problem_hashes.append(hashlib.sha256(
                json.dumps(sample, sort_keys=True, default=str).encode()
            ).hexdigest())
        for code in code_list:
            samples_linear.append(sample)
            generations_linear.append([code])
            remap_index.append(sorted_idx)
    t_build_samples = time.monotonic()

    t0 = time.monotonic()

    # evaluate_generations_by_problem takes args = (generations, sample, debug, timeout).
    futures = {}
    results_linear: dict[int, list] = {}
    metadatas_linear: dict[int, list] = {}
    # Duplicates within one request share the first submission's future (the cache
    # alone would miss them, since they arrive before the first one finishes).
    pending_by_key: dict[tuple[str, str], list[int]] = {}
    cache_hits = 0
    coalesced = 0
    for i in range(len(samples_linear)):
        cache_key = None
        if _CACHE_ON:
            cache_key = (
                problem_hashes[remap_index[i]],
                hashlib.sha256(generations_linear[i][0].encode()).hexdigest(),
            )
            cached = _EXEC_CACHE.get(cache_key)
            if cached is not None:
                results_linear[i], metadatas_linear[i] = cached
                cache_hits += 1
                continue
            shadows = pending_by_key.get(cache_key)
            if shadows is not None:
                shadows.append(i)
                coalesced += 1
                continue
            pending_by_key[cache_key] = []
        fut = pool.submit(
            evaluate_generations_by_problem,
            (generations_linear[i], samples_linear[i], False, _DEFAULT_TIMEOUT),
        )
        futures[fut] = (i, cache_key)
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
            i, key = futures[fut]
            res, meta = fut.result()
            results_linear[i] = res
            metadatas_linear[i] = meta
            if key is not None:
                if _CACHE_TIMEOUTS or not _is_timeout_result(meta):
                    _EXEC_CACHE[key] = (res, meta)
                for j in pending_by_key.get(key, ()):
                    results_linear[j] = res
                    metadatas_linear[j] = meta
    if wedged:
        print(f"[lcb_worker] POOL STALL: no progress for {_POOL_STALL_S:.0f}s "
              f"with {len(pending)}/{len(futures)} futures pending — failing "
              f"stragglers and recycling pool", file=sys.stderr, flush=True)
        for fut in pending:
            i, key = futures[fut]
            fut.cancel()
            results_linear[i] = [[-1]]  # error sentinel -> _parse_eval scores fail
            metadatas_linear[i] = [{"error": "pool_stall_timeout"}]
            if key is not None:
                for j in pending_by_key.get(key, ()):
                    results_linear[j] = [[-1]]
                    metadatas_linear[j] = [{"error": "pool_stall_timeout"}]
        _force_recycle_pool()
    t_wait = time.monotonic()

    elapsed = time.monotonic() - t0

    # Re-aggregate per problem exactly as codegen_metrics does (each future
    # contributed one element).
    results: dict[int, list] = defaultdict(list)
    metadatas: dict[int, list] = defaultdict(list)
    for i in sorted(results_linear):
        results[remap_index[i]].append(results_linear[i][0])
        metadatas[remap_index[i]].append(metadatas_linear[i][0])

    metrics_dict = compute_metrics_from_results(dict(results), k_list=[1])
    t_metrics = time.monotonic()

    final_metadata: list[list[str]] = []
    for key in sorted(metadatas.keys()):
        final_metadata.append(metadatas[key])
    for i in range(len(final_metadata)):
        if not isinstance(final_metadata[i], list):
            final_metadata[i] = [json.dumps(final_metadata[i])]
        else:
            final_metadata[i] = [json.dumps(x) for x in final_metadata[i]]
        # Same invariant codegen_metrics asserts.
        assert len(final_metadata[i]) == len(generations_list_per_problem[i]), (
            f"{len(final_metadata[i])=} vs expected "
            f"{len(generations_list_per_problem[i])} for sorted_idx {i}"
        )

    eval_data = [metrics_dict, dict(results), final_metadata]
    eval_path = verify_file.with_name(verify_file.stem + "_output_eval.json")
    with open(eval_path, "w") as f:
        json.dump(eval_data, f, indent=4)
    t_write = time.monotonic()

    print(f"[lcb_worker] flush phases n_problems={len(common_qids)} "
          f"n_futures={len(samples_linear)}: "
          f"pool={t_pool - t_setup_start:.2f}s "
          f"build_samples={t_build_samples - t_pool:.2f}s "
          f"submit={t_submit - t0:.2f}s "
          f"wait={t_wait - t_submit:.2f}s "
          f"metrics={t_metrics - t_wait:.2f}s "
          f"write={t_write - t_metrics:.2f}s "
          f"total={t_write - t_setup_start:.2f}s "
          f"cache step={step} hits={cache_hits} coalesced={coalesced} "
          f"misses={len(futures)} size={len(_EXEC_CACHE)}",
          file=sys.stderr, flush=True)

    global _POOL_FLUSH_COUNTER
    _POOL_FLUSH_COUNTER += 1

    fail_ids, correct_ids = _parse_eval(eval_path, ordered_qids, qid_to_full_ids)

    return {
        "fail_ids": fail_ids,
        "correct_ids": correct_ids,
        "fail_feedback": [""] * len(fail_ids),
        "elapsed_s": elapsed,
    }


def _handle_score_legacy(req: dict, benchmark: list) -> dict:
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


def _handle_score(req: dict, benchmark: list) -> dict:
    """Route to persistent-pool path or legacy run_custom_evaluator path
    based on PDB_PERSISTENT_POOL.
    """
    if _USE_PERSISTENT_POOL:
        return _handle_score_pool(req, benchmark)
    return _handle_score_legacy(req, benchmark)


def main() -> int:
    # lcb_runner resolves some paths relative to the install dir.
    install_dir = Path(__file__).resolve().parent
    os.chdir(install_dir)

    print(f"[lcb_worker] loading benchmark "
          f"(persistent_pool={_USE_PERSISTENT_POOL})",
          file=sys.stderr, flush=True)
    t0 = time.monotonic()
    with _stdout_to_stderr():
        benchmark, _ = build_prompt_benchmark(_make_args())
    print(f"[lcb_worker] loaded {len(benchmark)} problems in {time.monotonic() - t0:.1f}s",
          file=sys.stderr, flush=True)

    # Build the pool before emitting READY so the first request doesn't pay for it.
    # Children inherit `benchmark` copy-on-write but never touch it, so pages stay shared.
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
                result = _handle_score(req, benchmark)
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
            print(f"[lcb_worker] req {req_id} crashed: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            _emit_error(req_id, exc)

    return 0


if __name__ == "__main__":
    sys.exit(main())
