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
import atexit
import contextlib
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
# everything LCB needs, so per-request invocations only do the actual
# evaluation work — no module-load tax.
from lcb_runner.runner.scenario_router import build_prompt_benchmark
from lcb_runner.utils.scenarios import Scenario

# Persistent-pool path uses the lower-level per-task function directly,
# bypassing run_custom_evaluator + codegen_metrics' built-in pool spawn.
from lcb_runner.evaluation.compute_code_generation_metrics import (
    evaluate_generations_by_problem,
)
from lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results

# Legacy path (PDB_PERSISTENT_POOL!=1) still calls run_custom_evaluator
# which builds a fresh multiprocessing.Pool every flush. Kept behind a
# gate so we can A/B test against the new persistent-pool path without
# ripping the old code out.
_USE_PERSISTENT_POOL = os.environ.get("PDB_PERSISTENT_POOL", "0") == "1"
if not _USE_PERSISTENT_POOL:
    from lcb_runner.runner.custom_evaluator import run as run_custom_evaluator  # noqa: E402

# Persistent ProcessPoolExecutor — built once in main() if
# PDB_PERSISTENT_POOL=1. None if gated off.
_POOL: ProcessPoolExecutor | None = None
# Python 3.10 doesn't support `max_tasks_per_child`, so we recycle the
# whole pool every N flushes to mitigate heap fragmentation.
_POOL_FLUSH_COUNTER = 0
_POOL_RECYCLE_EVERY = int(os.environ.get("PDB_LCB_POOL_RECYCLE_EVERY", "50"))
# Per-candidate timeout passed to evaluate_generations_by_problem.
# Overridable via PDB_LCB_TIMEOUT to allow A/B comparison at runtime.
_DEFAULT_TIMEOUT = int(os.environ.get("PDB_LCB_TIMEOUT", "3"))
# Stall watchdog: if no candidate future completes within this many seconds
# while work is still pending, treat the pool as wedged (a sandbox grandchild
# deadlocked outside its per-task timeout) and recycle it instead of blocking
# the training step indefinitely. Healthy flushes complete a future every few
# seconds, so this never trips in normal operation.
_POOL_STALL_S = float(os.environ.get("PDB_LCB_POOL_STALL_S", "90"))


def _pool_child_init() -> None:
    # Redirect fd 1 to fd 2 so descendants can't corrupt the JSON IPC stream.
    os.dup2(2, 1)


def _make_pool() -> ProcessPoolExecutor:
    # Default 8 (not 12 like legacy num_process_evaluate) to reduce
    # sustained scheduler pressure on the 16-CPU SLURM allocation. Today
    # the pool lives only during a flush, but persistent pools are hot
    # the entire training run.
    n = int(os.environ.get("PDB_LCB_POOL_SIZE", "8"))
    print(f"[lcb_worker] spawning persistent ProcessPoolExecutor max_workers={n}",
          file=sys.stderr, flush=True)
    t0 = time.monotonic()
    pool = ProcessPoolExecutor(max_workers=n, initializer=_pool_child_init)
    # ProcessPoolExecutor.__init__ does NOT fork workers — they're spawned
    # lazily on first submit(). Force the fork now by running a no-op
    # future per worker, so the first real flush doesn't pay the fork
    # cost. Submit 2*n so every slot is touched (futures are racy across
    # workers; oversubscribing the warm-up is the simplest reliable way).
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

    with open(verify_file) as f:
        verify_input = json.load(f)
    ordered_qids = [d.get("question_id") for d in verify_input]
    qid_to_full_ids = {}
    if map_file.exists():
        with open(map_file) as f:
            qid_to_full_ids = json.load(f)

    if not ordered_qids:
        return {"fail_ids": [], "correct_ids": [], "fail_feedback": [], "elapsed_s": 0.0}

    # Pair verify_input candidates with benchmark instances. Filter to
    # qids present in both, then sort by question_id to match the
    # convention `_parse_eval` expects (per_index keyed by
    # str(idx_in_sorted_qids)). Upstream's sort_and_extract_save_results
    # does the same sort (scenario_router.py:147).
    out_by_id = {str(d["question_id"]): d["code_list"] for d in verify_input}
    benchmark_by_id = {str(inst.question_id): inst for inst in benchmark}
    common_qids = sorted(
        qid for qid in (str(q) for q in ordered_qids)
        if qid in out_by_id and qid in benchmark_by_id
    )
    if not common_qids:
        # Nothing to score — emit empty eval file so _parse_eval can
        # still parse without raising FileNotFoundError.
        eval_path = verify_file.with_name(verify_file.stem + "_output_eval.json")
        with open(eval_path, "w") as f:
            json.dump([{}, {}, []], f)
        return {"fail_ids": [], "correct_ids": [], "fail_feedback": [], "elapsed_s": 0.0}

    t_setup_start = time.monotonic()
    pool = _ensure_pool()
    t_pool = time.monotonic()

    # Linearize: one future per (problem_idx_in_sorted_order, candidate).
    # Mirrors codegen_metrics' linearization
    # (compute_code_generation_metrics.py:166-179) so the result
    # aggregation is byte-identical.
    samples_linear = []
    generations_linear = []
    remap_index = []  # future_idx -> sorted_problem_idx
    generations_list_per_problem = []  # for the metadata length assertion
    for sorted_idx, qid in enumerate(common_qids):
        instance = benchmark_by_id[qid]
        code_list = out_by_id[qid]
        generations_list_per_problem.append(code_list)
        with _stdout_to_stderr():
            sample = instance.get_evaluation_sample()
        for code in code_list:
            samples_linear.append(sample)
            generations_linear.append([code])
            remap_index.append(sorted_idx)
    t_build_samples = time.monotonic()

    t0 = time.monotonic()

    # Submit all candidates at once. evaluate_generations_by_problem's
    # signature is `args = (generations_list, sample, debug, timeout)`.
    futures = {}
    for i in range(len(samples_linear)):
        fut = pool.submit(
            evaluate_generations_by_problem,
            (generations_linear[i], samples_linear[i], False, _DEFAULT_TIMEOUT),
        )
        futures[fut] = i
    t_submit = time.monotonic()

    results_linear: dict[int, list] = {}
    metadatas_linear: dict[int, list] = {}
    pending = set(futures)
    wedged = False
    while pending:
        done, pending = wait(pending, timeout=_POOL_STALL_S,
                             return_when=FIRST_COMPLETED)
        if not done:
            wedged = True
            break
        for fut in done:
            i = futures[fut]
            res, meta = fut.result()
            results_linear[i] = res
            metadatas_linear[i] = meta
    if wedged:
        print(f"[lcb_worker] POOL STALL: no progress for {_POOL_STALL_S:.0f}s "
              f"with {len(pending)}/{len(futures)} futures pending — failing "
              f"stragglers and recycling pool", file=sys.stderr, flush=True)
        for fut in pending:
            i = futures[fut]
            fut.cancel()
            results_linear[i] = [[-1]]  # error sentinel -> _parse_eval scores fail
            metadatas_linear[i] = [{"error": "pool_stall_timeout"}]
        _force_recycle_pool()
    t_wait = time.monotonic()

    elapsed = time.monotonic() - t0

    # Re-aggregate by sorted_problem_idx — verbatim copy of codegen_metrics
    # lines 191-211. Each linear future contributed one element to both
    # lists (we passed a 1-element generations_list per future).
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
        # Same invariant codegen_metrics asserts. We construct the data,
        # so this should always hold — assert defensively.
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
          f"total={t_write - t_setup_start:.2f}s",
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
    # Mirror the legacy `cwd=self.install_dir` from
    # LiveCodeBenchHandler.verify_unit_test() — some lcb_runner internals may
    # reference paths relative to the install dir (e.g., cached datasets).
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

    # Eagerly build the persistent pool BEFORE emitting READY, so the
    # one-time pool-spawn cost is folded into cold start instead of
    # showing up on the first flush. Pool children fork from the master
    # at this point — they inherit the loaded `benchmark` list via COW
    # but never touch it (each request looks up via benchmark_by_id in
    # the master), so the pages stay shared.
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
