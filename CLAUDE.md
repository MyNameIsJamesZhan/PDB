# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment setup

PDB uses [uv](https://docs.astral.sh/uv/) and lives in **three independent venvs**: the top-level project venv plus one per dataset sandbox. The dataset sandboxes are vendored, intentionally isolated, and called via subprocess — never activate them in the parent shell.

```bash
uv sync                                                       # top-level: .venv/
cd dataset/bigcodebench/install   && uv sync --extra eval && cd -
cd dataset/livecodebench/install  && uv sync              && cd -
```

The shell drivers under `scripts/` invoke `$REPO_ROOT/.venv/bin/python` directly and set `PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT"`. There is no `pip install -e .`, no `python -m precise_debugging_benchmarking`, and no test suite — `src/` is treated as a flat module namespace via `sys.path.insert` at the top of every entrypoint.

API keys live one-per-file under `keys/` (gitignored). The provider is resolved from the **model-name prefix** (`openai/`, `anthropic/`, `gemini/`, `deepseek/`, `xai/`, `together_ai/`) by `src/api_config.py:resolve_api_key`. Override per-run with `--model_api_file <name>`.

## Common commands

```bash
# End-to-end debug + score for ONE model on both BCB and LCB (parallel datasets):
bash scripts/simple_debug_eval.sh <single|single-hard|multi> <dspy-model-string>

# Loop over the paper's reference-model list (edit THINKING_MODELS / NON_THINKING_MODELS at top of file):
bash scripts/run_debug_eval.sh <single|single-hard|multi>

# Score an EXISTING debug-results file (no model calls):
python src/evaluator.py --dataset_name bigcodebench --eval_model_name <name> \
  --input_file <model>_on_<eval_set>_round_<k>.json \
  --eval_set_name <eval_set> --max_iter 1

# Validate raw GT data against the sandbox before generation (writes <stem>_valid.json):
python src/preprocess.py --dataset_name bigcodebench --input_file claude.json gemini.json gpt.json

# Generate a fresh PDB test set:
bash scripts/run_bug_gen_single.sh    # 3 generators × 2 datasets, then merge -> <bench>_pdb_single.json
bash scripts/run_bug_gen_multi.sh     # multi-line variant -> <bench>_pdb_multi.json
```

There is no lint or test target. The bug-correction pipeline parallelises over tasks within a dataset via `--n_workers <N>`; the dataset sandboxes themselves run serially.

## Architecture

PDB is a **three-stage pipeline** chained through stable JSON contracts on disk; each stage's output is the next stage's input.

```
bug_generation.py  ─►  bug_correct.py  ─►  evaluator.py
   (inject bugs)        (LLM debugger)        (score)
        │                     │                   │
   <bench>_pdb_*.json    <model>_on_<set>_   <model>_on_<set>_
                         round_<k>.json       round_<k>_scores.json
```

The pipeline is **diff-centric**, not patch-centric. Every stage operates on line-keyed block diffs in this exact format: `{line_no: {"type": "Modify"|"Insert"|"Delete", "original": str, "modified": str}}`. Two diff directions are used and easily confused:

- `diff` — **forward** (`gt -> buggy`). Used to *compose* bugs onto a clean solution.
- `gt_diff` / `pred_diff` — **reverse** (`buggy -> gt` / `buggy -> pred`). The target a fix must produce, and what the evaluator scores against.

If you see `file_diff(buggy, fixed)`, the result is reverse; if you see `file_diff(gt, buggy)`, it's forward. The convention is documented in `src/bug_generation.py`'s module docstring and is load-bearing — flipping it silently inverts every metric.

### Stage 1 — bug generation (`src/bug_generation.py`)

Three sub-stages: `bug_generate` (LLM proposes single-block edits, validated atomically), `bug_compose` (combines `k` non-adjacent verified blocks for `k=2..max_bugs` per program), `validate_and_sample` (round-trip check that applying `gt_diff` to `buggy_code` recovers `gt_solution`, then bin-bucketed sampling). Two **modes** thread through everything: `single` (one-line bugs, `stride=2`) and `multi` (`MIN_MULTILINES..MAX_MULTILINES` contiguous lines, `stride=4`). Mode-shared constants live in `src/config.py` — edit there, not in CLI defaults.

### Stage 2 — bug correction (`src/bug_correct.py`)

Round-based debugging loop. Round 1 calls `Debugger` (a dspy module from `src/module.py`) on raw buggy code. Between rounds, the evaluator scores every attempt; tasks that failed have their solution appended to a per-task `failed_attempts` list and have `debug_results` stripped, so the next call re-attempts only those. Starting at round 2 the `debug_mode` is auto-suffixed with `_with_feedback`; `--use_tests` adds `_unit`; `--error_msg` adds the sandbox stdout/stderr. Successful tasks carry over verbatim, so per-round JSON files are non-monotonic in size (round_k for k≥2 only contains tasks still failing).

`--use_claude_code` swaps the dspy-LM Debugger for `src/claude_code_wrapper.py:ClaudeCodeGenerator`, which shells out to the `claude` CLI. It implies `--use_tests`, and `--max_rounds 1` is normal because the agent iterates inside its own loop.

### Stage 3 — evaluator (`src/evaluator.py`)

Two metrics, both keyed by `task_id`:
1. **Unit score** — pass/fail from the dataset sandbox. Always run; the sandbox call is the most expensive part.
2. **Symbolic block scores** — multi-pass alignment of `pred_diff` against `gt_diff` to compute edit-level **precision**, bug-level **recall**, and F1. The `--tolerance N` knob (default 1 for multi, 2 for single — see `DEFAULT_TOLERANCE_*` in `config.py`) gives each matched GT block N free extra predicted lines before docking precision.

Per-(model, dataset, round) summary lines (`[summary] ... unit=... prec=... rec=... f1=... (n=...)`) are emitted by `Evaluator.print_summary()` and aggregated to a `union` line by the driver.

### Dataset handler abstraction (`dataset/`)

Every dataset is a subclass of `DatasetHandler` (in `dataset/base.py`) registered explicitly in `dataset/__init__.py:_REGISTRY`. Five operations: `preprocess`, `mark_editable_lines` (concrete default works for Python with starter-code frozen lines — only override if your rules differ), `build_verify_unit_test`, `verify_unit_test`, `save_formatted_gt`. The handler shells out to its **own** vendored uv venv via `self.venv_cmd("module", *args)` (which builds `[<install>/.venv/bin/python, -m, module, ...]`); the parent shell env is never modified, no activate/deactivate plumbing. `mark_editable_lines` is the source of truth for which lines a bug injector may touch — `NO_CHANGE_KEYWORDS` (def/class/import/try/except/finally/async) are immutable, `NO_DELETE_KEYWORDS` (control-flow) can be edited but not removed. To add a dataset, see `dataset/README.md`.

### File and naming conventions

- `results/<bench>/bug_data/<bench>_pdb_<subset>.json` — final test sets (input to stage 2).
- `results/<bench>/debug_results/<short_model>_on_<eval_set>_round_<k>.json` — stage 2 output.
- `results/<bench>/eval_results/<short_model>_on_<eval_set>_round_<k>_scores.json` — stage 3 output.
- `bash_log/dbg_<run_tag>_<short_model>_<bench>_<subset>.log` — per-subprocess driver logs.
- `<short_model>` is the **basename** after the last `/` of the dspy model string (e.g. `openai/gpt-5.1-codex` → `gpt-5.1-codex`). Drivers and aggregators rely on this convention.

### DSPy modules (`src/module.py`)

`Debugger`, `BugInjector`, `MultilineBugInjector`, `Rewriter` are dspy `Predict`/`ChainOfThought` wrappers around hand-written prompt templates (the templates are inline in the file and intentionally not parameterised — change them in place). `ExternalModelWrapper` adapts arbitrary callable LMs into the dspy interface for the Claude Code path. Prompt-template selection (`minimal` vs `free`, `_with_feedback`, `_unit`) happens at call time inside `bug_correct.py` based on round index plus `--use_tests`/`--error_msg` flags.
