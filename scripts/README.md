# `scripts/` — Shell drivers

Thin bash wrappers over the Python entrypoints in [../src/](../src/). They exist so the common fan-out patterns are one command instead of a dozen flags. Every driver exposes its tunables at the top of the file under a `# --- Run-wide knobs (edit here to change everything) ---` block.

---

# 1. Debug & evaluation

## Drivers

| Script | Purpose |
|---|---|
| [`simple_debug_eval.sh`](simple_debug_eval.sh) | Bug-correct + score **one model** on BigCodeBench and LiveCodeBench for a chosen subset; runs the two datasets in parallel. |
| [`run_debug_eval.sh`](run_debug_eval.sh) | Loops over a list of reference models (edit `THINKING_MODELS` / `NON_THINKING_MODELS` at the top of the script); serial across models, parallel across datasets per model. |

## How to run

```bash
# One model
bash scripts/simple_debug_eval.sh <subset> <model>
#   <subset> ∈ {single, single-hard, multi}
#   <model>  = any dspy model name, e.g. openai/gpt-5.1-codex

# All reference models
bash scripts/run_debug_eval.sh <subset>
```

Inputs are auto-resolved: `results/<bench>/bug_data/<bench>_pdb_<subset>.json`. Outputs (debug results + scores + logs) are described below.

## Running a local / self-hosted model

DSPy accepts any OpenAI-compatible endpoint (vLLM, SGLang, Ollama, LM Studio). Point it at your server with env vars and use the `openai/` prefix on the model name:

```bash
vllm serve Qwen/Qwen3-Coder-7B --port 8000
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY=dummy                      # litellm requires any value
bash scripts/simple_debug_eval.sh single openai/Qwen/Qwen3-Coder-7B
```

For Ollama specifically, swap the prefix: `ollama_chat/qwen2.5-coder:7b` (default base `http://localhost:11434`; override with `OLLAMA_API_BASE`).

## Global parameters

| Var | Default | Meaning |
|---|---|---|
| `DEBUG_MODE` | `minimal` | Instruction template for the Debugger. `minimal` = "fix only the bugs, do not reformat" (paper setting); `free` = unconstrained "debug this code" (over-edit baseline). Automatic `*_with_feedback` / `*_unit` suffixes are appended per round when `--error_msg` / `--use_tests` are set. |
| `MAX_ROUNDS` | `1` | Iterative debugging depth. Paper reports **1 to 3** rounds: each subsequent round sees the prior failed patches (and, if `--use_tests`/`--error_msg`, the sandbox's stdout/stderr) as `failed_attempts`. |
| `TEMPERATURE` | `1.0` | Sampling temperature. Paper uses `1.0` throughout. GPT-5 models reject anything but 1.0. |
| `MAX_TOKENS` | `32000` | Max output tokens. |
| `N_WORKERS` | `1` | Threads inside `bug_correct.py` for the Debugger fix-loop. Each dataset is already its own process in the driver; this further parallelizes over tasks within a dataset. Evaluation stays serial. |

### Choosing values

- **`DEBUG_MODE`** — use `minimal` to reproduce paper numbers. `free` only if you're measuring the over-edit gap. For anything else (`minimal_with_feedback`, `free_unit_with_feedback`, …) — these are auto-assigned from round index + flags; to force a specific template, copy the script and hard-code `--debug_mode`.
- **`MAX_ROUNDS`** — 1 for a single-pass eval (fastest, cheapest). 2-3 for the iterative/agentic ablation. Rounds 2+ re-run only the tasks that failed in round 1 (via the `--reload_first_round` path when you wire it through).
- **`N_WORKERS`** — raise for big runs if your API rate limit allows. Each worker issues one concurrent request. Evaluation parallelism is fixed by the sandbox.

## Output files

| Path | Contents |
|---|---|
| `results/<bench>/debug_results/<model>_on_<bench>_<subset>_round_<k>.json` | List of entries: `task_id`, `buggy_code`, `gt_solution`, `task_prompt`, `gt_diff`, `bug_count`, … plus a new `debug_results` block with `model`, `solution` (the patch), `pred_diff`. |
| `results/<bench>/eval_results/<model>_on_<bench>_<subset>_round_<k>_scores.json` | Top-level `Unit score` (`{task_id: float}` — pass@1 from the sandbox) and `Symbolic block scores` (`{task_id: {precision, recall, f1, matched_blocks, unmatched_pred, unmatched_gt}}`). |
| `bash_log/dbg_<tag>_<model>_<bench>_<subset>.log` | stdout/stderr of the corresponding subprocess. |

Each round re-evaluates only the tasks that were still wrong at the end of the previous round, so `round_1.json` covers every task and `round_k.json` (k≥2) only covers the ones that needed another try. Scores files still aggregate over the full task set.

Example `debug_results` block:

```json
{
  "model": "openai/gpt-5.1-codex",
  "solution": "def task_func(...):\n    ...\n",
  "pred_diff": {"11": {"type": "Modify", "original": "return x", "modified": "return x + 1"}}
}
```

Example `Symbolic block scores[task_id]`:

```json
{
  "precision": 1.0,
  "recall": 1.0,
  "f1": 1.0,
  "matched_blocks": {"BigCodeBench/0_0_em_0": {"block_start": 11, "block_end": 11, "diff": {...}}},
  "unmatched_pred": {},
  "unmatched_gt": {}
}
```

## Summary format

`Evaluator.print_summary()` emits one `[summary]` line per (model, dataset, round). Drivers aggregate only the **final round** across both datasets:

```
[summary] gpt-5.1-codex on bigcodebench_pdb_single round 1: unit=0.733 prec=0.548 rec=0.777 f1=0.602 (n=3697)
[summary] gpt-5.1-codex on livecodebench_pdb_single round 1: unit=0.914 prec=0.465 rec=0.789 f1=0.540 (n=3892)
  union  unit=0.826 prec=0.505 rec=0.783 f1=0.570 (n=7589)
```

Per-dataset lines land in `bash_log/dbg_*.log` when datasets run in parallel subprocesses; the `union` line goes to the driver's stdout.

---

# 2. Bug generation

## Drivers

| Script | Purpose |
|---|---|
| [`run_bug_gen_single.sh`](run_bug_gen_single.sh) | Generate + merge a **single-line** PDB dataset (`<bench>_pdb_single.json`) from 3 generators × 2 datasets. Preflights API keys first. |
| [`run_bug_gen_multi.sh`](run_bug_gen_multi.sh) | Same flow for **multi-line** bugs (`<bench>_pdb_multi.json`), reading per-generator `long_<name>.json` splits. |

## How to generate your own data

1. Drop a source dataset file at `dataset/<your-bench>/data/full_data.json` and register a handler — see [../dataset/README.md](../dataset/README.md).

2. **Validate the source data first.** Run [`src/preprocess.py`](../src/preprocess.py) on every input file to drop tasks whose ground-truth solution fails the sandbox's own unit tests or has too few editable lines. The valid subset is written back as `<stem>_valid.json` next to the original and is what the bug-generation step should consume.

   ```bash
   python src/preprocess.py --dataset_name bigcodebench \
     --input_file claude.json gemini.json gpt.json
   ```

   Each input gets its own filter pass. The script prints the INVALID `task_id`s for each file and a final `=== summary ===` table; Pass `--skip_verify` to skip the sandbox call and only apply the editable-line filter.

   Use your `<stem>_valid.json` (e.g., `claude_valid.json`) as the validated data for bug generation.

3. Pick your generator pool by editing the `MODELS` + `PREFIXES` arrays at the top of the driver. Anything supported by LiteLLM works.

4. Tune the generation hyperparameters (also near the top of the driver) — the ones that matter:

| Flag on `bug_generation.py` | Default | Meaning |
|---|---|---|
| `--mode` | `single` | `single` (1-line bugs) or `multi` (2-4 line contiguous blocks). |
| `--stride` | `2` / `4` | Minimum inter-block line gap during composition (single / multi). |
| `--max_bugs` | `4` | Max bugs composed per program (`k_max`). |
| `--bug_per_time` | `20` | Atomic-bug drafting attempts per task (`m₁`). |
| `--max_gen_per_bin` | `5` | Subsampling cap per `(task, bug_count)` bin (`m₃`). |
| `--max_lines_per_block` | `1` / `4` | Block-size cap for diff validation (single / multi). |

5. Run the driver. Each generator dumps a per-model `<prefix>_buggy_<mode>_<timestamp>.json`, then the script merges those into `<bench>_pdb_<mode>.json`:

```bash
bash scripts/run_bug_gen_single.sh   # produces <bench>_pdb_single.json
bash scripts/run_bug_gen_multi.sh    # produces <bench>_pdb_multi.json
```

Both drivers run all 6 (model × dataset) jobs concurrently and do a cheap API preflight first so you don't discover a dead key halfway through.
