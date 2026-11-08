#!/bin/bash
# Multi-line bug generation with 3 models on both datasets IN PARALLEL, then merge.
#
# Generators: openai/gpt-5.1-codex, gemini/gemini-2.5-pro, anthropic/claude-sonnet-4-5-20250929
# Each generator reads its own long_{gpt|gemini|claude}.json split.
# Settings: mode=multi, stride=4, max_bugs=3, bug_per_time=20, max_gen_per_bin=5,
#           temperature=1.0, max_tokens=32000.
#
# NOTE: [design thought] All 6 (model, dataset) jobs are independent, so they
# fan out concurrently; merges are serial per-dataset. Logs land in bash_log/.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:$PYTHONPATH"
export TRANSFORMERS_VERBOSITY=error    # silence bcb sandbox's PyTorch-not-found banner
PYTHON="$REPO_ROOT/.venv/bin/python"
LOG_DIR="$REPO_ROOT/bash_log"
mkdir -p "$LOG_DIR"
RUN_TAG="multi_$(date +%m%d-%H%M)"

# --- Trap and zombie cleanup ---
kill_zombies() {
  USER_NAME="$(id -un)"
  for pattern in \
    "dataset/bigcodebench/install/.venv/bin/python.*bigcodebench\.evaluate" \
    "dataset/livecodebench/install/.venv/bin/python.*lcb_runner"; do
    if pgrep -u "$USER_NAME" -af "$pattern" > /dev/null 2>&1; then
      echo "Killing zombie processes matching: $pattern"
      pkill -u "$USER_NAME" -f "$pattern" || true
    fi
  done
}

cleanup() {
  echo ""
  echo "Caught interrupt, cleaning up..."
  local pids
  pids=$(jobs -p 2>/dev/null)
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null || true
  fi
  kill_zombies
  exit 1
}
trap cleanup INT TERM

# --- Generator models, prefixes, and per-model input splits ---
MODELS=("openai/gpt-5.1-codex" "gemini/gemini-2.5-pro" "anthropic/claude-sonnet-4-5-20250929")
PREFIXES=("oai" "gg" "ar")
INPUTS=("long_gpt.json" "long_gemini.json" "long_claude.json")
DATASETS=("bigcodebench" "livecodebench")

# --- Preflight: verify each API key is live + has credit with a tiny call ---
echo "=== API preflight ==="
$PYTHON -c "
import sys, dspy
sys.path.insert(0, 'src')
from api_config import resolve_api_key
# NOTE: [design thought] Each provider's API key gates all their models, so we
# preflight with a cheap non-thinking model per provider to keep the check fast
# and avoid gpt-5's temp/min-token constraints.
probes = [
    ('openai/gpt-5.1-codex',                    'openai/gpt-4o-mini'),
    ('gemini/gemini-2.5-pro',                   'gemini/gemini-2.5-flash'),
    ('anthropic/claude-sonnet-4-5-20250929',    'anthropic/claude-haiku-4-5'),
]
for real_model, probe in probes:
    try:
        lm = dspy.LM(probe, api_key=resolve_api_key(real_model),
                     max_tokens=5, temperature=0, num_retries=0)
        lm('ping')
        print(f'[ok] {real_model}  (probed via {probe})')
    except Exception as e:
        print(f'[FAIL] {real_model}: {e}')
        sys.exit(1)
" || { echo 'API preflight failed. Aborting.'; exit 1; }
echo ""

# --- Fan-out: all (model, dataset) jobs in parallel ---
echo "=== Launching 6 parallel multi-line generation jobs ==="
declare -a PIDS=()
for dataset in "${DATASETS[@]}"; do
  for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    prefix="${PREFIXES[$i]}"
    input="${INPUTS[$i]}"
    log="$LOG_DIR/gen_${RUN_TAG}_${prefix}_${dataset}.log"
    echo "  -> $model on $dataset  (input: $input, log: $log)"
    (
      $PYTHON src/bug_generation.py \
        --dataset_name "$dataset" \
        --model_name "$model" \
        --input_file "$input" \
        --output_prefix "${prefix}_buggy_multi" \
        --mode multi \
        --stride 4 \
        --max_bugs 3 \
        --bug_per_time 20 \
        --max_gen_per_bin 5 \
        --temperature 1.0 \
        --max_tokens 32000
    ) > "$log" 2>&1 &
    PIDS+=($!)
  done
done

echo ""
echo "=== Waiting for ${#PIDS[@]} jobs ==="
fail=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "  job pid=$pid failed"
    fail=1
  fi
done
if [[ $fail -ne 0 ]]; then
  echo "ERROR: one or more generation jobs failed. Check $LOG_DIR."
  kill_zombies
  exit 1
fi
echo "All generation jobs done."

# --- Merge each dataset (serial) ---
# NOTE: [design thought] A dedicated output_prefix ("*_buggy_multi_*") keeps
# this merge from picking up single-line artifacts ("*_buggy_code_*") in the
# same directory.
for dataset in "${DATASETS[@]}"; do
  echo ""
  echo "--- Merging $dataset ---"
  GEN_FILES=$(ls "results/$dataset/bug_data/"*_buggy_multi_*.json 2>/dev/null | xargs -n1 basename)
  if [[ -z "$GEN_FILES" ]]; then
    echo "ERROR: No generated multi-line files found for $dataset"
    exit 1
  fi
  echo "Files to merge: $GEN_FILES"

  $PYTHON src/merge.py \
    --input_dir "results/$dataset/bug_data" \
    --in_files $GEN_FILES \
    --output_file "results/$dataset/bug_data/${dataset}_pdb_multi.json" \
    --samples_per_group 5 \
    --max_lines_per_block 4 \
    --min_lines_per_block 2 \
    --stride 4

  echo ""
  echo "=== Merged output for $dataset ==="
  $PYTHON -c "
import json
from collections import Counter
data = json.load(open('results/$dataset/bug_data/${dataset}_pdb_multi.json'))
print(f'Total entries: {len(data)}')
print('Bug counts:', dict(sorted(Counter(d[\"bug_count\"] for d in data).items())))
print('Models:', dict(sorted(Counter(d.get(\"source_model\",\"unknown\") for d in data).items())))
"
done

# --- Final zombie sweep ---
kill_zombies

echo ""
echo "=== Multi-line Generation + Merge Complete ==="
