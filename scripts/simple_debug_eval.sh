#!/bin/bash
# Bug correction + evaluation for ONE model across both datasets.
# All models use max_tokens=32000.
#
# Usage:  bash scripts/simple_debug_eval.sh <subset> <model>
#   <subset>  ∈ {single, single-hard, multi}
#   <model>   full dspy model name, e.g. "openai/gpt-5.1-codex"
#
# Examples:
#   bash scripts/simple_debug_eval.sh single openai/gpt-5.1-codex
#   bash scripts/simple_debug_eval.sh multi gemini/gemini-2.5-pro
#   bash scripts/simple_debug_eval.sh single-hard deepseek/deepseek-chat
set -e

SUBSET="${1:?missing subset (single | single-hard | multi)}"
MODEL="${2:?missing model name}"

# --- Run-wide knobs (edit here to change everything) ---
DEBUG_MODE=minimal
MAX_ROUNDS=1
TEMPERATURE=1.0
MAX_TOKENS=32000
N_WORKERS=4

case "$SUBSET" in
  single)      FILE_TAG="pdb_single" ;;
  single-hard) FILE_TAG="pdb_single_hard" ;;
  multi)       FILE_TAG="pdb_multi" ;;
  *)
    echo "ERROR: unknown subset '$SUBSET' (expected: single | single-hard | multi)"
    exit 2
    ;;
esac

echo "Subset: $SUBSET    (file tag: $FILE_TAG)"
echo "Model:  $MODEL"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT:$PYTHONPATH"
export TRANSFORMERS_VERBOSITY=error    # silence bcb sandbox's PyTorch-not-found banner
PYTHON="$REPO_ROOT/.venv/bin/python"

# --- Trap: on interrupt, kill our backgrounded jobs ---
cleanup() {
  echo ""
  echo "Caught interrupt, killing background jobs..."
  local pids
  pids=$(jobs -p 2>/dev/null)
  if [[ -n "$pids" ]]; then
    kill $pids 2>/dev/null || true
  fi
  exit 1
}
trap cleanup INT TERM

# --- Multi subset requires --mode multi so tolerance defaults to 1 ---
MODE_ARGS=()
if [[ "$SUBSET" == "multi" ]]; then
  MODE_ARGS=(--mode multi)
fi

# --- Sanity-check inputs first ---
DATASETS=("bigcodebench" "livecodebench")
for dataset in "${DATASETS[@]}"; do
  input="${dataset}_${FILE_TAG}.json"
  if [[ ! -f "results/$dataset/bug_data/$input" ]]; then
    echo "ERROR: missing input file results/$dataset/bug_data/$input"
    exit 1
  fi
done

# --- Launch both datasets in parallel ---
LOG_DIR="$REPO_ROOT/bash_log"
mkdir -p "$LOG_DIR"
SHORT_MODEL="${MODEL##*/}"
RUN_TAG="$(date +%m%d-%H%M)"

declare -a PIDS=()
for dataset in "${DATASETS[@]}"; do
  input="${dataset}_${FILE_TAG}.json"
  eval_set="${dataset}_${FILE_TAG}"
  log="$LOG_DIR/dbg_${RUN_TAG}_${SHORT_MODEL}_${dataset}_${FILE_TAG}.log"
  echo "  -> $MODEL on $dataset  (log: $log)"
  (
    $PYTHON src/bug_correct.py \
      --dataset_name "$dataset" \
      --model_name "$MODEL" \
      --input_file "$input" \
      --debug_mode "$DEBUG_MODE" \
      --max_rounds "$MAX_ROUNDS" \
      --eval_set_name "$eval_set" \
      --temperature "$TEMPERATURE" \
      --max_tokens "$MAX_TOKENS" \
      --n_workers "$N_WORKERS" \
      "${MODE_ARGS[@]}"
  ) > "$log" 2>&1 &
  PIDS+=($!)
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
  echo "ERROR: one or more jobs failed. Check $LOG_DIR."
  exit 1
fi
echo "All jobs done."

echo ""
echo "============================================"
echo "=== Union Summary ($MODEL, $SUBSET) ==="
echo "============================================"
# NOTE: [design thought] Per-dataset lines are already emitted by Evaluator
# .print_summary() inside bug_correct.py's round loop. Here we only aggregate
# them into a union across datasets.

FILE_TAG="$FILE_TAG" SHORT_MODEL="$SHORT_MODEL" MAX_ROUNDS="$MAX_ROUNDS" $PYTHON -c "
import json, os
file_tag    = os.environ['FILE_TAG']
short_model = os.environ['SHORT_MODEL']
# NOTE: [design thought] Aggregate the FINAL round's scores; earlier rounds
# are intermediate snapshots that get refined as failed attempts feed into
# the next round.
final_round = int(os.environ['MAX_ROUNDS'])
datasets = ['bigcodebench', 'livecodebench']

union = [0.0, 0.0, 0.0, 0.0, 0]  # unit, prec, rec, f1 sums + total n
for dataset in datasets:
    sf = f'results/{dataset}/eval_results/{short_model}_on_{dataset}_{file_tag}_round_{final_round}_scores.json'
    if not os.path.exists(sf):
        continue
    scores = json.load(open(sf))
    unit = scores.get('Unit score', {})
    sym  = scores.get('Symbolic block scores', {})
    n = len(unit)
    if n == 0:
        continue
    union[0] += sum(unit.values())
    union[1] += sum(v['precision'] for v in sym.values())
    union[2] += sum(v['recall']    for v in sym.values())
    union[3] += sum(v['f1']        for v in sym.values())
    union[4] += n

if union[4] > 0:
    u, p, r, f, n = union
    print(f'  union  unit={u/n:.3f} prec={p/n:.3f} rec={r/n:.3f} f1={f/n:.3f} (n={n})')
else:
    print('  union  (no results found)')
"
