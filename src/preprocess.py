"""
Standalone preprocessing script for PDB (Precise Debugging Benchmarking) datasets.

Converts raw dataset files into a standardized format, filters by editable
line count, and validates every ground-truth solution against the dataset's
vendored unit-test sandbox. Task IDs that fail either filter are reported
and the valid subset is saved under the same directory with the `_valid`
postfix (preserving the original dict-vs-list raw format).

Usage example (validates 3 long splits on BigCodeBench):

    python src/preprocess.py --dataset_name bigcodebench \
        --input_file long_claude.json long_gemini.json long_gpt.json
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json

from dataset import get_handler

_VALIDATE_FILENAME_PATTERN = "_validate_"


# NOTE: [design thought] We must save the _valid file in the raw format so
# downstream callers (bug_generation.py etc.) can re-preprocess it. Different
# datasets use different raw shapes: bcb is a dict keyed by task_id,
# lcb/kod are lists keyed by an inner id field. This helper inspects the
# raw container and subsets it without reformatting the entries.
def filter_raw_to_valid(raw, valid_ids, id_field_candidates=("task_id", "question_id", "id")):
    if isinstance(raw, dict):
        return {k: v for k, v in raw.items() if k in valid_ids}
    if isinstance(raw, list):
        valid = set(valid_ids)
        id_fn = None
        if raw:
            for f in id_field_candidates:
                if f in raw[0]:
                    id_fn = (lambda fname: (lambda e: e[fname]))(f)
                    break
        if id_fn is None:
            raise ValueError(
                f"Cannot infer task_id field from raw entry keys "
                f"{list(raw[0].keys()) if raw else '[]'}"
            )
        return [e for e in raw if id_fn(e) in valid]
    raise ValueError(f"Unsupported raw container type: {type(raw).__name__}")


def process_single_file(input_name, data_dir, handler, min_editable, skip_verify=False):
    """Validate one input file; return (fail_ids, correct_ids, valid_count, total)."""
    in_path = os.path.join(data_dir, input_name)
    raw = json.load(open(in_path, "r"))

    processed = handler.preprocess(raw)
    total = len(processed)
    print(f"[{input_name}] raw={total}")

    handler.mark_editable_lines(processed)
    ed_ok = [d for d in processed if len(d["editable_lines"]) >= min_editable]
    ed_fail_ids = {d["task_id"] for d in processed} - {d["task_id"] for d in ed_ok}
    print(f"[{input_name}] editable>={min_editable}: kept {len(ed_ok)}/{total}; "
          f"dropped {len(ed_fail_ids)} by low-editable-count")

    if skip_verify or not ed_ok:
        correct_ids = {d["task_id"] for d in ed_ok}
        unit_fail_ids = set()
    else:
        # NOTE: [design thought] Sandbox intermediates (verify .jsonl, eval
        # results, pass-at-k summaries) are throwaway artifacts; route them
        # to data/log/ so they don't pollute the data dir.
        log_dir = os.path.join(data_dir, "log")
        os.makedirs(log_dir, exist_ok=True)
        verify_prefix = os.path.join(
            log_dir, f"{_VALIDATE_FILENAME_PATTERN}{os.path.splitext(input_name)[0]}"
        )
        verify_file = handler.build_verify_unit_test(
            verify_prefix, ed_ok, sol_field="gt_solution"
        )
        _, correct_ids, _ = handler.verify_unit_test(verify_file, timeout=1800)
        correct_ids = set(correct_ids)
        unit_fail_ids = {d["task_id"] for d in ed_ok} - correct_ids
        print(f"[{input_name}] GT unit-tests: passed {len(correct_ids)}, "
              f"failed {len(unit_fail_ids)}")

    all_invalid = ed_fail_ids | unit_fail_ids
    if all_invalid:
        print(f"[{input_name}] INVALID task_ids ({len(all_invalid)}): "
              f"{sorted(all_invalid)}")
    else:
        print(f"[{input_name}] all tasks valid")

    # Save the valid subset preserving the raw format
    valid_raw = filter_raw_to_valid(raw, correct_ids)
    stem, ext = os.path.splitext(input_name)
    out_path = os.path.join(data_dir, f"{stem}_valid{ext}")
    with open(out_path, "w") as f:
        json.dump(valid_raw, f, indent=2)
    kept = len(valid_raw) if hasattr(valid_raw, "__len__") else 0
    print(f"[{input_name}] wrote {out_path} ({kept} valid entries)")

    return all_invalid, correct_ids, kept, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="bigcodebench",
                        help="Dataset name")
    parser.add_argument("--input_file", nargs='+', default=["full_data.json"],
                        help="Input files under dataset/<dataset_name>/data; "
                             "each is validated independently and yields its "
                             "own <stem>_valid.json")
    parser.add_argument("--min_editable", type=int, default=6,
                        help="Minimal number of editable lines per task")
    parser.add_argument("--skip_verify", action="store_true",
                        help="Skip the ground-truth unit-test run; keep only "
                             "the min_editable filter. Useful for smoke tests.")
    args = parser.parse_args()

    data_dir = os.path.join("dataset", args.dataset_name, "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    handler = get_handler(args.dataset_name)
    overall_invalid = {}
    for input_name in args.input_file:
        invalid, correct, kept, total = process_single_file(
            input_name, data_dir, handler, args.min_editable,
            skip_verify=args.skip_verify,
        )
        overall_invalid[input_name] = (invalid, kept, total)
        print("")

    print("=== summary ===")
    for name, (inv, kept, total) in overall_invalid.items():
        print(f"  {name:30s} valid={kept}/{total}  invalid={len(inv)}")
