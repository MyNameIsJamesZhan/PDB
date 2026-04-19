"""
Merge buggy code results from multiple model runs into a single dataset.

Reads specific JSON files from an input directory, tags each entry with its
source model, computes gt_diff, builds a bug type dictionary for multi-bug
type inference, samples up to k examples per (orig_id, bug_count) group,
and writes the merged output.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import argparse
import numpy as np
from collections import defaultdict, Counter
from utils import file_diff, verify_block_diff, parse_diff_to_blocks, single_diff_to_str


# Filename prefix -> canonical model name
MODEL_PREFIX_MAP = {
    "oai": "gpt-5.1-codex",
    "gg": "gemini-2.5-pro",
    "ar": "claude-sonnet-4.5",
}


def infer_model(filename):
    """Infer source model from filename prefix (e.g. 'oai_buggy_...' -> 'gpt-5.1-codex')."""
    base = os.path.basename(filename)
    for prefix, model in MODEL_PREFIX_MAP.items():
        if base.startswith(prefix + "_"):
            return model
    raise ValueError(f"Undefined model for filename: {filename}")


def merge_buggy_files(input_dir, in_files, output_file, samples_per_group=5,
                      type_dict_file=None, max_lines_per_block=1,
                      min_lines_per_block=1, stride=2):
    """
    Merge specified JSON files in input_dir into a single deduplicated dataset.

    Each entry is tagged with source_model, gt_diff is recomputed, entries
    failing block-diff validation are skipped, then we sample up to
    samples_per_group examples per (orig_id, bug_count) group.
    """
    # --- Phase 1: Load, tag, validate, and build bug type dictionary ---
    all_data = []
    bug_dict = defaultdict(dict)
    type_dict = defaultdict(dict)
    skipped = 0
    merge_conflicts = 0

    for filename in in_files:
        model = infer_model(filename)
        filepath = os.path.join(input_dir, filename)
        data = json.load(open(filepath))

        for d in data:
            d["source_model"] = model
            d.pop("test", None)
            d["buggy_code"] = d["buggy_code"].strip()
            d["gt_solution"] = d["gt_solution"].strip()
            d.pop("diff", None)
            d["gt_diff"] = file_diff(d["buggy_code"], d["gt_solution"])[2]

            # NOTE: [design thought] bug_count is the number of contiguous
            # blocks, not the number of line-level edits. A 2-block multi-line
            # bug with 3 lines per block has bug_count=2, not 6.
            n_blocks = len(parse_diff_to_blocks(d["gt_diff"]))
            if not verify_block_diff(d["gt_diff"], block_count=n_blocks, stride=stride,
                                     max_lines_per_block=max_lines_per_block,
                                     min_lines_per_block=min_lines_per_block)[0]:
                print(f"Skipping {d['task_id']}")
                skipped += 1
                continue

            d["bug_count"] = n_blocks
            orig_id = d["task_id"].rsplit("_", 1)[0]

            # NOTE: [design thought] The per-block bug-type lookup only makes
            # sense in single-line mode where each block has exactly one edit
            # keyed by line number. In multi-line mode, a block spans 2-4
            # lines so single_diff_to_str can't represent it; we skip the
            # dedup/type-inference phase entirely and keep the entry's
            # source-reported bug_type/bug_subtype as-is.
            is_single_line_mode = (max_lines_per_block == 1)
            if not is_single_line_mode:
                type_dict[d["task_id"]] = {
                    "bug_type": d.get("bug_type"),
                    "bug_subtype": d.get("bug_subtype"),
                }
                all_data.append(d)
                continue

            if d["bug_count"] == 1:
                extract_diff = single_diff_to_str(d["gt_diff"]).split(": ", 1)[1]

                # Handle dedup conflicts: when two entries share the same diff text
                # but differ in bug_subtype, keep the later one (except BigCodeBench/273)
                if extract_diff in bug_dict[orig_id] and bug_dict[orig_id][extract_diff][1][0] != d["bug_subtype"][0]:
                    if orig_id != "BigCodeBench/273":
                        bug_dict[orig_id][extract_diff] = (
                            d["bug_type"], d["bug_subtype"], d["task_id"],
                            list(d["gt_diff"].keys())[0], extract_diff
                        )
                    merge_conflicts += 1
                else:
                    bug_dict[orig_id][extract_diff] = (
                        d["bug_type"], d["bug_subtype"], d["task_id"],
                        list(d["gt_diff"].keys())[0], extract_diff
                    )

                type_dict[d["task_id"]] = {
                    "bug_type": [d["bug_type"]],
                    "bug_subtype": [d["bug_subtype"]],
                }
            else:
                types, subtypes = [], []
                for line_number, edit in d["gt_diff"].items():
                    extract_diff = single_diff_to_str({line_number: edit}).split(": ", 1)[1]
                    if extract_diff in bug_dict[orig_id]:
                        bug_type, bug_subtype, _, _, _ = bug_dict[orig_id][extract_diff]
                        types.append(bug_type)
                        subtypes.append(bug_subtype)
                type_dict[d["task_id"]] = {
                    "bug_type": types,
                    "bug_subtype": subtypes,
                }

            all_data.append(d)

    print(f"After loading: {len(all_data)} entries ({skipped} skipped, {merge_conflicts} merge conflicts)")

    # --- Phase 2: Sample up to k examples per (orig_id, bug_count) group ---
    orig_id_bins = defaultdict(lambda: {1: [], 2: [], 3: [], 4: []})
    for d in all_data:
        orig_id = d["task_id"].rsplit("_", 1)[0]
        orig_id_bins[orig_id][d["bug_count"]].append(d)

    sampled_data = []
    for orig_id, bug_count_groups in orig_id_bins.items():
        for bug_count, examples in bug_count_groups.items():
            if len(examples) > samples_per_group:
                selected_indices = sorted(
                    np.random.choice(range(len(examples)), samples_per_group, replace=False)
                )
                sampled_data += [examples[i] for i in selected_indices]
            else:
                sampled_data += examples

    all_data = sampled_data

    # --- Phase 3: Print statistics and write output ---
    bug_count_dist = Counter(d["bug_count"] for d in all_data)
    model_dist = Counter(d["source_model"] for d in all_data)

    print(f"After sampling (k={samples_per_group}): {len(all_data)} entries")
    print("Bug count distribution:")
    for k in sorted(bug_count_dist):
        print(f"  bug_count {k}: {bug_count_dist[k]}")
    print("Source model distribution:")
    for model in sorted(model_dist):
        print(f"  {model}: {model_dist[model]}")

    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Wrote {output_file}")

    if type_dict_file:
        with open(type_dict_file, "w") as f:
            json.dump(type_dict, f, indent=2)
        print(f"Wrote {type_dict_file}")

    return all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge per-model buggy code files")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing per-model buggy code JSON files")
    parser.add_argument("--in_files", type=str, nargs="+", required=True,
                        help="List of specific JSON filenames to merge")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path for the merged output JSON")
    parser.add_argument("--samples_per_group", type=int, default=5,
                        help="Max samples per (orig_id, bug_count) group (default: 5)")
    parser.add_argument("--type_dict_file", type=str, default=None,
                        help="Path for the bug type dictionary (optional)")
    parser.add_argument("--max_lines_per_block", type=int, default=1,
                        help="Max lines per block for diff validation. 1 for "
                             "single-line mode, up to MAX_MULTILINES for multi.")
    parser.add_argument("--min_lines_per_block", type=int, default=1,
                        help="Min lines per block for diff validation. "
                             "Use MIN_MULTILINES (2) in multi mode.")
    parser.add_argument("--stride", type=int, default=2,
                        help="Min stride between blocks. Use 4 in multi mode.")
    args = parser.parse_args()
    merge_buggy_files(args.input_dir, args.in_files, args.output_file,
                      args.samples_per_group, args.type_dict_file,
                      max_lines_per_block=args.max_lines_per_block,
                      min_lines_per_block=args.min_lines_per_block,
                      stride=args.stride)
