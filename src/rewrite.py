"""
Code rewriting pipeline for PDB (Precise Debugging Benchmarking).

Rewrites ground-truth solutions using an LLM while preserving frozen lines
(function signatures, imports) and checking semantic similarity via CodeBLEU.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import re
import datetime
import numpy as np
import random
import tqdm
import json
import copy
import dspy
import argparse
from dataset import get_handler
from utils import file_diff, apply_diff, verify_block_single_diff
from module import Rewriter, CODE_BLOCK_REGEX, SIMPLE_CODE_BLOCK_REGEX
from codebleu import calc_codebleu


def check_frozen(rewritten, original, frozen_lines):
    rewritten_frozen_lines = rewritten.strip().splitlines()[:frozen_lines]
    original_frozen_lines = original.strip().splitlines()[:frozen_lines]
    return "\n".join(rewritten_frozen_lines) == "\n".join(original_frozen_lines)


def rewrite(data, handler, log_file_prefix, max_inner_try=2, max_outer_try=1, lang="python", threshold=0.7):
    all_results = []
    success_rewritten_data = []
    rewriter = Rewriter()

    print("Rewriting")

    for outer_try in range(max_outer_try):
        results = []
        for index, item in tqdm.tqdm(enumerate(data), total=len(data)):
            if "rewritten_success" in item and item["rewritten_success"]:
                continue
            if "editable_lines" in item:
                del item["editable_lines"]
            if "deletable_lines" in item:
                del item["deletable_lines"]
            if "gt_length" in item:
                del item["gt_length"]
            task_id = item.get("task_id")
            gt_solution = item.get("gt_solution")
            frozen_lines = item.get("frozen_lines")
            task_prompt = item.get("task_prompt")

            trial = 0
            while trial < max_inner_try:
                try:
                    response = rewriter(task_prompt=task_prompt, gt_solution=gt_solution)
                    match = CODE_BLOCK_REGEX.search(response.rewritten_code)
                    match_simple = SIMPLE_CODE_BLOCK_REGEX.search(response.rewritten_code)
                    if match:
                        item["rewritten_solution"] = match.group(1).strip()
                    elif match_simple:
                        item["rewritten_solution"] = match_simple.group(1).strip()
                    else:
                        item["rewritten_solution"] = response.rewritten_code.strip()
                except Exception as e:
                    item["rewritten_solution"] = None
                    print(f"Error processing task_id {task_id}: {e}")

                # check frozen and enough difference
                if not item["rewritten_solution"]:
                    print("No rewritten solution.")
                    trial += 1
                    continue

                if not check_frozen(item["rewritten_solution"], gt_solution, frozen_lines):
                    print("Fail on editing frozen lines.")
                    item["rewritten_solution"] = None
                    trial += 1
                    continue

                if not calc_codebleu([gt_solution], [item["rewritten_solution"]], lang)["codebleu"] <= threshold:
                    print("Fail on similarity test.")
                    item["rewritten_solution"] = None
                    trial += 1
                    continue

                break

            if item["rewritten_solution"]:
                results.append(item)

        verify_file = handler.build_verify_unit_test(log_file_prefix + "_rewrite_verify", results,
                                                     sol_field="rewritten_solution")
        fail_ids, correct_ids, _ = handler.verify_unit_test(verify_file, timeout=1800)

        # Update results with success status
        for entry in results:
            if entry["task_id"] in correct_ids:
                entry["rewritten_success"] = True
                success_rewritten_data.append(entry)
            else:
                entry["rewritten_success"] = False

        all_results += results
        print("Rewriting success:", len(correct_ids), "in", len(results))

    with open(log_file_prefix + "_rewrite.json", "w") as f:
        json.dump(all_results, f, indent=2)

    for entry in success_rewritten_data:
        entry["gt_solution"] = entry["rewritten_solution"]
        del entry["rewritten_solution"]
        del entry["rewritten_success"]
    return success_rewritten_data


def rewrite_main(args):
    data_dir = os.path.join("data", args.dataset_name)
    log_dir = os.path.join("results", args.dataset_name, "bug_data", "log")

    os.makedirs("keys", exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Add datetime
    time_to_add = datetime.datetime.now().strftime("%m%d-%H%M")

    model_api_file = os.path.join("keys", args.model_api_file)
    id_filtering_file = os.path.join(data_dir, args.id_filtering_file)
    log_file_prefix = os.path.join(log_dir, args.log_prefix) + "_" + time_to_add

    # Load the dataset
    if len(args.input_file) == 1:
        input_file = os.path.join(data_dir, args.input_file[0])
        raw_data = json.load(open(input_file, "r"))
    else:
        input_files = [os.path.join(data_dir, args.input_file[i]) for i in range(len(args.input_file))]
        raw_data_list = [json.load(open(input_file, "r")) for input_file in input_files]
        raw_data = raw_data_list[0]
        for d in raw_data_list[1:]:
            raw_data.extend(d)

    handler = get_handler(args.dataset_name)
    print("Preprocessing data...")
    raw_data = handler.preprocess(raw_data)

    # Handle optional ID filtering file
    if os.path.exists(id_filtering_file):
        id_filtering = json.load(open(id_filtering_file, "r"))
        data_dict = {d["task_id"]: d for d in raw_data}
        raw_data = [data_dict[idx] for idx in id_filtering]

    # Load the model
    api_key = open(model_api_file, "r").read().strip()
    rewriter = dspy.LM(args.model_name, api_key=api_key, temperature=args.temperature, max_tokens=args.max_tokens)
    dspy.settings.configure(lm=rewriter)

    handler.mark_editable_lines(raw_data)
    new_data = rewrite(raw_data, handler, log_file_prefix)
    with open(os.path.join(data_dir, args.output_file), "w") as f:
        json.dump(new_data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="Dataset name", required=True)
    parser.add_argument("--model_name", type=str, help="Bug generation model name", required=True)
    parser.add_argument("--model_api_file", type=str, required=True,
                        help="Model API file is required for generation")
    parser.add_argument("--input_file", nargs='+', help="Input file path, under data/{dataset_name}",
                        default=["bigcodebench-full-data.json"])
    # parser.add_argument("--reload_from_save", type=str, default="", help="Reload from saved dir")
    parser.add_argument("--id_filtering_file", type=str,
                        help="ID filtering file path, under data/{dataset_name}", default="id_filtering.json")
    parser.add_argument("--log_prefix", type=str, help="Log file under log/{dataset_name}",
                        default="log")
    parser.add_argument("--output_file", type=str, help="Output file path, under data/{dataset_name}",
                        default="bigcodebench-rewritten-claude.json")
    parser.add_argument("--max_tokens", type=int, default=16000, help="Maximum number of tokens")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the generator")

    args = parser.parse_args()
    rewrite_main(args)
