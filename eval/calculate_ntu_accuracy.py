#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skeletonllm.data.ntu_meta import (
    NTU60_CLASSES,
    NTU60_SPLIT_PRESETS,
    test_action_ids,
)

ACTION_RE = re.compile(r"A(\d{3})")


def load_predictions(result_file: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    with result_file.open("r", encoding="utf-8", errors="ignore") as fin:
        for raw in fin:
            line = raw.rstrip("\n")
            if not line or "\t" not in line:
                continue
            sample_id, prediction = line.split("\t", 1)
            base_id = re.split(r"_q\d+", sample_id)[0]
            grouped[base_id].append(prediction)
    return grouped


def score(result_file: Path, action_ids: set[int]) -> tuple[int, int, dict[int, dict[str, int]]]:
    labels = {idx + 1: label.lower() for idx, label in enumerate(NTU60_CLASSES)}
    grouped = load_predictions(result_file)
    total = 0
    correct = 0
    per_class: dict[int, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    for sample_id, predictions in grouped.items():
        match = ACTION_RE.search(sample_id)
        if match is None:
            continue
        action_id = int(match.group(1))
        if action_id not in action_ids:
            continue
        label = labels[action_id]
        total += 1
        per_class[action_id]["total"] += 1

        hits = 0
        valid = 0
        for prediction in predictions:
            pred = prediction.lower().strip()
            if "error" in pred:
                continue
            valid += 1
            if label in pred:
                hits += 1
        if valid > 0 and hits >= valid / 2:
            correct += 1
            per_class[action_id]["correct"] += 1

    return total, correct, per_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate NTU action-recognition accuracy from TSV predictions.")
    parser.add_argument("result_file", type=Path)
    parser.add_argument(
        "--split",
        default="ntu60_48_cs",
        choices=sorted(NTU60_SPLIT_PRESETS),
        help="Preset NTU60 action split. Use --action-ids to override it.",
    )
    parser.add_argument(
        "--action-ids",
        default=None,
        help="Optional comma-separated unseen/test action ids. Overrides --split.",
    )
    args = parser.parse_args()

    action_ids = set(test_action_ids(args.split, args.action_ids))
    total, correct, per_class = score(args.result_file, action_ids)
    if total == 0:
        print("No matching samples found.")
        return

    print("Overall")
    print(f"  total:   {total}")
    print(f"  correct: {correct}")
    print(f"  acc:     {correct / total * 100:.2f}%")
    print("")
    print("Per class")
    for action_id in sorted(action_ids):
        stats = per_class.get(action_id, {"total": 0, "correct": 0})
        total_i = stats["total"]
        correct_i = stats["correct"]
        acc_i = correct_i / total_i * 100 if total_i else 0.0
        print(f"  {action_id:03d} {NTU60_CLASSES[action_id - 1]:35s} {correct_i:4d}/{total_i:<4d} {acc_i:6.2f}%")


if __name__ == "__main__":
    main()
