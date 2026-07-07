#!/usr/bin/env python
"""Build a label-only NTU-60 MQA training annotation (JSONL) for SkeletonLLM.

Each output line is one training sample::

    {"id": <stem>, "image": <file>.skeleton,
     "conversations": [{"from": "human", "value": <MQA question>},
                       {"from": "gpt",   "value": <label>}]}

The MQA question lists the SEEN classes of the chosen zero-shot split (unseen
classes are excluded from training), and the answer is the class label only.
This annotation drives Stage 1 (Render Warm-up) and Stage 4 (Recognition
Refinement). The single skeleton file is rendered on the fly into ``--num-frames``
frames by DrAction, one per ``<image>`` placeholder in the question.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skeletonllm.data.ntu_meta import (
    NTU60_CLASSES,
    NTU60_SPLIT_PRESETS,
    build_mqa_question,
    is_ntu60_cs_train_file,
    parse_int_csv,
    parse_sample_id,
    seen_action_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a label-only NTU-60 MQA training annotation JSONL for SkeletonLLM."
    )
    parser.add_argument("--skeleton-root", type=Path, required=True,
                        help="Directory containing NTU .skeleton files.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--split", default="ntu60_48_cs", choices=sorted(NTU60_SPLIT_PRESETS),
                        help="Zero-shot split; its SEEN classes are used for training.")
    parser.add_argument("--num-frames", type=int, default=12,
                        help="Number of <image> placeholders (rendered frames) per prompt.")
    parser.add_argument("--include-action-ids", default="",
                        help="Optional comma-separated SEEN action ids that override the split.")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Optional cap for smoke tests. 0 = use all matching samples.")
    args = parser.parse_args()

    if not args.skeleton_root.is_dir():
        raise FileNotFoundError(f"skeleton root not found: {args.skeleton_root}")

    if args.include_action_ids:
        seen = sorted(set(parse_int_csv(args.include_action_ids)))
    else:
        seen = seen_action_ids(args.split)
    seen_set = set(seen)
    question = build_mqa_question(seen, args.num_frames)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for sk_file in sorted(args.skeleton_root.glob("*.skeleton")):
            stem = sk_file.stem
            ids = parse_sample_id(stem)
            if ids is None or not is_ntu60_cs_train_file(stem):
                skipped += 1
                continue
            if ids["action"] not in seen_set:
                skipped += 1
                continue
            item = {
                "id": stem,
                "image": sk_file.name,
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": NTU60_CLASSES[ids["action"] - 1]},
                ],
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1
            if args.max_samples and written >= args.max_samples:
                break

    print(f"Wrote {written} samples to {args.output}")
    print(f"Skipped {skipped} files")
    print(f"Split '{args.split}': {len(seen)} seen classes used for training")


if __name__ == "__main__":
    main()
