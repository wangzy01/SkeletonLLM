#!/usr/bin/env python
"""Build the CR-Distill (Stage 3) student annotation from teacher rationales.

Consumes the ``teacher_rationales.jsonl`` produced by
``generate_teacher_rationales.py`` and emits the student training annotation.
The student prompt does NOT contain the ground-truth label; the target is the
full teacher response (the causal rationale together with its terminal
``Label: <action>`` line), matching the paper's main CR-Distill setting. Stage 3
updates DrAction, the projector, and the LLM LoRA adapters.

Each output line::

    {"id": <sample_id>, "image": <file>.skeleton,
     "conversations": [{"from": "human", "value": <student prompt>},
                       {"from": "gpt",   "value": <teacher rationale>}]}

This repository ships no rationales; generate your own with GPT-4o (or another
teacher) via ``generate_teacher_rationales.py`` first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skeletonllm.data.prompts import build_cr_student_question


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the CR-Distill (Stage 3) student annotation JSONL for SkeletonLLM."
    )
    parser.add_argument("--rationales", type=Path, required=True,
                        help="Teacher rationales JSONL from generate_teacher_rationales.py.")
    parser.add_argument("--skeleton-root", type=Path, required=True,
                        help="Directory containing NTU .skeleton files (to resolve the image field).")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--num-frames", type=int, default=12,
                        help="Number of <image> placeholders (rendered frames) per prompt.")
    parser.add_argument("--keep-missing-label", action="store_true",
                        help="Keep rationales that lack a final 'Label:' line. By default such "
                             "rationales are skipped, so the distillation target always includes "
                             "the terminal label line.")
    args = parser.parse_args()

    question = build_cr_student_question(args.num_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = missing = no_label = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for line in args.rationales.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stem, rationale = rec["id"], rec["rationale"].strip()
            sk_file = args.skeleton_root / f"{stem}.skeleton"
            if not sk_file.exists():
                missing += 1
                continue
            if not args.keep_missing_label and "label:" not in rationale.lower():
                no_label += 1
                continue
            item = {
                "id": stem,
                "image": sk_file.name,
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": rationale},
                ],
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} CR-Distill samples to {args.output}")
    if missing:
        print(f"Skipped {missing} rationales with no matching .skeleton file")
    if no_label:
        print(f"Skipped {no_label} rationales without a 'Label:' line")


if __name__ == "__main__":
    main()
