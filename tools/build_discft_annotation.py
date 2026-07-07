#!/usr/bin/env python
"""Build the Discriminative Finetuning (Disc-FT, Stage 2) annotation for SkeletonLLM.

Disc-FT sharpens decision boundaries between visually similar actions with a
binary judgment task: "Is the action in this video clip '<X>'? YES/NO". Hard
negatives are the top-5 MLLM-mined semantically similar actions per class
(``data/ntu60_similar_actions.json``), intersected with the SEEN classes of the
chosen zero-shot split (open-vocabulary protocol: unseen classes never appear in
training prompts).

For every training clip we emit a positive (YES, true label) sample and, when a
seen hard-negative neighbor exists, a negative (NO, a random neighbor) sample,
yielding an approximately 1:1 YES:NO balance. Output format matches the MQA
annotation (one JSONL line per sample with a single skeleton ``image`` that
DrAction renders into ``--num-frames`` frames).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skeletonllm.data.ntu_meta import (
    NTU60_CLASSES,
    NTU60_SPLIT_PRESETS,
    is_ntu60_cs_train_file,
    parse_sample_id,
    seen_action_ids,
)
from skeletonllm.data.prompts import build_discft_question

DEFAULT_SIMILAR = REPO_ROOT / "data" / "ntu60_similar_actions.json"
LABEL_TO_ID = {name: idx + 1 for idx, name in enumerate(NTU60_CLASSES)}


def make_item(stem: str, image: str, question: str, answer: str, suffix: str) -> dict:
    return {
        "id": f"{stem}_{suffix}",
        "image": image,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": answer},
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the Disc-FT (Stage 2) binary-judgment annotation JSONL for SkeletonLLM."
    )
    parser.add_argument("--skeleton-root", type=Path, required=True,
                        help="Directory containing NTU .skeleton files.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--split", default="ntu60_48_cs", choices=sorted(NTU60_SPLIT_PRESETS),
                        help="Zero-shot split; negatives are restricted to its SEEN classes.")
    parser.add_argument("--similar-actions", type=Path, default=DEFAULT_SIMILAR,
                        help="JSON mapping each class to its top-k similar actions (hard negatives).")
    parser.add_argument("--num-frames", type=int, default=12,
                        help="Number of <image> placeholders (rendered frames) per prompt.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for negative selection.")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Optional cap on total emitted samples (smoke tests). 0 = all.")
    args = parser.parse_args()

    if not args.skeleton_root.is_dir():
        raise FileNotFoundError(f"skeleton root not found: {args.skeleton_root}")
    similar = json.loads(args.similar_actions.read_text(encoding="utf-8"))

    seen_ids = set(seen_action_ids(args.split))
    rng = random.Random(args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = pos = neg = skipped = no_neg = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for sk_file in sorted(args.skeleton_root.glob("*.skeleton")):
            stem = sk_file.stem
            ids = parse_sample_id(stem)
            if ids is None or not is_ntu60_cs_train_file(stem) or ids["action"] not in seen_ids:
                skipped += 1
                continue
            label = NTU60_CLASSES[ids["action"] - 1]

            # positive (YES): ask about the true label
            fout.write(json.dumps(make_item(
                stem, sk_file.name, build_discft_question(label, args.num_frames), "YES", "pos"
            ), ensure_ascii=False) + "\n")
            written += 1
            pos += 1
            if args.max_samples and written >= args.max_samples:
                break

            # negative (NO): a random seen hard-negative neighbor, if any
            neighbors = [n for n in similar.get(label, [])
                         if n != label and LABEL_TO_ID.get(n) in seen_ids]
            if not neighbors:
                no_neg += 1
                continue
            z = rng.choice(neighbors)
            fout.write(json.dumps(make_item(
                stem, sk_file.name, build_discft_question(z, args.num_frames), "NO", "neg"
            ), ensure_ascii=False) + "\n")
            written += 1
            neg += 1
            if args.max_samples and written >= args.max_samples:
                break

    print(f"Wrote {written} Disc-FT samples to {args.output} (YES={pos}, NO={neg})")
    print(f"Split '{args.split}': {len(seen_ids)} seen classes")
    print(f"Skipped {skipped} non-training files; {no_neg} clips had no seen hard-negative neighbor")


if __name__ == "__main__":
    main()
