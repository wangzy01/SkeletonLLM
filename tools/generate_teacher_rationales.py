#!/usr/bin/env python
"""Generate teacher causal-reasoning rationales for CR-Distill (Stage 3).

This is the distillation *pipeline* — it does not ship any generated captions.
Given DrAction-rendered frames for training clips, it queries a strong MLLM
teacher (GPT-4o by default, via an OpenAI-compatible API) with the
label-conditioned teacher prompt from the paper appendix, and writes the raw
teacher rationales to a JSONL file. ``build_crdistill_annotation.py`` then turns
those rationales into the student training annotation.

Recommended workflow (matches the paper):
  1. Train Stage 1 (warm-up) and Stage 2 (Disc-FT); export the Stage-2 DrAction
     weights (``tools/export_draction_weights.py``).
  2. Render training clips into frame folders with those weights:
       ``visualize/draction_visualize.py --dataset ntu --enable-nfm ...``
     producing ``<frames-root>/<sample_id>/frame_01.png ...``.
  3. Run this script over ``<frames-root>`` to produce ``teacher_rationales.jsonl``.

Each output line: ``{"id": <sample_id>, "label": <class>, "rationale": <text>}``.
Only clips whose class is in the split's SEEN set are queried (open-vocab).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skeletonllm.data.ntu_meta import (
    NTU60_CLASSES,
    NTU60_SPLIT_PRESETS,
    parse_sample_id,
    seen_action_ids,
)
from skeletonllm.data.prompts import SYSTEM_PROMPT, build_cr_teacher_prompt

FRAME_EXTS = (".png", ".jpg", ".jpeg")


def load_frames(clip_dir: Path, num_frames: int) -> list[Path]:
    frames = sorted(p for p in clip_dir.iterdir() if p.suffix.lower() in FRAME_EXTS)
    if num_frames > 0:
        frames = frames[:num_frames]
    return frames


def encode_data_uri(path: Path) -> str:
    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_messages(label: str, frame_paths: list[Path]) -> list[dict]:
    """OpenAI-style vision messages: system + user(frames interleaved with prompt)."""
    content: list[dict] = []
    for i, fp in enumerate(frame_paths, start=1):
        content.append({"type": "text", "text": f"Frame{i}:"})
        content.append({"type": "image_url", "image_url": {"url": encode_data_uri(fp)}})
    # textual instruction (the frame placeholders are already satisfied by the images above)
    instruction = build_cr_teacher_prompt(label, num_frames=0)
    content.append({"type": "text", "text": instruction})
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CR-Distill teacher rationales via an OpenAI-compatible MLLM."
    )
    parser.add_argument("--frames-root", type=Path, required=True,
                        help="Directory of per-clip frame folders (<sample_id>/frame_*.png).")
    parser.add_argument("--output", type=Path, required=True, help="Output rationales JSONL.")
    parser.add_argument("--dataset", default="ntu", choices=["ntu"],
                        help="Label source for parsing sample ids (NTU only for now).")
    parser.add_argument("--split", default="ntu60_48_cs", choices=sorted(NTU60_SPLIT_PRESETS),
                        help="Restrict generation to this split's SEEN classes.")
    parser.add_argument("--num-frames", type=int, default=12)
    parser.add_argument("--model", default="gpt-4o", help="Teacher model name.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"),
                        help="OpenAI-compatible base URL (optional; else the SDK default).")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="Environment variable holding the API key.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-query clips already present in the output (else resume/skip).")
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Please `pip install openai` to run teacher generation.") from exc

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Set the API key in ${{{args.api_key_env}}}.")
    client = OpenAI(api_key=api_key, base_url=args.base_url) if args.base_url else OpenAI(api_key=api_key)

    seen_ids = set(seen_action_ids(args.split))

    # resume support: skip ids already written
    done: set[str] = set()
    if args.output.exists() and not args.overwrite:
        for line in args.output.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass

    clip_dirs = sorted(p for p in args.frames_root.iterdir() if p.is_dir())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = errors = 0
    mode = "w" if args.overwrite else "a"
    with args.output.open(mode, encoding="utf-8") as fout:
        for clip_dir in clip_dirs:
            stem = clip_dir.name
            ids = parse_sample_id(stem)
            if ids is None or ids["action"] not in seen_ids:
                skipped += 1
                continue
            if stem in done:
                skipped += 1
                continue
            frames = load_frames(clip_dir, args.num_frames)
            if not frames:
                skipped += 1
                continue
            label = NTU60_CLASSES[ids["action"] - 1]
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=build_messages(label, frames),
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                rationale = resp.choices[0].message.content.strip()
            except Exception as exc:  # network/API errors: log and continue
                print(f"[error] {stem}: {exc}", file=sys.stderr)
                errors += 1
                continue
            fout.write(json.dumps({"id": stem, "label": label, "rationale": rationale},
                                  ensure_ascii=False) + "\n")
            fout.flush()
            written += 1
            if args.max_samples and written >= args.max_samples:
                break

    print(f"Wrote {written} rationales to {args.output}; skipped {skipped}; errors {errors}")


if __name__ == "__main__":
    main()
