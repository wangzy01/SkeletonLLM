#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skeletonllm.model.skeletonllm_chat import InternVLChatModel
from skeletonllm.data.ntu_meta import (
    NTU60_SPLIT_PRESETS,
    build_mqa_question,
    is_ntu60_cs_test_file,
    test_action_ids,
)


def torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping[name]


def load_inner_renderer_if_needed(model, model_path: Path, enable_nfm: bool) -> None:
    if not getattr(model, "use_skeleton", False):
        return
    renderer_mod = getattr(model, "_skeleton_renderer_module", None)
    if renderer_mod is None:
        return
    if getattr(renderer_mod, "_renderer", None) is not None:
        renderer_mod.enable_nfm = enable_nfm
        return
    if not hasattr(renderer_mod, "_ensure_renderer"):
        return

    device = next(model.parameters()).device
    renderer_mod.enable_nfm = enable_nfm
    inner = renderer_mod._ensure_renderer(device)
    ckpt_files = sorted(model_path.glob("pytorch_model*.bin")) or sorted(model_path.glob("model*.safetensors"))

    renderer_state = {}
    for ckpt_file in ckpt_files:
        if ckpt_file.suffix == ".bin":
            state = torch.load(ckpt_file, map_location="cpu")
        else:
            from safetensors.torch import load_file

            state = load_file(str(ckpt_file))
        for key, value in state.items():
            prefix = "_skeleton_renderer_module._renderer."
            if key.startswith(prefix):
                renderer_state[key[len(prefix) :]] = value

    if renderer_state:
        inner.load_state_dict(renderer_state, strict=False)
        renderer_mod._renderer = inner.to(device=device, dtype=torch.float32)
        print(f"[Info] Loaded {len(renderer_state)} renderer tensors.", file=sys.stderr)
    else:
        print("[Warn] No renderer tensors found in checkpoint.", file=sys.stderr)


def save_frames(model, sample_id: str, skeleton_path: Path, save_root: Path, fmt: str):
    frames = model._maybe_render_skeleton_input(str(skeleton_path))
    out_dir = save_root / sample_id
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_np = frames.squeeze(0).detach().clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy()
    from PIL import Image

    for idx, frame in enumerate(frames_np, start=1):
        Image.fromarray(frame, mode="RGB").save(out_dir / f"frame_{idx:02d}.{fmt}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SkeletonLLM NTU action recognition inference.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--skeleton-root", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument(
        "--split",
        default="ntu60_48_cs",
        choices=sorted(NTU60_SPLIT_PRESETS),
    )
    parser.add_argument(
        "--action-ids",
        default=None,
        help="Optional comma-separated action ids. Overrides --split.",
    )
    parser.add_argument("--render-frames", type=int, default=12)
    parser.add_argument("--render-size", type=int, default=448)
    parser.add_argument("--enable-nfm", action="store_true", default=True)
    parser.add_argument("--disable-nfm", action="store_false", dest="enable_nfm")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--simple-single-gpu", action="store_true")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--save-rendered-frames", action="store_true")
    parser.add_argument("--render-save-root", type=Path, default=Path("rendered_frames"))
    parser.add_argument("--save-format", default="png", choices=["png", "jpg", "jpeg"])
    args = parser.parse_args()

    if not args.skeleton_root.is_dir():
        raise FileNotFoundError(f"skeleton root not found: {args.skeleton_root}")

    action_ids = set(test_action_ids(args.split, args.action_ids))
    question = build_mqa_question(sorted(action_ids), args.render_frames)
    per_frame_counts = [1] * args.render_frames

    print("[Info] Loading model...", file=sys.stderr)
    if args.simple_single_gpu:
        model = InternVLChatModel.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype(args.dtype),
            low_cpu_mem_usage=False,
            use_flash_attn=True,
        ).eval().to("cuda:0")
    else:
        model = InternVLChatModel.from_pretrained(
            args.model_path,
            torch_dtype=torch_dtype(args.dtype),
            low_cpu_mem_usage=False,
            use_flash_attn=True,
            device_map="auto",
        ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Warn on an unmerged LoRA checkpoint: from_pretrained does not re-wrap LoRA,
    # so the checkpoint's LoRA weights are silently ignored and the reported
    # accuracy would reflect the base model without the trained LoRA adaptation.
    if int(getattr(model.config, "use_llm_lora", 0) or 0) > 0:
        print(
            f"[Warn] '{args.model_path}' has use_llm_lora>0 but LoRA is not re-wrapped on load, so its "
            "LoRA weights are ignored. Merge first: `python tools/merge_lora.py <ckpt> <merged>` and "
            "evaluate the merged checkpoint.",
            file=sys.stderr,
        )

    load_inner_renderer_if_needed(model, args.model_path, args.enable_nfm)
    model.config.skeleton_target_num_frames = int(args.render_frames)
    model.config.force_image_size = int(args.render_size)

    generation_config = dict(max_new_tokens=16, do_sample=False, temperature=0.0, top_p=None, repetition_penalty=1.05)
    skeleton_files = sorted(args.skeleton_root.glob("*.skeleton"))
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    used = 0
    skipped = 0
    with args.output_file.open("w", encoding="utf-8") as fout:
        for sk_file in tqdm(skeleton_files, desc="Evaluating", unit="video"):
            sample_id = sk_file.stem
            if not is_ntu60_cs_test_file(sample_id, action_ids):
                skipped += 1
                continue
            try:
                pixel_values = str(sk_file)
                if args.save_rendered_frames:
                    pixel_values = save_frames(model, sample_id, sk_file, args.render_save_root, args.save_format)
                answer = model.chat(
                    tokenizer,
                    pixel_values=pixel_values,
                    question=question,
                    generation_config=generation_config,
                    num_patches_list=per_frame_counts,
                )
                fout.write(f"{sample_id}\t{answer}\n")
                used += 1
            except Exception as exc:
                fout.write(f"{sample_id}\tERROR: {exc}\n")
            fout.flush()
            if args.max_samples and used >= args.max_samples:
                break

    print(f"[Info] Saved results to {args.output_file}. Used={used}, skipped={skipped}", file=sys.stderr)


if __name__ == "__main__":
    main()
