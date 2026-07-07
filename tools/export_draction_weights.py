#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file


RENDERER_PREFIXES = (
    "_skeleton_renderer_module._renderer.",
    "module._skeleton_renderer_module._renderer.",
)


def find_index(checkpoint: Path) -> Path | None:
    if checkpoint.is_file() and checkpoint.name.endswith(".index.json"):
        return checkpoint
    if not checkpoint.is_dir():
        return None
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        candidate = checkpoint / name
        if candidate.is_file():
            return candidate
    return None


def renderer_key(key: str) -> str | None:
    for prefix in RENDERER_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return None


def renderer_shards(checkpoint: Path) -> list[Path]:
    index_path = find_index(checkpoint)
    if index_path is not None:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = {
            shard
            for key, shard in index["weight_map"].items()
            if renderer_key(key) is not None
        }
        if not shard_names:
            raise ValueError(f"No DrAction tensors found in {index_path}")
        return [index_path.parent / name for name in sorted(shard_names)]

    if checkpoint.is_file():
        return [checkpoint]

    candidates = sorted(checkpoint.glob("model*.safetensors"))
    candidates += sorted(checkpoint.glob("pytorch_model*.bin"))
    if not candidates:
        raise FileNotFoundError(f"No model weights found under {checkpoint}")
    return candidates


def load_shard(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload in {path}")
    return state


def extract_renderer_state(checkpoint: Path) -> dict[str, torch.Tensor]:
    renderer_state: dict[str, torch.Tensor] = {}
    for shard in renderer_shards(checkpoint):
        state = load_shard(shard)
        for key, value in state.items():
            short_key = renderer_key(key)
            if short_key is not None:
                renderer_state[short_key] = value.detach().cpu().contiguous()
        del state
    if not renderer_state:
        raise ValueError(f"No DrAction tensors found in {checkpoint}")
    return renderer_state


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_metadata(state: dict[str, torch.Tensor]) -> dict[str, Any]:
    joints = state.get("canonical_joints")
    features = state.get("canonical_features")
    return {
        "num_joints": int(joints.shape[0]) if joints is not None else None,
        "num_gaussians": int(features.shape[0]) if features is not None else None,
        "feature_dim": int(features.shape[1]) if features is not None else None,
        "enable_nfm": any(key.startswith("nfm.") for key in state),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract standalone DrAction tensors from a SkeletonLLM checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True, help="Public candidate name.")
    parser.add_argument("--metric", default="", help="Optional public metric string.")
    parser.add_argument("--description", default="")
    parser.add_argument("--num-line-samples", type=int, default=10)
    parser.add_argument("--render-frames", type=int, default=12)
    parser.add_argument("--render-size", type=int, default=448)
    args = parser.parse_args()

    state = extract_renderer_state(args.checkpoint)
    inferred = infer_metadata(state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_metadata = {
        "format": "skeletonllm-draction-v1",
        "name": args.name,
        "metric": args.metric,
        "description": args.description,
        "num_line_samples": str(args.num_line_samples),
        "render_frames": str(args.render_frames),
        "render_size": str(args.render_size),
        "enable_nfm": str(inferred["enable_nfm"]).lower(),
    }
    save_file(state, str(args.output), metadata=tensor_metadata)

    manifest = {
        "format": "skeletonllm-draction-v1",
        "name": args.name,
        "metric": args.metric,
        "description": args.description,
        "source_checkpoint": args.checkpoint.name,
        "num_line_samples": args.num_line_samples,
        "render_frames": args.render_frames,
        "render_size": args.render_size,
        **inferred,
        "tensor_count": len(state),
        "tensors": {key: list(value.shape) for key, value in sorted(state.items())},
        "sha256": sha256(args.output),
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(state)} tensors to {args.output}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
