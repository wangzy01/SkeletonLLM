#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skeletonllm.model.skeletonllm_chat.draction import (
    DifferentiableSkeletonRenderer,
    _build_line_samples,
    _build_line_samples_adaptive,
    _compute_adaptive_scales,
    _make_lbs_logits_for_samples,
    _preprocess_poses_for_rendering,
    _sample_indices_uniform,
    get_bone_pairs,
    get_num_joints,
    parse_skeleton_file,
)


TOPOLOGY_STATE = {
    "canonical_joints",
    "canonical_means",
    "canonical_opacities",
    "canonical_quats",
    "canonical_scales",
    "lbs_weights_logits",
}


def weight_metadata(path: Path) -> dict[str, str]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def resize_canonical_features(
    source: torch.Tensor,
    source_joints: int,
    source_line_samples: int,
    target_joints: int,
    target_bones: int,
    target_line_samples: int,
) -> torch.Tensor:
    """Resize joint and bone feature banks separately for a new topology."""
    source_joint_features = source[:source_joints].T.unsqueeze(0)
    target_joint_features = F.interpolate(
        source_joint_features,
        size=target_joints,
        mode="linear",
        align_corners=True,
    ).squeeze(0).T

    source_line_features = source[source_joints:]
    if source_line_samples <= 0 or source_line_features.numel() == 0:
        mean_feature = source.mean(dim=0, keepdim=True)
        target_lines = mean_feature.expand(target_bones * target_line_samples, -1)
    else:
        source_bones = source_line_features.shape[0] // source_line_samples
        source_line_features = source_line_features[
            : source_bones * source_line_samples
        ].reshape(source_bones, source_line_samples, -1)
        features = source_line_features.permute(2, 0, 1).unsqueeze(0)
        target_lines = F.interpolate(
            features,
            size=(target_bones, target_line_samples),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0).permute(1, 2, 0).reshape(-1, source.shape[1])
    return torch.cat([target_joint_features, target_lines], dim=0)


def load_weights(
    renderer: DifferentiableSkeletonRenderer,
    weight_path: Path,
    skeleton_type: str,
    num_line_samples: int,
) -> dict[str, object]:
    source = load_file(str(weight_path), device="cpu")
    target = renderer.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped: dict[str, list[int]] = {}

    for key, value in source.items():
        if key in TOPOLOGY_STATE:
            continue
        if key == "canonical_features":
            continue
        if key in target and target[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped[key] = list(value.shape)

    source_joints = int(source["canonical_joints"].shape[0])
    source_gaussians = int(source["canonical_features"].shape[0])
    source_line_samples = int(
        weight_metadata(weight_path).get("num_line_samples", num_line_samples)
    )
    target_joints = get_num_joints(skeleton_type)
    target_bones = len(get_bone_pairs(skeleton_type))
    exact_topology = (
        source["canonical_features"].shape == target["canonical_features"].shape
        and source_joints == target_joints
    )

    if exact_topology:
        for key in TOPOLOGY_STATE | {"canonical_features"}:
            if key in source and key in target and source[key].shape == target[key].shape:
                compatible[key] = source[key]
    else:
        compatible["canonical_features"] = resize_canonical_features(
            source["canonical_features"],
            source_joints=source_joints,
            source_line_samples=source_line_samples,
            target_joints=target_joints,
            target_bones=target_bones,
            target_line_samples=num_line_samples,
        )

    missing, unexpected = renderer.load_state_dict(compatible, strict=False)
    if exact_topology:
        renderer._canonical_joints_initialized = True
        renderer._canonical_gaussians_initialized = True
        renderer._canonical_scales_initialized = True
        renderer._lbs_initialized = True

    return {
        "exact_topology": exact_topology,
        "source_joints": source_joints,
        "source_gaussians": source_gaussians,
        "loaded": sorted(compatible),
        "missing": list(missing),
        "unexpected": list(unexpected),
        "skipped": skipped,
    }


def setup_input_topology(
    renderer: DifferentiableSkeletonRenderer,
    canonical: torch.Tensor,
    pairs: list[tuple[int, int]],
    num_line_samples: int,
    joint_scale: float = 0.030,
    line_scale: float = 0.020,
    scale_gamma: float = 1.0,
) -> None:
    line_samples, sample_defs = _build_line_samples(
        canonical, num_line_samples, pairs=pairs
    )
    num_joints = canonical.shape[0]
    canonical_means = torch.cat([canonical, line_samples], dim=0)
    renderer.set_canonical_joints(canonical)
    renderer.set_canonical_means(canonical_means)

    # joint_scale / line_scale set both the base and the upper clamp so the
    # caller can directly thicken the rendered bones (only used for the
    # cross-topology path, e.g. HumanML3D). scale_gamma keeps a natural taper
    # from thick proximal bones to thinner distal ones.
    joint_scales, line_scales = _compute_adaptive_scales(
        canonical,
        sample_defs,
        num_joints=num_joints,
        base_joint_scale=joint_scale,
        base_line_scale=line_scale,
        min_joint_scale=min(0.010, joint_scale),
        max_joint_scale=max(joint_scale, 0.035),
        min_line_scale=min(0.006, line_scale),
        max_line_scale=max(line_scale, 0.030),
        gamma=scale_gamma,
        pairs=pairs,
    )
    with torch.no_grad():
        renderer.canonical_scales.copy_(torch.cat([joint_scales, line_scales], dim=0))

    logits = torch.full(
        (canonical_means.shape[0], num_joints),
        -10.0,
        device=canonical.device,
    )
    indices = torch.arange(num_joints, device=canonical.device)
    logits[indices, indices] = 10.0
    logits[num_joints:] = _make_lbs_logits_for_samples(
        num_joints, sample_defs
    ).to(canonical.device)
    renderer.set_lbs_weights_logits(logits)


def setup_cross_topology(
    renderer: DifferentiableSkeletonRenderer,
    weight_path: Path,
    canonical: torch.Tensor,
    line_samples: torch.Tensor,
    sample_defs: list,
    pairs: list[tuple[int, int]],
    num_joints: int,
    joint_scale: float,
    line_scale: float,
    scale_gamma: float,
) -> dict[str, object]:
    """Cross-topology setup (e.g. NTU-trained weights -> HumanML3D).

    Loads the shared, topology-independent trained parameters (appearance head,
    NFM, depth mix), transfers the learned appearance features onto the new
    topology by 1-D interpolation (joint and line banks separately), and sets
    per-bone geometry. Gaussian radii use ``scale_gamma`` so that ``gamma=0``
    gives a uniform radius across all bones (limbs no longer thicken with bone
    length); combined with length-proportional line sampling this yields thin,
    continuous, uniform limbs.
    """
    source = load_file(str(weight_path), device="cpu")
    target = renderer.state_dict()
    shared = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
        and not key.startswith("canonical_") and key != "lbs_weights_logits"
    }
    renderer.load_state_dict(shared, strict=False)

    src_joints = int(source["canonical_joints"].shape[0])
    src_feats = source["canonical_features"].to(device=canonical.device, dtype=torch.float32)
    n_lines = int(line_samples.shape[0])
    joint_feats = F.interpolate(
        src_feats[:src_joints].T.unsqueeze(0), size=num_joints, mode="linear", align_corners=True
    ).squeeze(0).T
    src_line = src_feats[src_joints:]
    if src_line.shape[0] == 0 or n_lines == 0:
        line_feats = joint_feats.mean(dim=0, keepdim=True).expand(max(1, n_lines), -1)
    else:
        line_feats = F.interpolate(
            src_line.T.unsqueeze(0), size=n_lines, mode="linear", align_corners=True
        ).squeeze(0).T
    with torch.no_grad():
        renderer.canonical_features.copy_(torch.cat([joint_feats, line_feats], dim=0))

    means = torch.cat([canonical, line_samples], dim=0)
    renderer.set_canonical_joints(canonical)
    renderer.set_canonical_means(means)

    joint_scales, line_scales = _compute_adaptive_scales(
        canonical, sample_defs, num_joints=num_joints,
        base_joint_scale=joint_scale, base_line_scale=line_scale,
        min_joint_scale=min(0.006, joint_scale), max_joint_scale=joint_scale,
        min_line_scale=min(0.004, line_scale), max_line_scale=line_scale,
        gamma=scale_gamma, pairs=pairs,
    )
    with torch.no_grad():
        renderer.canonical_scales.copy_(torch.cat([joint_scales, line_scales], dim=0))

    logits = torch.full((means.shape[0], num_joints), -10.0, device=canonical.device)
    idx = torch.arange(num_joints, device=canonical.device)
    logits[idx, idx] = 10.0
    logits[num_joints:] = _make_lbs_logits_for_samples(num_joints, sample_defs).to(canonical.device)
    renderer.set_lbs_weights_logits(logits)

    return {
        "exact_topology": False,
        "cross_topology": True,
        "target_joints": num_joints,
        "target_line_gaussians": n_lines,
        "loaded_shared": sorted(shared),
    }


def make_camera(
    frames: int,
    size: int,
    fov_degrees: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    focal = size / (2.0 * math.tan(math.radians(fov_degrees) * 0.5))
    intrinsics = torch.tensor(
        [[focal, 0.0, size / 2.0], [0.0, focal, size / 2.0], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    intrinsics = intrinsics.expand(frames, 3, 3).contiguous()
    world_to_camera = torch.eye(
        4, dtype=torch.float32, device=device
    ).expand(frames, 4, 4).contiguous()
    return intrinsics, world_to_camera


def reposition_humanml3d(
    poses: torch.Tensor,
    cam_distance: float,
    center_xy: bool,
) -> torch.Tensor:
    """View-only repositioning for HumanML3D clips.

    Centers the figure in the frame and places it at ``cam_distance`` from the
    camera so it fills the view. This only affects standalone visualization and
    does not change the training/inference render path.
    """
    valid_mask = poses.abs().sum(dim=(-1, -2)) > 1e-6
    if not valid_mask.any():
        return poses
    valid = poses[valid_mask]
    x_off = -float(valid[..., 0].median()) if center_xy else 0.0
    y_off = -float(valid[..., 1].median()) if center_xy else 0.0
    z_off = cam_distance - float(valid[..., 2].median())
    offsets = poses.new_tensor([x_off, y_off, z_off])
    poses = poses.clone()
    poses[valid_mask] = poses[valid_mask] + offsets
    return poses


def render_file(
    input_path: Path,
    weight_path: Path,
    skeleton_type: str,
    render_frames: int,
    render_size: int,
    num_line_samples: int,
    fov_degrees: float,
    enable_nfm: bool,
    sample_seed: int,
    device: torch.device,
    joint_scale: float = 0.030,
    line_scale: float = 0.020,
    cam_distance: float = 2.0,
    center_xy: bool = True,
    adaptive_base_samples: int = 24,
    sample_beta: float = 1.0,
    scale_gamma: float = 0.0,
) -> tuple[torch.Tensor, list[int], dict[str, object]]:
    poses, metas, orients, detected_type = parse_skeleton_file(str(input_path))
    if detected_type != skeleton_type:
        raise ValueError(
            f"{input_path} is {detected_type}, but --dataset is {skeleton_type}"
        )

    poses = poses.to(device=device, dtype=torch.float32)
    metas = metas.to(device=device, dtype=torch.float32)
    orients = orients.to(device=device, dtype=torch.float32)
    full_frames = poses.shape[0]
    future = torch.clamp(
        torch.arange(full_frames, device=device) + 4, max=full_frames - 1
    )
    full_velocities = poses.index_select(0, future) - poses
    indices = _sample_indices_uniform(
        full_frames, render_frames, device=device, seed=sample_seed
    )
    poses = poses.index_select(0, indices)
    metas = metas.index_select(0, indices)
    orients = orients.index_select(0, indices)
    velocities = full_velocities.index_select(0, indices)

    pairs = get_bone_pairs(skeleton_type)
    if skeleton_type == "humanml3d":
        poses = reposition_humanml3d(poses, cam_distance=cam_distance, center_xy=center_xy)
    else:
        poses = _preprocess_poses_for_rendering(
            poses, do_scale_unify=False, pairs=pairs
        )

    num_joints = get_num_joints(skeleton_type)
    canonical = poses[0, 0]

    def _make_renderer(k_total: int) -> DifferentiableSkeletonRenderer:
        return DifferentiableSkeletonRenderer(
            num_gaussians=k_total,
            num_joints=num_joints,
            feature_dim=3,
            metadata_dim=10,
            H=render_size,
            W=render_size,
            use_gsplat=False,
            temporal_stride=4,
            use_temporal_gru=False,
            use_nn_modulation=True,
            enable_nfm=enable_nfm,
            bone_pairs=pairs,
        ).to(device=device, dtype=torch.float32)

    if skeleton_type == "ntu":
        # Exact topology: the trained NTU weights (uniform Gaussian scales) are
        # loaded verbatim. This is the path the MLLM was trained on; leave it alone.
        renderer = _make_renderer(num_joints + len(pairs) * num_line_samples)
        load_report = load_weights(renderer, weight_path, skeleton_type, num_line_samples)
        forward_line_samples = num_line_samples
        if not load_report["exact_topology"]:
            setup_input_topology(
                renderer, canonical, pairs, num_line_samples,
                joint_scale=joint_scale, line_scale=line_scale,
            )
    else:
        # Cross-topology (e.g. HumanML3D): sample line points proportionally to
        # bone length and use a uniform Gaussian radius (scale_gamma=0), so limbs
        # are thin, continuous, and equally thick regardless of bone length.
        line_samples, sample_defs = _build_line_samples_adaptive(
            canonical, adaptive_base_samples, min_samples=2,
            max_samples=adaptive_base_samples, beta=sample_beta, pairs=pairs,
        )
        renderer = _make_renderer(num_joints + int(line_samples.shape[0]))
        load_report = setup_cross_topology(
            renderer, weight_path, canonical, line_samples, sample_defs, pairs,
            num_joints, joint_scale, line_scale, scale_gamma,
        )
        forward_line_samples = adaptive_base_samples

    intrinsics, world_to_camera = make_camera(
        len(indices), render_size, fov_degrees, device
    )
    renderer.eval()
    with torch.inference_mode():
        rendered = renderer(
            poses=poses,
            metas=metas,
            K=intrinsics,
            w2c=world_to_camera,
            vels=velocities,
            # The released training/inference path uses translation-only
            # skinning; NTU quaternions are parsed but intentionally not used.
            orients=None,
            num_line_samples=forward_line_samples,
        )
    return rendered.squeeze(0), indices.cpu().tolist(), load_report


def save_visualization(
    frames: torch.Tensor,
    output_dir: Path,
    gif_duration_ms: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    array = (
        frames.detach().clamp(0, 1).mul(255).round().byte().cpu().numpy()
    )
    images = [Image.fromarray(frame, mode="RGB") for frame in array]
    for index, image in enumerate(images, start=1):
        image.save(output_dir / f"frame_{index:02d}.png")
    images[0].save(
        output_dir / "animation.gif",
        save_all=True,
        append_images=images[1:],
        duration=gif_duration_ms,
        loop=0,
        disposal=2,
    )


def collect_inputs(
    input_path: Path,
    skeleton_type: str,
    sample_ids: set[str],
    max_samples: int,
) -> list[Path]:
    suffix = ".skeleton" if skeleton_type == "ntu" else ".json"
    if input_path.is_file():
        candidates = [input_path]
    else:
        candidates = sorted(input_path.glob(f"*{suffix}"))
    if sample_ids:
        candidates = [path for path in candidates if path.stem in sample_ids]
    if max_samples > 0:
        candidates = candidates[:max_samples]
    if not candidates:
        raise FileNotFoundError(f"No {suffix} files selected from {input_path}")
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize NTU or HumanML3D skeletons with standalone DrAction weights."
    )
    parser.add_argument("--dataset", choices=["ntu", "humanml3d"], required=True)
    parser.add_argument("--input", type=Path, required=True, help="Skeleton file or directory.")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-ids", default="", help="Optional comma-separated file stems.")
    parser.add_argument("--max-samples", type=int, default=3)
    parser.add_argument("--render-frames", type=int, default=12)
    parser.add_argument("--render-size", type=int, default=448)
    parser.add_argument(
        "--num-line-samples",
        type=int,
        default=None,
        help="Intermediate points sampled per bone. Defaults to the value stored "
        "in the weight metadata (10 for the released candidates). It must match "
        "the weights for NTU exact-topology loading; a mismatch silently falls "
        "back to the cross-topology feature-resize path.",
    )
    parser.add_argument("--fov", type=float, default=60.0)
    parser.add_argument(
        "--joint-scale",
        type=float,
        default=0.026,
        help="Joint Gaussian radius for the cross-topology path (e.g. HumanML3D). Uniform "
        "across bones (scale-gamma=0). Larger = thicker.",
    )
    parser.add_argument(
        "--line-scale",
        type=float,
        default=0.016,
        help="Bone-line Gaussian radius for the cross-topology path (e.g. HumanML3D). Uniform "
        "across bones (scale-gamma=0). Larger = thicker.",
    )
    parser.add_argument(
        "--adaptive-base-samples",
        type=int,
        default=24,
        help="Cross-topology only: max line samples on the longest bone; each bone gets a "
        "count proportional to its length (keeps thin limbs continuous).",
    )
    parser.add_argument(
        "--sample-beta",
        type=float,
        default=1.0,
        help="Cross-topology only: exponent for length-proportional line sampling.",
    )
    parser.add_argument(
        "--scale-gamma",
        type=float,
        default=0.0,
        help="Cross-topology only: Gaussian-radius vs bone-length exponent. 0 = uniform "
        "radius (recommended; legs no longer thicker than torso); 1 = radius proportional to length.",
    )
    parser.add_argument(
        "--cam-distance",
        type=float,
        default=1.8,
        help="HumanML3D only: distance to place the figure; smaller = larger in frame.",
    )
    parser.add_argument(
        "--no-center-xy",
        action="store_false",
        dest="center_xy",
        help="HumanML3D only: disable centering the figure in the frame.",
    )
    parser.set_defaults(center_xy=True)
    parser.add_argument("--gif-duration-ms", type=int, default=120)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="PyTorch CPU threads; 1 avoids overhead in the per-Gaussian rasterizer.",
    )
    parser.add_argument("--enable-nfm", action="store_true", default=None)
    parser.add_argument("--disable-nfm", action="store_false", dest="enable_nfm")
    args = parser.parse_args()

    metadata = weight_metadata(args.weights)
    enable_nfm = args.enable_nfm
    if enable_nfm is None:
        enable_nfm = metadata.get("enable_nfm", "false").lower() == "true"
    num_line_samples = args.num_line_samples
    if num_line_samples is None:
        num_line_samples = int(metadata.get("num_line_samples", 10))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.cpu_threads))

    sample_ids = {item for item in args.sample_ids.split(",") if item}
    inputs = collect_inputs(
        args.input, args.dataset, sample_ids, args.max_samples
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for input_file in tqdm(inputs, desc=f"Rendering {args.dataset}", unit="clip"):
        frames, frame_indices, load_report = render_file(
            input_path=input_file,
            weight_path=args.weights,
            skeleton_type=args.dataset,
            render_frames=args.render_frames,
            render_size=args.render_size,
            num_line_samples=num_line_samples,
            fov_degrees=args.fov,
            enable_nfm=enable_nfm,
            sample_seed=args.sample_seed,
            device=device,
            joint_scale=args.joint_scale,
            line_scale=args.line_scale,
            cam_distance=args.cam_distance,
            center_xy=args.center_xy,
            adaptive_base_samples=args.adaptive_base_samples,
            sample_beta=args.sample_beta,
            scale_gamma=args.scale_gamma,
        )
        sample_dir = args.output_dir / input_file.stem
        save_visualization(frames, sample_dir, args.gif_duration_ms)
        report = {
            "sample_id": input_file.stem,
            "dataset": args.dataset,
            "weight_name": metadata.get("name", args.weights.stem),
            "weight_metric": metadata.get("metric", ""),
            "render_frames": args.render_frames,
            "render_size": args.render_size,
            "frame_indices": frame_indices,
            "enable_nfm": enable_nfm,
            "load_report": load_report,
        }
        (sample_dir / "render.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(f"Saved {len(inputs)} visualization(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
