
import warnings
import math
from contextlib import nullcontext
from typing import List, Optional, Tuple, Union, Any, Dict

import torch
import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from skeletonllm.conversation import get_conv_template
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from .draction import (
    DifferentiableSkeletonRenderer,
    NTU_PAIRS,
    get_bone_pairs,
    get_num_joints,
    parse_skeleton_file,
    _build_line_samples,
    _build_line_samples_adaptive,
    _compute_adaptive_scales,
    _make_lbs_logits_for_samples,
    _preprocess_humanml3d_poses,
    _preprocess_poses_for_rendering,
    _sample_indices_uniform,
)
import os

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class Skeleton3DGSRendererModule(nn.Module):
    """Render NTU or HumanML3D skeleton files with trainable DrAction modules.

    Inputs: one path or a list of `.skeleton`/converted HumanML3D `.json` paths.
    Output: Tensor (B, T, H, W, 3) in [0,1], differentiable w.r.t renderer parameters.
    """
    def __init__(self, H: int, W: int, num_line_samples: int = 10, target_num_frames: int = 16,
                 fovx_deg: float = 60.0, fovy_deg: float = 60.0, 
                 disable_autocast: bool = False,
                 enable_nfm: bool = False):  # 🆕 Enable NFM network (default False for stability)
        super().__init__()
        self.H = int(H)
        self.W = int(W)
        self.num_line_samples = int(num_line_samples)
        self.target_num_frames = int(target_num_frames)
        self.fovx_deg = float(fovx_deg)
        self.fovy_deg = float(fovy_deg)
        self.disable_autocast = bool(disable_autocast)
        self.enable_nfm = bool(enable_nfm)  # 🆕
        self._renderer: Optional[DifferentiableSkeletonRenderer] = None
        self._renderer_humanml3d: Optional[DifferentiableSkeletonRenderer] = None
        self._metadata_dim = 10
        self._feature_dim = 3
        self._trainable = True
        # Cross-topology (e.g. HumanML3D) line-sampling budget: samples are drawn
        # proportionally to bone length up to this many on the longest bone.
        self._humanml3d_base_samples = 24

    def _ensure_renderer(
        self,
        device: torch.device,
        skeleton_type: str = 'ntu',
    ) -> DifferentiableSkeletonRenderer:
        attr_name = '_renderer' if skeleton_type == 'ntu' else '_renderer_humanml3d'
        renderer = getattr(self, attr_name)
        pairs = get_bone_pairs(skeleton_type)
        num_joints = get_num_joints(skeleton_type)
        if renderer is None:
            k_total = num_joints + self.num_line_samples * len(pairs)
            renderer = DifferentiableSkeletonRenderer(
                num_gaussians=k_total,
                num_joints=num_joints,
                feature_dim=self._feature_dim,
                metadata_dim=self._metadata_dim,
                H=self.H,
                W=self.W,
                use_gsplat=True,
                temporal_stride=4,
                use_temporal_gru=False,  # Disabled for stability
                use_nn_modulation=True,
                enable_nfm=self.enable_nfm,  # 🆕 Pass enable_nfm parameter
                bone_pairs=pairs,
            ).to(device=device, dtype=torch.float32)
            renderer.requires_grad_(self._trainable)
            setattr(self, attr_name, renderer)
        else:
            # When moving to device, ensure we stay in float32
            renderer = renderer.to(device=device, dtype=torch.float32)
            # Re-ensure appearance_head and nfm are in float32 (no hooks, simple conversion)
            if renderer.appearance_head is not None:
                renderer.appearance_head = renderer.appearance_head.to(dtype=torch.float32)
            if renderer.nfm is not None:
                renderer.nfm = renderer.nfm.to(dtype=torch.float32)
            if renderer.temporal_gru is not None:
                renderer.temporal_gru = renderer.temporal_gru.to(dtype=torch.float32)
            setattr(self, attr_name, renderer)
        return renderer

    def _transfer_trained_appearance(self, renderer: DifferentiableSkeletonRenderer,
                                     n_lines: int, num_joints: int, device: torch.device) -> None:
        """Copy the trained appearance (appearance head, NFM, depth mix) from the
        NTU renderer onto a cross-topology renderer, interpolating the learned
        canonical features (joint and line banks separately) to the new topology.
        Guarded: if the NTU renderer is unavailable/untrained, the cross-topology
        renderer keeps its random init (rendering still runs)."""
        import torch.nn.functional as F
        src = self._renderer
        if src is None:
            return
        try:
            renderer.appearance_head.load_state_dict(src.appearance_head.state_dict())
            if renderer.nfm is not None and getattr(src, 'nfm', None) is not None:
                renderer.nfm.load_state_dict(src.nfm.state_dict())
            with torch.no_grad():
                renderer.depth_mix_logit.copy_(src.depth_mix_logit.to(device))
                sf = src.canonical_features.detach().to(device=device, dtype=torch.float32)
                sj = int(src.canonical_joints.shape[0])
                joint_feats = F.interpolate(sf[:sj].T.unsqueeze(0), size=num_joints,
                                            mode='linear', align_corners=True).squeeze(0).T
                src_line = sf[sj:]
                if src_line.shape[0] == 0 or n_lines == 0:
                    line_feats = joint_feats.mean(dim=0, keepdim=True).expand(max(1, n_lines), -1)
                else:
                    line_feats = F.interpolate(src_line.T.unsqueeze(0), size=n_lines,
                                               mode='linear', align_corners=True).squeeze(0).T
                renderer.canonical_features.copy_(torch.cat([joint_feats, line_feats], dim=0))
        except Exception as exc:  # pragma: no cover - defensive
            import logging
            logging.getLogger(__name__).warning(f"[HumanML3D] trained-appearance transfer failed: {exc}")

    @staticmethod
    def _autocast_disabled(device: torch.device):
        device_type = device.type if isinstance(device, torch.device) else str(device)
        if isinstance(device_type, str):
            device_type = device_type.split(':', 1)[0]
        if device_type == 'cuda':
            return torch.cuda.amp.autocast(enabled=False)
        if device_type == 'cpu':
            try:
                return torch.cpu.amp.autocast(enabled=False)  # type: ignore[attr-defined]
            except AttributeError:
                return nullcontext()
        try:
            return torch.autocast(device_type=device_type, enabled=False)
        except (AttributeError, TypeError):
            return nullcontext()

    def set_trainable(self, trainable: bool):
        """
        Enable or disable gradients for the renderer parameters.
        """
        flag = bool(trainable)
        self._trainable = flag
        for param in self.parameters():
            param.requires_grad_(flag)

    def _make_cameras(self, T_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        fovx = math.radians(self.fovx_deg)
        fovy = math.radians(self.fovy_deg)
        fx = self.W / (2.0 * math.tan(fovx * 0.5))
        fy = self.H / (2.0 * math.tan(fovy * 0.5))
        cx, cy = self.W / 2.0, self.H / 2.0
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=dtype, device=device)
        K = K.expand(T_len, 3, 3).contiguous()
        w2c = torch.eye(4, dtype=dtype, device=device).expand(T_len, 4, 4).contiguous()
        return K, w2c

    def _sample_frames(self, poses: torch.Tensor, metas: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample frames and compute velocities on ORIGINAL frames before sampling.
        
        Returns:
            poses_sampled: (T_target, P, J, 3)
            metas_sampled: (T_target, P, metadata_dim)
            vels_sampled: (T_target, P, J, 3) velocities computed from original sequence
        """
        T_len = poses.shape[0]
        P, J = poses.shape[1], poses.shape[2]
        
        # 🔧 Compute velocities on ORIGINAL frames (before sampling)
        # For each frame, compute difference with frame at t+4 (clamped at boundary)
        stride = 4
        idx_fut = torch.clamp(torch.arange(T_len, device=device) + stride, max=T_len - 1)
        vels_full = poses.index_select(0, idx_fut) - poses  # (T_full, P, J, 3)
        
        if self.target_num_frames is not None:
            idx = _sample_indices_uniform(T_len, self.target_num_frames, device=device, seed=None)
            poses = poses.index_select(0, idx)
            metas = metas.index_select(0, idx)
            vels = vels_full.index_select(0, idx)  # Sample velocities too
        else:
            vels = vels_full
        
        return poses, metas, vels

    def forward(self, paths: Union[str, List[str]], device: Optional[torch.device] = None) -> torch.Tensor:
        if isinstance(paths, str):
            paths = [paths]
        assert isinstance(paths, (list, tuple)) and len(paths) > 0
        if device is None:
            device = self._renderer.depth_mix_logit.device if self._renderer is not None else torch.device('cpu')
        outputs = []
        # 🔧 CRITICAL FIX: Always disable autocast for renderer to avoid NaN in bf16
        autocast_guard = self._autocast_disabled(device)
        with autocast_guard:
            for path in paths:
                poses, metas, _, skeleton_type = parse_skeleton_file(path)
                renderer = self._ensure_renderer(device, skeleton_type=skeleton_type)
                pairs = get_bone_pairs(skeleton_type)
                poses = poses.to(device=device, dtype=torch.float32)
                metas = metas.to(device=device, dtype=torch.float32)
                poses, metas, vels = self._sample_frames(poses, metas, device)
                if skeleton_type == 'humanml3d':
                    poses = _preprocess_humanml3d_poses(poses)
                    canonical = poses[0, 0]
                    num_joints = canonical.shape[0]
                    # Cross-topology rendering: sample line points proportionally to
                    # bone length and use a uniform Gaussian radius (gamma=0) so limbs
                    # stay thin and equally thick regardless of bone length. Build a
                    # fresh renderer sized to the adaptive sample count and transfer the
                    # trained appearance from the NTU renderer (self._renderer).
                    line_samples, sample_defs = _build_line_samples_adaptive(
                        canonical, self._humanml3d_base_samples, min_samples=2,
                        max_samples=self._humanml3d_base_samples, beta=1.0, pairs=pairs,
                    )
                    n_lines = int(line_samples.shape[0])
                    renderer = DifferentiableSkeletonRenderer(
                        num_gaussians=num_joints + n_lines, num_joints=num_joints,
                        feature_dim=self._feature_dim, metadata_dim=self._metadata_dim,
                        H=self.H, W=self.W, use_gsplat=True, temporal_stride=4,
                        use_temporal_gru=False, use_nn_modulation=True,
                        enable_nfm=self.enable_nfm, bone_pairs=pairs,
                    ).to(device=device, dtype=torch.float32)
                    renderer.requires_grad_(self._trainable)
                    self._transfer_trained_appearance(renderer, n_lines, num_joints, device)
                    canonical_means = torch.cat([canonical, line_samples], dim=0)
                    renderer.set_canonical_joints(canonical)
                    renderer.set_canonical_means(canonical_means)
                    joint_scales, line_scales = _compute_adaptive_scales(
                        canonical, sample_defs, num_joints=num_joints,
                        base_joint_scale=0.026, base_line_scale=0.016,
                        min_joint_scale=0.006, max_joint_scale=0.026,
                        min_line_scale=0.004, max_line_scale=0.016,
                        gamma=0.0, pairs=pairs,
                    )
                    with torch.no_grad():
                        renderer.canonical_scales.copy_(
                            torch.cat([joint_scales, line_scales], dim=0)
                        )
                    logits = torch.full(
                        (canonical_means.shape[0], num_joints),
                        -10.0,
                        device=device,
                    )
                    joint_idx = torch.arange(num_joints, device=device)
                    logits[joint_idx, joint_idx] = 10.0
                    logits[num_joints:] = _make_lbs_logits_for_samples(
                        num_joints, sample_defs
                    ).to(device)
                    renderer.set_lbs_weights_logits(logits)
                    forward_num_line_samples = self._humanml3d_base_samples
                else:
                    poses = _preprocess_poses_for_rendering(
                        poses,
                        root_idx=1,
                        target_bone_len=0.3,
                        do_scale_unify=False,
                        pairs=NTU_PAIRS,
                    )
                    forward_num_line_samples = self.num_line_samples
                T_len = poses.shape[0]
                K, w2c = self._make_cameras(T_len, device=device, dtype=torch.float32)
                # Ensure renderer runs entirely in float32
                frames = renderer(
                    poses=poses,
                    metas=metas,
                    K=K,
                    w2c=w2c,
                    vels=vels,  # Pass pre-computed velocities
                    num_line_samples=forward_num_line_samples,
                ).to(dtype=torch.float32)
                outputs.append(frames)
        if len(outputs) == 1:
            return outputs[0]
        return torch.cat(outputs, dim=0)

class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    # Use input_ids as the main input for FLOPs estimation to avoid Trainer calling
    # .numel() on non-tensor pixel_values (we render skeletons in-model).
    main_input_name = 'input_ids'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'Qwen2DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config._attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

        # Skeleton rendering integration
        self.use_skeleton = bool(config.use_skeleton)
        self._skeleton_renderer_module: Optional[nn.Module] = None
        self._skeleton_renderer_trainable: bool = bool(getattr(config, 'skeleton_renderer_trainable', True))
        # Do NOT eagerly instantiate at init, since force_image_size can be
        # changed by training script after model load. We'll lazily create
        # (or rebuild) the renderer at first use to ensure H/W match.

        # log-once guards
        self._log_once_vit_batch = False
        self._log_once_fill_warning = False
        self._log_once_label_info = False

        # If using skeletons, eagerly instantiate the renderer (and its internal differentiable renderer)
        # so that checkpoint weights under `_skeleton_renderer_module._renderer.*` can be loaded.
        if self.use_skeleton:
            try:
                renderer_mod = self._get_skeleton_renderer()
                # Ensure the inner renderer exists to match checkpoint parameter names
                if hasattr(renderer_mod, '_ensure_renderer'):
                    inner = renderer_mod._ensure_renderer(self.device)
                    # Keep inner renderer in float32 for numerical stability
                    try:
                        renderer_mod._renderer = inner.to(device=self.device, dtype=torch.float32)
                    except Exception:
                        pass
            except Exception:
                # Defer to lazy creation at first use if anything goes wrong
                pass

    def wrap_backbone_lora(self, r: int = 128, lora_alpha: int = 256, lora_dropout: float = 0.05):
        """Apply LoRA adapters to the ViT backbone (vision model)."""
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r: int = 128, lora_alpha: int = 256, lora_dropout: float = 0.05):
        """Apply LoRA adapters to the language model with architecture-specific target modules."""
        from peft import LoraConfig, get_peft_model

        llm_arch_name = self.config.llm_config.architectures[0] if hasattr(self.config.llm_config, 'architectures') else ''
        if llm_arch_name == 'LlamaForCausalLM' or llm_arch_name == 'Qwen2ForCausalLM':
            target_modules = [
                'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj'
            ]
        else:
            raise NotImplementedError(f'LoRA target modules not configured for arch: {llm_arch_name}')

        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        # Allow LoRA to receive grads through inputs
        if hasattr(self.language_model, 'enable_input_require_grads'):
            self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def _get_skeleton_image_size(self) -> Tuple[int, int]:
        size = self.config.force_image_size or self.config.vision_config.image_size
        return int(size), int(size)

    def set_skeleton_renderer_trainable(self, trainable: bool, apply_immediately: bool = True):
        """
        Toggle whether the skeleton renderer's parameters receive gradients.

        Args:
            trainable: If True, renderer parameters require grad; otherwise they are frozen.
            apply_immediately: If True, update the existing renderer module (if instantiated).
        """
        self._skeleton_renderer_trainable = bool(trainable)
        if apply_immediately:
            self._apply_skeleton_renderer_trainable_state()

    def _apply_skeleton_renderer_trainable_state(self):
        if self._skeleton_renderer_module is None:
            return
        setter = getattr(self._skeleton_renderer_module, 'set_trainable', None)
        if callable(setter):
            setter(self._skeleton_renderer_trainable)
        else:
            try:
                self._skeleton_renderer_module.requires_grad_(self._skeleton_renderer_trainable)
            except Exception:
                pass
            inner = getattr(self._skeleton_renderer_module, '_renderer', None)
            if isinstance(inner, nn.Module):
                for param in inner.parameters():
                    param.requires_grad_(self._skeleton_renderer_trainable)

    def _get_skeleton_renderer(self) -> nn.Module:
        H, W = self._get_skeleton_image_size()
        enable_nfm = bool(getattr(self.config, 'skeleton_enable_nfm', True))  # 🆕 Get from config
        if self._skeleton_renderer_module is None:
            self._skeleton_renderer_module = Skeleton3DGSRendererModule(
                H=H,
                W=W,
                num_line_samples=int(self.config.skeleton_num_line_samples),
                target_num_frames=int(self.config.skeleton_target_num_frames),
                fovx_deg=float(self.config.skeleton_fovx_deg),
                fovy_deg=float(self.config.skeleton_fovy_deg),
                enable_nfm=enable_nfm,  # 🆕 Pass enable_nfm parameter
            )
            # Keep renderer in float32; cast frames to vision dtype later
            self._skeleton_renderer_module = self._skeleton_renderer_module.to(device=self.device, dtype=torch.float32)
        else:
            # Rebuild renderer if force_image_size changed (H/W mismatch)
            try:
                cur_H = int(getattr(self._skeleton_renderer_module, 'H', H))
                cur_W = int(getattr(self._skeleton_renderer_module, 'W', W))
            except Exception:
                cur_H, cur_W = H, W
            if cur_H != H or cur_W != W:
                old_state = {k: v for k, v in self._skeleton_renderer_module.state_dict().items()}
                new_mod = Skeleton3DGSRendererModule(
                    H=H,
                    W=W,
                    num_line_samples=int(self.config.skeleton_num_line_samples),
                    target_num_frames=int(self.config.skeleton_target_num_frames),
                    fovx_deg=float(self.config.skeleton_fovx_deg),
                    fovy_deg=float(self.config.skeleton_fovy_deg),
                    enable_nfm=enable_nfm,
                )
                new_mod = new_mod.to(device=self.device, dtype=torch.float32)
                # Best-effort load to preserve compatible params
                try:
                    new_mod.load_state_dict(old_state, strict=False)
                except Exception:
                    pass
                self._skeleton_renderer_module = new_mod
            else:
                # Ensure device sync if model moved
                self._skeleton_renderer_module = self._skeleton_renderer_module.to(device=self.device, dtype=torch.float32)
        self._apply_skeleton_renderer_trainable_state()
        return self._skeleton_renderer_module

    def _inject_visual_embeds(
        self,
        input_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        vit_embeds: torch.Tensor,
        image_flags: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Differentiably inject visual embeddings into positions of IMG_CONTEXT tokens.

        - input_embeds: (B, N, C)
        - input_ids: (B, N)
        - vit_embeds: (I, T, C) or (K, C) where K = I*T
        - image_flags: (I,) 1 for real images to keep, optional
        """
        B, N, C = input_embeds.shape
        device = input_embeds.device

        # Prepare visual features
        if vit_embeds is None:
            return input_embeds
        if vit_embeds.dim() == 3:
            # Optionally filter by image_flags (keep where flag==1)
            if image_flags is not None:
                try:
                    vit_embeds = vit_embeds[image_flags == 1]
                except Exception:
                    pass
            vit_flat = vit_embeds.reshape(-1, C)
        elif vit_embeds.dim() == 2:
            vit_flat = vit_embeds
        else:
            # Unsupported shape; no-op
            return input_embeds

        # Compute selected token indices
        selected = (input_ids == self.img_context_token_id)
        if selected.dim() == 2:
            selected_flat = selected.reshape(B * N)
        else:
            selected_flat = selected
        sel_idx = torch.nonzero(selected_flat, as_tuple=False).squeeze(-1)

        if sel_idx.numel() == 0 or vit_flat.numel() == 0:
            return input_embeds

        # Match counts (partial fill if mismatched)
        k = int(min(sel_idx.numel(), vit_flat.shape[0]))
        sel_idx = sel_idx[:k]
        vit_src = vit_flat[:k].to(device)

        flat_in = input_embeds.reshape(B * N, C)
        # Build update tensor in a way that preserves gradients for vit_src
        base = torch.zeros_like(flat_in)
        try:
            updated = base.index_put((sel_idx, slice(None)), vit_src, accumulate=False)
        except Exception:
            idx_expand = sel_idx.unsqueeze(-1).expand(-1, C)
            updated = base.scatter(0, idx_expand, vit_src)
        mask_bool = torch.zeros(flat_in.shape[0], dtype=torch.bool, device=device)
        mask_bool[sel_idx] = True
        mask = mask_bool.unsqueeze(-1).to(flat_in.dtype)
        flat_out = flat_in * (1.0 - mask) + updated
        return flat_out.view(B, N, C)

    def _maybe_render_skeleton_input(self, pixel_values: Any):
        if not self.use_skeleton or pixel_values is None:
            return pixel_values
        if torch.is_tensor(pixel_values):
            return pixel_values
        # Accept str or list[str]
        if isinstance(pixel_values, str):
            paths = [pixel_values]
        elif isinstance(pixel_values, (list, tuple)) and len(pixel_values) > 0 and isinstance(pixel_values[0], str):
            paths = list(pixel_values)
        else:
            return pixel_values

        renderer = self._get_skeleton_renderer()
        frames = renderer(paths, device=self.device)

        # 🔍 DEBUG: Check renderer output statistics
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            with torch.no_grad():
                logger.info(f"[RENDER_DEBUG] frames.shape={frames.shape}, "
                           f"min={frames.min().item():.4f}, max={frames.max().item():.4f}, "
                           f"mean={frames.mean().item():.4f}, std={frames.std().item():.4f}, "
                           f"requires_grad={frames.requires_grad}, dtype={frames.dtype}")
                
                # 💾 Save rendered frames as images (first batch only)
                if not hasattr(self, '_saved_render_images'):
                    try:
                        import os
                        from PIL import Image
                        import numpy as np
                        
                        save_dir = "debug_renders"
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # frames shape: (B, T, H, W, 3) in [0, 1]
                        frames_np = frames.cpu().float().numpy()
                        B, T = frames_np.shape[0], frames_np.shape[1]
                        
                        for b in range(min(B, 2)):  # Save first 2 samples
                            for t_idx in range(min(T, 4)):  # Save first 4 frames per sample
                                frame = frames_np[b, t_idx]  # (H, W, 3)
                                frame_uint8 = (frame * 255).clip(0, 255).astype(np.uint8)
                                img = Image.fromarray(frame_uint8)
                                img_path = os.path.join(save_dir, f"batch0_sample{b}_frame{t_idx}.png")
                                img.save(img_path)
                        
                        logger.info(f"[RENDER_SAVE] Saved rendered images to {save_dir}/")
                        self._saved_render_images = True
                    except Exception as e:
                        logger.warning(f"[RENDER_SAVE] Failed to save images: {e}")
                        self._saved_render_images = True  # Don't retry

        # Cast to vision dtype for Conv2D
        embed_dtype = self.vision_model.embeddings.patch_embedding.weight.dtype
        frames = frames.to(device=self.device, dtype=embed_dtype)
        
        # 🔍 DEBUG: Check if gradient flow is preserved after dtype conversion
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            logger.info(f"[RENDER_DEBUG] After dtype conversion: requires_grad={frames.requires_grad}, dtype={frames.dtype}")
        
        return frames

    def _flatten_pixel_values_for_vit(self, pixel_values):
        """
        Normalize pixel_values to NCHW 4D for ViT and infer per-sample frame counts.

        Supports shapes:
          - (B, T, H, W, C) channels-last video
          - (B, T, C, H, W) channels-first video
          - (T, H, W, C) single-sample channels-last video
          - (T, C, H, W) single-sample channels-first video
          - (B, H, W, C) batch of images, channels-last
          - (B, C, H, W) batch of images, channels-first
          - (H, W, C) single image channels-last
          - (C, H, W) single image channels-first

        Returns:
          flat_images: (N_images, 3, H, W)
          num_patches_list: list[int] length B (frames per sample); empty list if batch size unknown
          effective_num_image_token: tokens per image after pixel shuffle given H, W, patch_size, downsample_ratio
        """
        if pixel_values is None:
            return None, [], self.num_image_token

        dims = pixel_values.dim()
        device = pixel_values.device
        dtype = pixel_values.dtype

        def to_nchw(x, c_last: bool):
            return x.permute(0, 3, 1, 2).contiguous() if c_last else x

        if dims == 5:
            # (B, T, H, W, C) or (B, T, C, H, W)
            B, T = pixel_values.shape[0], pixel_values.shape[1]
            # Heuristic: channels-last if last dim in {1,3,4}
            channels_last = pixel_values.shape[-1] in (1, 3, 4)
            if channels_last:
                _, _, H, W, C = pixel_values.shape
                x = pixel_values.reshape(B * T, H, W, C)
                x = to_nchw(x, c_last=True)
            else:
                _, _, C, H, W = pixel_values.shape
                x = pixel_values.reshape(B * T, C, H, W)

            if x.shape[1] != 3:
                raise ValueError(f'Expected 3 channels, got {x.shape[1]} for video input.')

            flat_images = x.to(device=device, dtype=dtype)
            num_patches_list = [T for _ in range(B)]

        elif dims == 4:
            # (B, H, W, C) or (B, C, H, W)
            channels_last = pixel_values.shape[-1] in (1, 3, 4)
            if channels_last:
                B, H, W, C = pixel_values.shape
                x = to_nchw(pixel_values, c_last=True)
            else:
                B, C, H, W = pixel_values.shape
                x = pixel_values

            if x.shape[1] != 3:
                raise ValueError(f'Expected 3 channels, got {x.shape[1]} for image batch input.')

            flat_images = x.to(device=device, dtype=dtype)
            # Ambiguous how many samples; assume single query with B images (chat) or require num_patches_list in batch_chat
            num_patches_list = [B]

        elif dims == 3:
            # (H, W, C) or (C, H, W)
            channels_last = pixel_values.shape[-1] in (1, 3, 4)
            if channels_last:
                H, W, C = pixel_values.shape
                x = pixel_values.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
            else:
                C, H, W = pixel_values.shape
                x = pixel_values.unsqueeze(0)

            if x.shape[1] != 3:
                raise ValueError(f'Expected 3 channels, got {x.shape[1]} for single image input.')

            flat_images = x.to(device=device, dtype=dtype)
            num_patches_list = [1]

        else:
            raise ValueError(f'Unsupported pixel_values dims: {pixel_values.shape}')

        # Compute tokens per image dynamically from H, W
        H, W = flat_images.shape[-2], flat_images.shape[-1]
        patch = self.config.vision_config.patch_size
        h_grid = H // patch
        w_grid = W // patch
        effective_num_image_token = int(h_grid * self.downsample_ratio) * int(w_grid * self.downsample_ratio)
        return flat_images, num_patches_list, max(effective_num_image_token, 1)

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Skeleton support: accept list[str] or str paths and render to tensor
        pixel_values = self._maybe_render_skeleton_input(pixel_values)

        if image_flags is not None:
            try:
                image_flags = image_flags.reshape(-1)
            except Exception:
                pass
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        vit_embeds = self.extract_feature(pixel_values)
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0 and not self._log_once_vit_batch:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
            self._log_once_vit_batch = True

        # 🔍 DEBUG: Check vit_embeds statistics
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            with torch.no_grad():
                logger.info(f"[VIT_DEBUG] vit_embeds.shape={vit_embeds.shape}, "
                           f"min={vit_embeds.min().item():.4f}, max={vit_embeds.max().item():.4f}, "
                           f"mean={vit_embeds.mean().item():.4f}, std={vit_embeds.std().item():.4f}, "
                           f"requires_grad={vit_embeds.requires_grad}")

        # Differentiable visual token injection
        input_embeds = self._inject_visual_embeds(input_embeds, input_ids, vit_embeds, image_flags)

        # Log-once diagnostics to catch all-ignored labels or injection issues
        if (not self._log_once_label_info) and (labels is not None):
            try:
                num_valid_labels = int((labels != -100).sum().item())
            except Exception:
                num_valid_labels = -1
            try:
                num_img_ctx = int((input_ids == self.img_context_token_id).sum().item()) if self.img_context_token_id is not None else -1
            except Exception:
                num_img_ctx = -1
            try:
                vit_token_count = int(vit_embeds.shape[0] * vit_embeds.shape[1]) if vit_embeds is not None else -1
            except Exception:
                vit_token_count = -1
            if num_valid_labels == 0:
                logger.warning(f'First batch has 0 valid labels (all IGNORE_INDEX). img_ctx_tokens={num_img_ctx}, vit_tokens={vit_token_count}')
            else:
                logger.info(f'First batch valid label tokens: {num_valid_labels}')
            self._log_once_label_info = True

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Robust FLOPs estimation when pixel_values are paths/lists pre-rendering
    def floating_point_ops(self, input_dict: Dict[str, Any], exclude_embeddings: bool = True) -> float:
        try:
            return super().floating_point_ops(input_dict, exclude_embeddings=exclude_embeddings)
        except Exception:
            # Fallback to count tokens from input_ids
            ids = input_dict.get('input_ids', None)
            try:
                return 6 * float(ids.numel()) * float(self.num_parameters(exclude_embeddings=exclude_embeddings)) if ids is not None else 0.0
            except Exception:
                return 0.0

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        # Normalize to 4D (N,3,H,W) if 5D/other accepted shapes are provided
        pixel_values = self._maybe_render_skeleton_input(pixel_values)
        flat_images, _, _ = self._flatten_pixel_values_for_vit(pixel_values)
        pixel_values_4d = flat_images if flat_images is not None else pixel_values

        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values_4d,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values_4d,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        # Always try to render skeleton early to avoid list/shape issues
        try:
            pixel_values = self._maybe_render_skeleton_input(pixel_values)
        except Exception:
            pass

        # Infer frames-per-sample and dynamic tokens if 5D input provided
        effective_img_tokens = self.num_image_token
        flat_images = None
        if pixel_values is not None:
            try:
                flat_images, inferred_patches_list, effective_img_tokens = self._flatten_pixel_values_for_vit(pixel_values)
            except Exception:
                flat_images, inferred_patches_list, effective_img_tokens = None, None, self.num_image_token
            if inferred_patches_list is not None and len(inferred_patches_list) > 0:
                # If caller didn't specify, adopt inferred list
                if num_patches_list is None:
                    num_patches_list = inferred_patches_list
                else:
                    # Sanity check length
                    if len(num_patches_list) != len(questions):
                        print('warning: num_patches_list length does not match number of questions; using inferred list.')
                        num_patches_list = inferred_patches_list

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None and hasattr(pixel_values, 'shape'):
            try:
                image_bs = pixel_values.shape[0]
                print(f'dynamic ViT batch size: {image_bs}')
            except Exception:
                pass

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * (effective_img_tokens * num_patches) + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=flat_images if flat_images is not None else pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        effective_img_tokens = self.num_image_token
        flat_images = None
        inferred_patches_list = None
        if pixel_values is not None:
            try:
                flat_images, inferred_patches_list, effective_img_tokens = self._flatten_pixel_values_for_vit(pixel_values)
            except Exception:
                flat_images, inferred_patches_list, effective_img_tokens = None, None, self.num_image_token

        if num_patches_list is None:
            if inferred_patches_list is not None and len(inferred_patches_list) > 0:
                num_patches_list = inferred_patches_list
            else:
                if pixel_values is not None and hasattr(pixel_values, 'shape'):
                    try:
                        num_patches_list = [pixel_values.shape[0]]
                    except Exception:
                        num_patches_list = [1]
                else:
                    num_patches_list = [1]
        # Cannot assert shapes safely because we may have flattened inside helper; rely on helper correctness

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None and hasattr(pixel_values, 'shape'):
            try:
                image_bs = pixel_values.shape[0]
                print(f'dynamic ViT batch size: {image_bs}')
            except Exception:
                pass

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * (effective_img_tokens * num_patches) + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=flat_images if flat_images is not None else pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                # Accept 5D/other shapes by normalizing before feature extraction
                pixel_values = self._maybe_render_skeleton_input(pixel_values)
                flat_images, _, _ = self._flatten_pixel_values_for_vit(pixel_values)
                vit_embeds = self.extract_feature(flat_images if flat_images is not None else pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            # Differentiable injection (no image_flags used during generation)
            input_embeds = self._inject_visual_embeds(input_embeds, input_ids, vit_embeds, image_flags=None)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
