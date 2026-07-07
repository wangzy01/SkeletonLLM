
import json
import math
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Kinect v2 / NTU RGB+D topology (1-based joint indices).
NTU_PAIRS: List[Tuple[int, int]] = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7),
    (9, 21), (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14),
    (16, 15), (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (23, 8),
    (24, 25), (25, 12)
]

# SMPL / HumanML3D topology (1-based joint indices).
HUMANML3D_PAIRS: List[Tuple[int, int]] = [
    (1, 3), (3, 6), (6, 9), (9, 12),
    (1, 2), (2, 5), (5, 8), (8, 11),
    (1, 4), (4, 7), (7, 10), (10, 13), (13, 16),
    (10, 15), (15, 18), (18, 20), (20, 22),
    (10, 14), (14, 17), (17, 19), (19, 21),
]

NTU_NUM_JOINTS = 25
HUMANML3D_NUM_JOINTS = 22

# Backward-compatible name used by existing NTU checkpoints and scripts.
PAIRS = NTU_PAIRS


# 2) Quaternion helpers (w, x, y, z)
def _normalize_quat(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return q / (q.norm(dim=-1, keepdim=True) + eps)


def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    q = _normalize_quat(q)
    w, x, y, z = q.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack([
        ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz
    ], dim=-1).reshape(q.shape[:-1] + (3, 3))
    return R


def _mat_to_quat(R: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m00, m01, m02 = R[...,0,0], R[...,0,1], R[...,0,2]
    m10, m11, m12 = R[...,1,0], R[...,1,1], R[...,1,2]
    m20, m21, m22 = R[...,2,0], R[...,2,1], R[...,2,2]
    trace = m00 + m11 + m22

    qw = torch.zeros_like(trace)
    qx = torch.zeros_like(trace)
    qy = torch.zeros_like(trace)
    qz = torch.zeros_like(trace)

    cond0 = trace > 0
    S0 = torch.sqrt(torch.clamp(trace[cond0] + 1.0, min=eps)) * 2
    qw[cond0] = 0.25 * S0
    qx[cond0] = (m21[cond0] - m12[cond0]) / S0
    qy[cond0] = (m02[cond0] - m20[cond0]) / S0
    qz[cond0] = (m10[cond0] - m01[cond0]) / S0

    cond1 = (~cond0) & (m00 >= m11) & (m00 >= m22)
    S1 = torch.sqrt(torch.clamp(1.0 + m00[cond1] - m11[cond1] - m22[cond1], min=eps)) * 2
    qw[cond1] = (m21[cond1] - m12[cond1]) / S1
    qx[cond1] = 0.25 * S1
    qy[cond1] = (m01[cond1] + m10[cond1]) / S1
    qz[cond1] = (m02[cond1] + m20[cond1]) / S1

    cond2 = (~cond0) & (~cond1) & (m11 >= m22)
    S2 = torch.sqrt(torch.clamp(1.0 + m11[cond2] - m00[cond2] - m22[cond2], min=eps)) * 2
    qw[cond2] = (m02[cond2] - m20[cond2]) / S2
    qx[cond2] = (m01[cond2] + m10[cond2]) / S2
    qy[cond2] = 0.25 * S2
    qz[cond2] = (m12[cond2] + m21[cond2]) / S2

    cond3 = (~cond0) & (~cond1) & (~cond2)
    S3 = torch.sqrt(torch.clamp(1.0 + m22[cond3] - m00[cond3] - m11[cond3], min=eps)) * 2
    qw[cond3] = (m10[cond3] - m01[cond3]) / S3
    qx[cond3] = (m02[cond3] + m20[cond3]) / S3
    qy[cond3] = (m12[cond3] + m21[cond3]) / S3
    qz[cond3] = 0.25 * S3

    q = torch.stack([qw, qx, qy, qz], dim=-1)
    return _normalize_quat(q)


def _project_to_rotation(R: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(R)
    R_ = U @ Vh
    det = torch.det(R_)
    fix = det < 0
    if fix.any():
        Vh[fix, -1, :] *= -1
        R_ = U @ Vh
    return R_


def _compose_T(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    T = torch.eye(4, device=R.device, dtype=R.dtype).expand(R.shape[:-2] + (4, 4)).clone()
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T


# 3) Minimal differentiable renderer (simplified)
class DifferentiableSkeletonRenderer(nn.Module):
    def __init__(self, num_gaussians: int, num_joints: int, feature_dim: int, metadata_dim: int, H: int, W: int,
                 use_gsplat: bool = True,
                 temporal_stride: int = 4,
                 use_temporal_gru: bool = False,  # Default False for stability
                 use_nn_modulation: bool = True,
                 enable_nfm: bool = False,
                 bone_pairs: Optional[List[Tuple[int, int]]] = None):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.num_joints = num_joints
        self.feature_dim = feature_dim
        self.H, self.W = H, W
        self.temporal_stride = int(max(1, temporal_stride))
        self.use_gsplat = use_gsplat
        self.use_temporal_gru = use_temporal_gru
        self.use_nn_modulation = use_nn_modulation
        self.enable_nfm = bool(enable_nfm)
        self.bone_pairs = list(bone_pairs) if bone_pairs is not None else list(NTU_PAIRS)

        # Canonical skeleton
        self.register_buffer('canonical_joints', torch.zeros(num_joints, 3))

        # Canonical Gaussian set (absolute canonical centers)
        self.register_buffer('canonical_means', torch.zeros(num_gaussians, 3))
        self.register_buffer('canonical_scales', torch.ones(num_gaussians, 3) * 0.02)
        self.register_buffer('canonical_quats', torch.tensor([1, 0, 0, 0.], dtype=torch.float32).repeat(num_gaussians, 1))
        self.register_buffer('canonical_opacities', torch.ones(num_gaussians, 1))

        # Learnable / trainable parameters (except LBS weights which are kept fixed)
        # 🔧 FIX: Initialize canonical_features with small random values instead of zeros
        # Zero initialization may cause the network to get stuck in a poor local minimum
        self.canonical_features = nn.Parameter(torch.randn(num_gaussians, feature_dim) * 0.01)
        self.register_buffer('lbs_weights_logits', torch.zeros(num_gaussians, num_joints))

        # State bookkeeping so we only auto-initialize once when buffers stay at defaults
        self._canonical_joints_initialized = False
        self._canonical_gaussians_initialized = False
        self._canonical_scales_initialized = False
        self._lbs_initialized = False

        # Base appearance heads
        self.appearance_head = nn.Sequential(
            nn.Linear(feature_dim, max(64, feature_dim * 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(64, feature_dim * 2), 4)  # rgb(3) + alpha(1)
        )
        
        # 🔧 CRITICAL: Initialize appearance_head with small random values
        # Use simple normal distribution to avoid any potential NaN from complex init functions
        with torch.no_grad():
            for m in self.appearance_head.modules():
                if isinstance(m, nn.Linear):
                    # Small random weights
                    m.weight.data.normal_(mean=0.0, std=0.01)
                    if m.bias is not None:
                        # Small random bias
                        m.bias.data.normal_(mean=0.0, std=0.1)
                    # Verify no NaN after initialization
                    if torch.isnan(m.weight.data).any():
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.error("[INIT] NaN in appearance_head weight after init! Replacing with zeros.")
                        m.weight.data.zero_()
                        m.weight.data.normal_(mean=0.0, std=0.001)  # Try again with smaller std
                    if m.bias is not None and torch.isnan(m.bias.data).any():
                        m.bias.data.zero_()

        # 🆕 Optional NFM (Neural Field Modulation) network
        # Only created if enable_nfm=True to avoid unnecessary parameters
        if self.enable_nfm:
            # Input: [agg_pos(3), agg_vel(3), base_app(4)] = 10 dims (no metadata for stability)
            mod_in = 10
            mod_hidden = max(64, feature_dim * 2)
            self.nfm = nn.Sequential(
                nn.Linear(mod_in, mod_hidden),
                nn.ReLU(inplace=False),  # inplace=False is safer for gradients
                nn.Linear(mod_hidden, 5)  # delta_rgb(3) + delta_alpha(1) + saliency_gate(1)
            )
            
            # 🔧 SAFE INIT: Use small random initialization (avoid xavier/uniform that caused NaN)
            with torch.no_grad():
                for m in self.nfm.modules():
                    if isinstance(m, nn.Linear):
                        # Small random weights
                        m.weight.data.normal_(mean=0.0, std=0.01)
                        if m.bias is not None:
                            # Small random bias, with saliency output (index 4) near 0
                            m.bias.data.normal_(mean=0.0, std=0.01)
                            # Ensure saliency gate starts at sigmoid(0)=0.5
                            if m.bias.data.numel() >= 5:
                                m.bias.data[4] = 0.0
            
            # Ensure nfm is in float32
            self.nfm = self.nfm.to(dtype=torch.float32)
        else:
            self.nfm = None  # Not used when disabled

        # Optional temporal module (can be enabled separately from nfm)
        if self.use_temporal_gru and self.enable_nfm:  # GRU only makes sense with nfm
            self.temporal_gru = nn.GRU(input_size=10, hidden_size=10, num_layers=1, batch_first=True)
            self.temporal_gru = self.temporal_gru.to(dtype=torch.float32)
        else:
            self.temporal_gru = None

        # Depth-color mixing weight
        self.depth_mix_logit = nn.Parameter(torch.tensor(-0.5, dtype=torch.float32))
        
        # 🔧 Ensure all modules are in float32 (simple conversion, no hooks to avoid NaN)
        self.appearance_head = self.appearance_head.to(dtype=torch.float32)
        if self.nfm is not None:
            self.nfm = self.nfm.to(dtype=torch.float32)
        if self.temporal_gru is not None:
            self.temporal_gru = self.temporal_gru.to(dtype=torch.float32)

        # No FK: no parents tree or canonical world transforms

        # Always use the pure PyTorch rasterizer for stability
        self.use_gsplat = False
        self._gsplat_ok = False
        self.gsplat = None

    def set_canonical_joints(self, joints_xyz: torch.Tensor):
        self.canonical_joints.copy_(joints_xyz)
        self._canonical_joints_initialized = True
        # No FK recomputation

    def set_canonical_means(self, means_xyz: torch.Tensor):
        self.canonical_means.copy_(means_xyz)
        self._canonical_gaussians_initialized = True

    def set_lbs_weights_logits(self, logits: torch.Tensor):
        # Initialize/overwrite fixed LBS logits without tracking grads
        with torch.no_grad():
            self.lbs_weights_logits.copy_(logits)
        self._lbs_initialized = True

    def reset_temporal_state(self, batch_size: int, K_total: int, device: torch.device):
        if self.temporal_gru is not None:
            # GRU expects hx shape (num_layers, batch, hidden_size). Our inputs use batch_first with batch=batch_size.
            self._h_gru = torch.zeros(1, batch_size, self.temporal_gru.hidden_size, device=device)
        else:
            self._h_gru = None

    # No FK helpers

    def compute_joint_transforms(self, joints_now: torch.Tensor, orients_now: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Translation aligning canonical joints to current positions
        # + Optional pose-dependent rotation from joint orientations
        B, J = joints_now.shape[:2]
        t = joints_now - self.canonical_joints.unsqueeze(0)
        
        if orients_now is not None:
            # 🔧 CRITICAL: Check and validate quaternion data before use
            # NTU orientation data may contain invalid quaternions (zeros, NaN, or unnormalized)
            quat_norm = orients_now.norm(dim=-1, keepdim=True)
            invalid_quats = (quat_norm < 1e-6) | torch.isnan(orients_now).any(dim=-1, keepdim=True)
            
            if invalid_quats.any():
                # Replace invalid quaternions with identity [1,0,0,0]
                identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], 
                                            device=orients_now.device, 
                                            dtype=orients_now.dtype)
                orients_clean = torch.where(invalid_quats.expand_as(orients_now), 
                                           identity_quat.view(1, 1, 4).expand_as(orients_now),
                                           orients_now)
                
                # Log warning once
                if not hasattr(self, '_warned_invalid_quats'):
                    import logging
                    logger = logging.getLogger(__name__)
                    invalid_count = invalid_quats.sum().item()
                    total_count = invalid_quats.numel()
                    logger.warning(f"[ORIENTATION] Found {invalid_count}/{total_count} invalid quaternions, replacing with identity")
                    self._warned_invalid_quats = True
            else:
                orients_clean = orients_now
            
            # Use pose-dependent rotations from joint orientations (quaternions)
            # orients_clean: (B, J, 4) in (w, x, y, z) format
            R = _quat_to_mat(orients_clean)  # (B, J, 3, 3)
        else:
            # Fallback to pure translation (no rotation) for backward compatibility
            R = torch.eye(3, device=joints_now.device, dtype=joints_now.dtype).expand(B, J, 3, 3)
        
        return _compose_T(R, t)

    def blend_transforms(self, joint_T: torch.Tensor, lbs_w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Bstar, J = joint_T.shape[:2]
        K = lbs_w.shape[0]
        Rj = joint_T[..., :3, :3]
        tj = joint_T[..., :3, 3]
        t_blend = torch.einsum('kj, bjc -> bkc', lbs_w, tj)
        R_lin = torch.einsum('kj, bjmn -> bkmn', lbs_w, Rj)
        R_blend = _project_to_rotation(R_lin.reshape(-1, 3, 3)).reshape(Bstar, K, 3, 3)
        per_gauss_T = _compose_T(R_blend, t_blend)
        return per_gauss_T, R_blend, t_blend

    def transform_gaussians(self, means0: torch.Tensor, scales0: torch.Tensor, quats0: torch.Tensor,
                            R_blend: torch.Tensor, t_blend: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        Bstar, K = R_blend.shape[:2]
        # Ensure all operands share the same dtype to prevent bf16/float32 mismatch under AMP
        op_dtype = R_blend.dtype
        means0 = means0.to(op_dtype).unsqueeze(0).expand(Bstar, K, 3)
        R0 = _quat_to_mat(quats0.to(op_dtype)).unsqueeze(0).expand(Bstar, K, 3, 3)
        Rtot = R_blend @ R0
        means = (R_blend @ means0.unsqueeze(-1)).squeeze(-1) + t_blend
        s = scales0.to(op_dtype).unsqueeze(0).expand(Bstar, K, 3)
        covs = Rtot @ torch.diag_embed(s * s) @ Rtot.transpose(-1, -2)
        quats = _mat_to_quat(Rtot.reshape(-1, 3, 3)).reshape(Bstar, K, 4)
        return means, covs, quats, s

    def _vectorized_rasterize(self, means3D: torch.Tensor, cov3D: torch.Tensor, colors: torch.Tensor,
                               opacities: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor,
                               H: int, W: int, chunk_k: int = 32) -> torch.Tensor:
        # Memory-efficient front-to-back alpha compositing by streaming Gaussians
        device = means3D.device
        B, Knum, _ = means3D.shape
        C = colors.shape[-1]

        # Project to camera
        means_h = torch.cat([means3D, torch.ones_like(means3D[..., :1])], dim=-1)
        w2c_exp = w2c.unsqueeze(1).expand(B, Knum, -1, -1)
        cam = (w2c_exp @ means_h.unsqueeze(-1)).squeeze(-1)
        Xc, Yc, Zc = cam[..., 0], cam[..., 1], torch.clamp(cam[..., 2], min=1e-4)
        fx = K[:, 0, 0].unsqueeze(-1)
        fy = K[:, 1, 1].unsqueeze(-1)
        cx = K[:, 0, 2].unsqueeze(-1)
        cy = K[:, 1, 2].unsqueeze(-1)
        u = fx * (Xc / Zc) + cx
        v = -fy * (Yc / Zc) + cy

        # Screen-space covariance via Jacobian
        J11 = fx / Zc
        J12 = torch.zeros_like(J11)
        J13 = -fx * (Xc / (Zc * Zc))
        J21 = torch.zeros_like(J11)
        J22 = -fy / Zc
        J23 = fy * (Yc / (Zc * Zc))
        J = torch.stack([
            torch.stack([J11, J12, J13], dim=-1),
            torch.stack([J21, J22, J23], dim=-1)
        ], dim=-2)  # (B,K,2,3)
        Sigma2D = J @ cov3D @ J.transpose(-1, -2)  # (B,K,2,2)
        # Regularize
        eye2 = torch.eye(2, device=device, dtype=Sigma2D.dtype).view(1, 1, 2, 2)
        Sigma2D = Sigma2D + 1e-5 * eye2
        inv2D = torch.linalg.inv(Sigma2D)  # (B,K,2,2)

        # Depth sort (front -> back)
        zsort_idx = torch.argsort(Zc, dim=1, descending=False)
        u = torch.gather(u, 1, zsort_idx)
        v = torch.gather(v, 1, zsort_idx)
        colors = torch.gather(colors, 1, zsort_idx.unsqueeze(-1).expand(-1, -1, C))
        opacities = torch.gather(opacities, 1, zsort_idx)
        inv2D = torch.gather(inv2D, 1, zsort_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 2, 2))

        # Pixel grid
        ys = torch.arange(0, H, device=device, dtype=means3D.dtype)
        xs = torch.arange(0, W, device=device, dtype=means3D.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H,W)
        grid_x = grid_x.unsqueeze(0)  # (1,H,W)
        grid_y = grid_y.unsqueeze(0)  # (1,H,W)

        out = torch.zeros(B, H, W, C, device=device, dtype=means3D.dtype)
        trans = torch.ones(B, H, W, 1, device=device, dtype=means3D.dtype)

        # Stream over K in chunks, and within each chunk compose per-Gaussian to avoid (B,H,W,K) tensors
        for start in range(0, Knum, chunk_k):
            end = min(Knum, start + chunk_k)

            u_chunk = u[:, start:end]            # (B,Ks)
            v_chunk = v[:, start:end]            # (B,Ks)
            inv_chunk = inv2D[:, start:end]      # (B,Ks,2,2)
            col_chunk = colors[:, start:end]     # (B,Ks,3)
            opa_chunk = opacities[:, start:end]  # (B,Ks)

            Ks = end - start
            for i in range(Ks):
                u_i = u_chunk[:, i]            # (B,)
                v_i = v_chunk[:, i]            # (B,)
                inv_i = inv_chunk[:, i]        # (B,2,2)
                col_i = col_chunk[:, i]        # (B,3)
                opa_i = opa_chunk[:, i]        # (B,)

                # Broadcast to image grid
                dx = grid_x - u_i.view(B, 1, 1)
                dy = grid_y - v_i.view(B, 1, 1)
                a = inv_i[:, 0, 0].view(B, 1, 1)
                b = inv_i[:, 0, 1].view(B, 1, 1)
                c = inv_i[:, 1, 0].view(B, 1, 1)
                d = inv_i[:, 1, 1].view(B, 1, 1)
                dist = a * dx * dx + (b + c) * dx * dy + d * dy * dy  # (B,H,W)
                w = torch.exp(-0.5 * dist).clamp(0.0, 1.0)
                alpha = 1.0 - torch.exp(-opa_i.view(B, 1, 1) * w)     # (B,H,W)

                alpha = alpha.unsqueeze(-1)  # (B,H,W,1)
                out = out + trans * alpha * col_i.view(B, 1, 1, 3)
                trans = trans * (1.0 - alpha)

        return out

    def forward(self, poses: torch.Tensor, metas: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor,
                vels: Optional[torch.Tensor] = None,
                orients: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        # poses: (T,P,J,3), metas: (T,P,10) [metas not used in simplified version]
        # vels: (T,P,J,3) pre-computed velocities from original frames
        # orients: (T,P,J,4) optional joint orientations (quaternions) for pose-dependent rotations
        # 🔧 CRITICAL FIX: Force entire renderer to run in float32 to avoid NaN in bf16 training
        device = poses.device
        poses = poses.to(device=device, dtype=torch.float32)
        K = K.to(device=device, dtype=torch.float32)
        w2c = w2c.to(device=device, dtype=torch.float32)
        
        # 🔧 SIMPLIFIED: Velocities are now pre-computed and passed in
        if vels is not None:
            vels = vels.to(device=device, dtype=torch.float32)
        
        # Handle optional orientations for pose-dependent rotations
        if orients is not None:
            orients = orients.to(device=device, dtype=torch.float32)
        
        T_len, P, J = poses.shape[0], poses.shape[1], poses.shape[2]

        num_line_samples = kwargs.get('num_line_samples', 10)
        use_adaptive_scales = kwargs.get('adaptive_scales', True)
        gamma_scale = kwargs.get('adaptive_scale_gamma', 1.0)
        base_joint_scale = kwargs.get('base_joint_scale', 0.030)
        base_line_scale = kwargs.get('base_line_scale', 0.020)
        min_joint_scale = kwargs.get('min_joint_scale', 0.010)
        max_joint_scale = kwargs.get('max_joint_scale', 0.035)
        min_line_scale = kwargs.get('min_line_scale', 0.006)
        max_line_scale = kwargs.get('max_line_scale', 0.030)

        # Sanity on stored template sizes
        if self.canonical_means.shape[0] != self.canonical_features.shape[0]:
            raise ValueError(
                f"canonical_means ({self.canonical_means.shape[0]}) and canonical_features "
                f"({self.canonical_features.shape[0]}) disagree on gaussian count."
            )

        K_total = self.canonical_means.shape[0]
        if K_total < J:
            raise ValueError(f"Configured gaussian count {K_total} is smaller than joint count {J}.")
        K_line_target = K_total - J

        # Resolve canonical joints from module state (fallback to first frame once if unset)
        canonical_joints_override = kwargs.get('canonical_joints')
        if canonical_joints_override is not None:
            if canonical_joints_override.shape != self.canonical_joints.shape:
                raise ValueError("Provided canonical_joints has incompatible shape.")
            with torch.no_grad():
                self.canonical_joints.copy_(canonical_joints_override)
            self._canonical_joints_initialized = True
        canonical = self.canonical_joints
        if not self._canonical_joints_initialized:
            check = canonical.detach()
            if float(check.abs().sum().item()) < 1e-8:
                fallback_joints = poses[0, 0].detach()
                with torch.no_grad():
                    self.canonical_joints.copy_(fallback_joints)
                canonical = self.canonical_joints
            self._canonical_joints_initialized = True
        if canonical.shape[0] != J:
            raise ValueError(
                f"Canonical joints expect {canonical.shape[0]} entries but current poses have {J} joints."
            )

        # Build line samples to support fallback initialisation of template state
        line_samples_full, sample_defs_full = _build_line_samples(
            canonical,
            num_line_samples=num_line_samples,
            pairs=self.bone_pairs,
        )
        if K_line_target > 0:
            if line_samples_full.shape[0] < K_line_target:
                raise ValueError(
                    f"Not enough line samples ({line_samples_full.shape[0]}) to satisfy "
                    f"configured gaussian count ({K_total})."
                )
            line_samples = line_samples_full[:K_line_target]
            sample_defs = sample_defs_full[:K_line_target]
        else:
            line_samples = canonical.new_zeros((0, 3))
            sample_defs = []

        # Resolve canonical means
        canonical_means_override = kwargs.get('canonical_means')
        if canonical_means_override is not None:
            if canonical_means_override.shape != self.canonical_means.shape:
                raise ValueError("Provided canonical_means has incompatible shape.")
            with torch.no_grad():
                self.canonical_means.copy_(canonical_means_override)
            self._canonical_gaussians_initialized = True
        canonical_means = self.canonical_means
        if not self._canonical_gaussians_initialized:
            check = canonical_means.detach()
            if float(check.abs().sum().item()) < 1e-8:
                new_means = torch.zeros_like(self.canonical_means)
                new_means[:J] = canonical
                if K_line_target > 0 and line_samples.shape[0] > 0:
                    end = J + min(line_samples.shape[0], new_means.shape[0] - J)
                    new_means[J:end] = line_samples[:max(0, end - J)]
                with torch.no_grad():
                    self.canonical_means.copy_(new_means)
                canonical_means = self.canonical_means
            self._canonical_gaussians_initialized = True

        # Resolve canonical scales
        canonical_scales_override = kwargs.get('canonical_scales')
        if canonical_scales_override is not None:
            if canonical_scales_override.shape != self.canonical_scales.shape:
                raise ValueError("Provided canonical_scales has incompatible shape.")
            with torch.no_grad():
                self.canonical_scales.copy_(canonical_scales_override)
            self._canonical_scales_initialized = True
        canonical_scales = self.canonical_scales
        needs_scales_init = not self._canonical_scales_initialized
        if needs_scales_init:
            check_scales = canonical_scales.detach()
            needs_scales_init = float(check_scales.abs().sum().item()) < 1e-8
        if needs_scales_init or canonical_scales.shape[0] != canonical_means.shape[0]:
            if use_adaptive_scales:
                joint_scales_t, line_scales_t = _compute_adaptive_scales(
                    canonical,
                    sample_defs,
                    num_joints=J,
                    base_joint_scale=base_joint_scale,
                    base_line_scale=base_line_scale,
                    min_joint_scale=min_joint_scale,
                    max_joint_scale=max_joint_scale,
                    min_line_scale=min_line_scale,
                    max_line_scale=max_line_scale,
                    gamma=gamma_scale,
                    pairs=self.bone_pairs,
                )
                new_scales = torch.zeros_like(self.canonical_scales)
                new_scales[:J] = joint_scales_t.to(new_scales.dtype)
                if K_line_target > 0 and line_scales_t.numel() > 0:
                    end = J + min(line_scales_t.shape[0], new_scales.shape[0] - J)
                    new_scales[J:end] = line_scales_t[:max(0, end - J)].to(new_scales.dtype)
            else:
                new_scales = torch.zeros_like(self.canonical_scales)
                new_scales[:J] = base_joint_scale
                if K_line_target > 0:
                    new_scales[J:] = base_line_scale
            with torch.no_grad():
                self.canonical_scales.copy_(new_scales)
            canonical_scales = self.canonical_scales
            self._canonical_scales_initialized = True
        else:
            self._canonical_scales_initialized = True

        # Resolve LBS weights (kept fixed; no gradients)
        lbs_logits_override = kwargs.get('lbs_weights_logits')
        if lbs_logits_override is not None:
            if lbs_logits_override.shape != self.lbs_weights_logits.shape:
                raise ValueError("Provided lbs_weights_logits has incompatible shape.")
            with torch.no_grad():
                self.lbs_weights_logits.copy_(lbs_logits_override)
            self._lbs_initialized = True
        lbs_logits = self.lbs_weights_logits
        needs_lbs_init = not self._lbs_initialized
        if needs_lbs_init:
            check_lbs = lbs_logits.detach()
            needs_lbs_init = float(check_lbs.abs().sum().item()) < 1e-8
        if needs_lbs_init or lbs_logits.shape[0] != K_total:
            default_logits = torch.full_like(self.lbs_weights_logits, -10.0)
            diag_n = min(J, default_logits.shape[0])
            idx = torch.arange(diag_n, device=default_logits.device)
            default_logits[idx, idx] = 10.0
            if K_line_target > 0 and len(sample_defs) > 0:
                line_logits = _make_lbs_logits_for_samples(J, sample_defs).to(default_logits.device, default_logits.dtype)
                end = J + min(line_logits.shape[0], default_logits.shape[0] - J)
                default_logits[J:end] = line_logits[:max(0, end - J)]
            with torch.no_grad():
                self.lbs_weights_logits.copy_(default_logits)
            lbs_logits = self.lbs_weights_logits
            self._lbs_initialized = True
        else:
            self._lbs_initialized = True

        lbs_w = torch.softmax(lbs_logits, dim=-1).to(dtype=poses.dtype)

        # 🔧 SIMPLIFIED: Velocities are now pre-computed from original frames and passed in
        # If not provided, compute from sampled frames as fallback
        if vels is None:
            stride = self.temporal_stride
            idx_fut = torch.clamp(torch.arange(T_len, device=device) + stride, max=T_len - 1)
            vels = poses.index_select(0, idx_fut) - poses
        # Ensure vels is on correct device and dtype
        vels = vels.to(device=device, dtype=poses.dtype)

        # 🔧 CRITICAL FIX: Force float32 for appearance_head to avoid NaN in bf16 training
        with torch.autocast(device_type=device.type, enabled=False):
            cf_float32 = self.canonical_features.float()
            base_rgba_head = self.appearance_head(cf_float32).unsqueeze(0)
            base_rgba_head = base_rgba_head.to(dtype=poses.dtype)
            
            # 🔧 NaN detection and replacement BEFORE sigmoid
            if torch.isnan(base_rgba_head).any():
                import logging
                logger = logging.getLogger(__name__)
                logger.error("[CRITICAL] NaN detected in appearance_head output! Replacing with zeros (will become 0.5 after sigmoid).")
                # Replace NaN with 0, so sigmoid(0)=0.5
                base_rgba_head = torch.nan_to_num(base_rgba_head, nan=0.0, posinf=0.0, neginf=0.0)
            
            base_rgb_head = base_rgba_head[..., 0:3].sigmoid()
            base_alpha_head = base_rgba_head[..., 3:4].sigmoid()
            
            # 🔧 Safety check AFTER sigmoid - if still NaN, force to 0.5
            if torch.isnan(base_rgb_head).any() or torch.isnan(base_alpha_head).any():
                import logging
                logger = logging.getLogger(__name__)
                logger.error("[CRITICAL] NaN persists after sigmoid! Force replacing with 0.5.")
                base_rgb_head = torch.where(torch.isnan(base_rgb_head), torch.full_like(base_rgb_head, 0.5), base_rgb_head)
                base_alpha_head = torch.where(torch.isnan(base_alpha_head), torch.full_like(base_alpha_head, 0.5), base_alpha_head)
        lam = torch.sigmoid(self.depth_mix_logit).to(poses.dtype)
        
        # 🔍 DEBUG: Log appearance head outputs (first frame only to reduce spam)
        if not hasattr(self, '_debug_logged_appearance'):
            self._debug_logged_appearance = False
        if not self._debug_logged_appearance:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[RENDERER_DEBUG] canonical_features: min={self.canonical_features.min().item():.6f}, "
                       f"max={self.canonical_features.max().item():.6f}, mean={self.canonical_features.mean().item():.6f}, "
                       f"norm={self.canonical_features.norm().item():.6f}, requires_grad={self.canonical_features.requires_grad}")
            
            # Check appearance_head parameters
            app_param = next(self.appearance_head.parameters())
            logger.info(f"[RENDERER_DEBUG] appearance_head param: dtype={app_param.dtype}, requires_grad={app_param.requires_grad}, norm={app_param.norm().item():.6f}")
            
            # Check raw output before sigmoid
            logger.info(f"[RENDERER_DEBUG] base_rgba_head (before sigmoid): min={base_rgba_head.min().item():.6f}, "
                       f"max={base_rgba_head.max().item():.6f}, std={base_rgba_head.std().item():.6f}")
            logger.info(f"[RENDERER_DEBUG] base_rgb_head (after sigmoid): min={base_rgb_head.min().item():.6f}, "
                       f"max={base_rgb_head.max().item():.6f}, mean={base_rgb_head.mean().item():.6f}, std={base_rgb_head.std().item():.6f}")
            logger.info(f"[RENDERER_DEBUG] base_alpha_head (after sigmoid): min={base_alpha_head.min().item():.6f}, "
                       f"max={base_alpha_head.max().item():.6f}, mean={base_alpha_head.mean().item():.6f}, std={base_alpha_head.std().item():.6f}")
            logger.info(f"[RENDERER_DEBUG] depth_mix_logit={self.depth_mix_logit.item():.6f}, lam={lam.item():.6f}, requires_grad={self.depth_mix_logit.requires_grad}")
            self._debug_logged_appearance = True

        frames_list: List[torch.Tensor] = []
        self.reset_temporal_state(batch_size=1, K_total=K_total, device=device)
        for t in range(T_len):
                per_means = []
                per_scales = []
                per_quats = []
                per_colors = []
                per_opac = []
                per_covs = []

                for p in range(P):
                    joints_now = poses[t, p].unsqueeze(0)
                    valid = (joints_now.abs().sum() > 0).float()
                    
                    # Extract orientations if available
                    orients_now = orients[t, p].unsqueeze(0) if orients is not None else None

                    joint_T = self.compute_joint_transforms(joints_now, orients_now)
                    _, R_blend, t_blend = self.blend_transforms(joint_T, lbs_w)
                    means, covs, quats, scales = self.transform_gaussians(
                        canonical_means, canonical_scales, self.canonical_quats, R_blend, t_blend
                    )

                    means_h = torch.cat([means, torch.ones_like(means[..., :1])], dim=-1)
                    cam = (w2c[t:t+1].unsqueeze(1) @ means_h.unsqueeze(-1)).squeeze(-1)
                    Zc = torch.clamp(cam[..., 2], min=1e-4)
                    depth_rgb = _depth_to_rgb(Zc)

                    # 🆕 Optional NFM modulation (only if enable_nfm=True)
                    if self.enable_nfm and self.nfm is not None:
                        # Compute nfm input: position + velocity + base appearance
                        vel_now = vels[t, p].unsqueeze(0)
                        agg_pos = torch.einsum('kj, bjc -> bkc', lbs_w, joints_now)
                        agg_vel = torch.einsum('kj, bjc -> bkc', lbs_w, vel_now)
                        base_rgba = base_rgba_head
                        mod_in = torch.cat([agg_pos, agg_vel, base_rgba], dim=-1)  # (1, K, 10)
                        
                        # 🔧 Force float32 for nfm to avoid NaN
                        with torch.autocast(device_type=device.type, enabled=False):
                            mod_in_float32 = mod_in.float()
                            
                            # Optional temporal GRU
                            if self.temporal_gru is not None:
                                x_seq = mod_in_float32.reshape(1, -1, mod_in_float32.shape[-1])
                                x_seq, self._h_gru = self.temporal_gru(x_seq, self._h_gru)
                                mod_in_eff = x_seq.reshape(1, -1, mod_in_float32.shape[-1])
                            else:
                                mod_in_eff = mod_in_float32
                            
                            # NFM forward
                            deltas = self.nfm(mod_in_eff).to(dtype=mod_in.dtype)
                            
                            # 🔧 NaN protection
                            if torch.isnan(deltas).any():
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.warning("[NFM] NaN detected in nfm output! Replacing with zeros.")
                                deltas = torch.nan_to_num(deltas, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        delta_rgb = deltas[..., 0:3]
                        delta_alpha = deltas[..., 3:4]
                        saliency = deltas[..., 4:5].sigmoid()
                        
                        # Apply modulation
                        base_rgb = base_rgb_head.to(delta_rgb.dtype)
                        base_alpha = base_alpha_head.to(delta_alpha.dtype)
                        learned_rgb = (base_rgb + delta_rgb).clamp(0.0, 1.0)
                        learned_alpha = (base_alpha + delta_alpha).sigmoid() * saliency
                        final_color = (1.0 - lam) * learned_rgb + lam * depth_rgb
                    else:
                        # 🔧 BASE MODE (no nfm): Use only base appearance + depth mix
                        # This is the stable default mode that works reliably
                        base_rgb = base_rgb_head.to(depth_rgb.dtype)
                        base_alpha = base_alpha_head.to(depth_rgb.dtype)
                        learned_rgb = base_rgb
                        learned_alpha = base_alpha
                        final_color = (1.0 - lam) * learned_rgb + lam * depth_rgb

                    # 🔧 SIMPLIFIED: Remove metadata-based gating (no left_conf, right_conf)
                    # opac now directly from learned_alpha without confidence modulation
                    opac = learned_alpha.squeeze(-1) * valid
                    
                    # 🔍 DEBUG: Log opacity and color statistics for first frame
                    if t == 0 and p == 0 and not hasattr(self, '_debug_logged_render_details'):
                        import logging
                        logger = logging.getLogger(__name__)
                        mode_str = "[NFM ENABLED]" if self.enable_nfm else "[BASE MODE]"
                        logger.info(f"[RENDER_DETAIL] t={t}, p={p} {mode_str}")
                        logger.info(f"  base_rgb: min={base_rgb.min().item():.6f}, max={base_rgb.max().item():.6f}, mean={base_rgb.mean().item():.6f}")
                        logger.info(f"  base_alpha: min={base_alpha.min().item():.6f}, max={base_alpha.max().item():.6f}, mean={base_alpha.mean().item():.6f}")
                        if self.enable_nfm and self.nfm is not None:
                            logger.info(f"  delta_rgb: min={delta_rgb.min().item():.6f}, max={delta_rgb.max().item():.6f}, mean={delta_rgb.mean().item():.6f}")
                            logger.info(f"  delta_alpha: min={delta_alpha.min().item():.6f}, max={delta_alpha.max().item():.6f}, mean={delta_alpha.mean().item():.6f}")
                            logger.info(f"  saliency: min={saliency.min().item():.6f}, max={saliency.max().item():.6f}, mean={saliency.mean().item():.6f}")
                        logger.info(f"  depth_rgb: min={depth_rgb.min().item():.6f}, max={depth_rgb.max().item():.6f}, mean={depth_rgb.mean().item():.6f}")
                        logger.info(f"  lam (depth_mix): {lam.item():.6f}")
                        logger.info(f"  learned_alpha: min={learned_alpha.min().item():.6f}, max={learned_alpha.max().item():.6f}, mean={learned_alpha.mean().item():.6f}")
                        logger.info(f"  valid: {valid.item()}")
                        logger.info(f"  opac (final): min={opac.min().item():.6f}, max={opac.max().item():.6f}, mean={opac.mean().item():.6f}, nonzero={int((opac > 0).sum().item())}/{opac.numel()}")
                        logger.info(f"  final_color: min={final_color.min().item():.6f}, max={final_color.max().item():.6f}, mean={final_color.mean().item():.6f}")
                        self._debug_logged_render_details = True

                    per_means.append(means)
                    per_scales.append(scales)
                    per_quats.append(quats)
                    per_colors.append(final_color)
                    per_opac.append(opac)
                    per_covs.append(covs)

                means_cat = torch.cat(per_means, dim=1)
                scales_cat = torch.cat(per_scales, dim=1)
                quats_cat = torch.cat(per_quats, dim=1)
                colors_cat = torch.cat(per_colors, dim=1)
                opac_cat = torch.cat(per_opac, dim=1)
                cov3D = torch.cat(per_covs, dim=1)

                colors_cat = torch.nan_to_num(colors_cat, nan=0.0, posinf=0.0, neginf=0.0)
                colors_cat = colors_cat.clamp(0.0, 1.0)
                opac_cat = torch.nan_to_num(opac_cat, nan=0.0, posinf=0.0, neginf=0.0)
                opac_cat = opac_cat.clamp(0.0, None)

                frame_t = self._vectorized_rasterize(
                    means3D=means_cat,
                    cov3D=cov3D,
                    colors=colors_cat,
                    opacities=opac_cat,
                    K=K[t:t+1],
                    w2c=w2c[t:t+1],
                    H=self.H,
                    W=self.W,
                )
                
                # 🔍 DEBUG: Log rasterization output before nan_to_num for first frame
                if t == 0 and not hasattr(self, '_debug_logged_raster'):
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"[RASTER_DEBUG] t={t}")
                    logger.info(f"  Input opac_cat: min={opac_cat.min().item():.6f}, max={opac_cat.max().item():.6f}, nonzero={int((opac_cat > 0).sum().item())}/{opac_cat.numel()}")
                    logger.info(f"  Input colors_cat: min={colors_cat.min().item():.6f}, max={colors_cat.max().item():.6f}")
                    logger.info(f"  Raw rasterized output: min={frame_t.min().item():.6f}, max={frame_t.max().item():.6f}, mean={frame_t.mean().item():.6f}")
                    has_nan = torch.isnan(frame_t).any().item()
                    has_inf = torch.isinf(frame_t).any().item()
                    logger.info(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
                    self._debug_logged_raster = True
                
                frame_t = torch.nan_to_num(frame_t, nan=0.0, posinf=0.0, neginf=0.0)
                frames_list.append(frame_t)

        if frames_list:
            frames_bt = torch.cat(frames_list, dim=0)
            channels = frames_bt.shape[-1]
            frames_bthwc = frames_bt.reshape(1, T_len, self.H, self.W, channels).contiguous().clamp(0.0, 1.0)
        else:
            channels = 3
            frames_bthwc = torch.zeros(1, T_len, self.H, self.W, channels, device=device, dtype=poses.dtype)
        
        # 🔍 DEBUG: Log final output statistics
        if not hasattr(self, '_debug_logged_output'):
            self._debug_logged_output = False
        if not self._debug_logged_output:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[RENDERER_DEBUG] Final output: shape={frames_bthwc.shape}, "
                       f"min={frames_bthwc.min().item():.6f}, max={frames_bthwc.max().item():.6f}, "
                       f"mean={frames_bthwc.mean().item():.6f}, std={frames_bthwc.std().item():.6f}")
            self._debug_logged_output = True
        
        return frames_bthwc


# 4) NTU RGB+D skeleton parser (supports up to 2 bodies per frame)
def parse_ntu_skeleton_file(file_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with open(file_path, 'r') as f:
        content = f.read()
    tokens = content.split()
    idx = 0

    def next_int() -> int:
        nonlocal idx
        val = int(tokens[idx]); idx += 1
        return val

    def next_floats(n: int) -> List[float]:
        nonlocal idx
        vals = list(map(float, tokens[idx:idx + n])); idx += n
        return vals

    T = next_int()

    poses: List[List[List[List[float]]]] = []    # (T, P<=2, 25, 3)
    metas: List[List[List[float]]] = []          # (T, P<=2, 10)
    orients: List[List[List[List[float]]]] = []  # (T, P<=2, 25, 4)

    for _ in range(T):
        bodies = next_int()
        frame_joints_list: List[List[List[float]]] = []
        frame_quats_list: List[List[List[float]]] = []
        frame_meta_list: List[List[float]] = []

        for b in range(bodies):
            meta10 = next_floats(10)
            J = next_int()
            joints_b: List[List[float]] = []
            quats_b: List[List[float]] = []
            for __ in range(J):
                vals = next_floats(12)
                x, y, z = vals[0], vals[1], vals[2]
                w, ox, oy, oz = vals[7], vals[8], vals[9], vals[10]
                joints_b.append([x, y, z])
                quats_b.append([w, ox, oy, oz])
            if J < 25:
                pad_n = 25 - J
                joints_b = joints_b + [[0.0, 0.0, 0.0]] * pad_n
                quats_b = quats_b + [[1.0, 0.0, 0.0, 0.0]] * pad_n
            elif J > 25:
                joints_b = joints_b[:25]
                quats_b = quats_b[:25]
            if len(frame_joints_list) < 2:
                frame_joints_list.append(joints_b)
                frame_quats_list.append(quats_b)
                frame_meta_list.append(meta10)

        while len(frame_joints_list) < 2:
            frame_joints_list.append([[0.0, 0.0, 0.0] for _ in range(25)])
            frame_quats_list.append([[1.0, 0.0, 0.0, 0.0] for _ in range(25)])
            frame_meta_list.append([0.0 for _ in range(10)])

        poses.append(frame_joints_list)
        metas.append(frame_meta_list)
        orients.append(frame_quats_list)

    poses_t = torch.tensor(poses, dtype=torch.float32)        # (T, 2, 25, 3)
    metas_t = torch.tensor(metas, dtype=torch.float32)        # (T, 2, 10)
    orients_t = torch.tensor(orients, dtype=torch.float32)    # (T, 2, 25, 4)
    return poses_t, metas_t, orients_t


def parse_humanml3d_skeleton_file(file_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parse a converted HumanML3D JSON file into DrAction tensors."""
    with open(file_path, 'r', encoding='utf-8') as fin:
        data = json.load(fin)

    if 'skeletons' not in data:
        raise ValueError(f"HumanML3D file has no 'skeletons' field: {file_path}")
    poses = torch.as_tensor(data['skeletons'], dtype=torch.float32)
    if poses.ndim != 3 or poses.shape[-2:] != (HUMANML3D_NUM_JOINTS, 3):
        raise ValueError(
            f"Expected HumanML3D poses with shape (T, {HUMANML3D_NUM_JOINTS}, 3), "
            f"got {tuple(poses.shape)} in {file_path}"
        )
    poses = poses.unsqueeze(1)
    num_frames = poses.shape[0]
    metas = torch.zeros(num_frames, 1, 10, dtype=torch.float32)
    orients = torch.zeros(
        num_frames, 1, HUMANML3D_NUM_JOINTS, 4, dtype=torch.float32
    )
    orients[..., 0] = 1.0
    return poses, metas, orients


def get_skeleton_type(file_path: str) -> str:
    suffix = str(file_path).lower()
    if suffix.endswith('.skeleton'):
        return 'ntu'
    if suffix.endswith('.json'):
        return 'humanml3d'
    raise ValueError(f"Unsupported skeleton file format: {file_path}")


def get_bone_pairs(skeleton_type: str) -> List[Tuple[int, int]]:
    if skeleton_type == 'ntu':
        return NTU_PAIRS
    if skeleton_type == 'humanml3d':
        return HUMANML3D_PAIRS
    raise ValueError(f"Unsupported skeleton type: {skeleton_type}")


def get_num_joints(skeleton_type: str) -> int:
    if skeleton_type == 'ntu':
        return NTU_NUM_JOINTS
    if skeleton_type == 'humanml3d':
        return HUMANML3D_NUM_JOINTS
    raise ValueError(f"Unsupported skeleton type: {skeleton_type}")


def parse_skeleton_file(
    file_path: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    skeleton_type = get_skeleton_type(file_path)
    if skeleton_type == 'ntu':
        poses, metas, orients = parse_ntu_skeleton_file(file_path)
    else:
        poses, metas, orients = parse_humanml3d_skeleton_file(file_path)
    return poses, metas, orients, skeleton_type


def _build_line_samples(
    canonical_joints: torch.Tensor,
    num_line_samples: int,
    pairs: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[torch.Tensor, List[Tuple[int, int, float]]]:
    pairs = NTU_PAIRS if pairs is None else pairs
    samples = []
    sample_defs: List[Tuple[int, int, float]] = []
    num_joints = canonical_joints.shape[0]
    for a, b in pairs:
        a_idx = a - 1
        b_idx = b - 1
        if not (0 <= a_idx < num_joints and 0 <= b_idx < num_joints):
            continue
        for s in range(1, num_line_samples + 1):
            alpha = s / (num_line_samples + 1)
            p = (1.0 - alpha) * canonical_joints[a_idx] + alpha * canonical_joints[b_idx]
            samples.append(p)
            sample_defs.append((a_idx, b_idx, alpha))
    samples_tensor = torch.stack(samples, dim=0) if len(samples) > 0 else torch.empty(0, 3, device=canonical_joints.device)
    return samples_tensor, sample_defs


def _compute_bone_lengths(
    canonical_joints: torch.Tensor,
    pairs: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[List[Tuple[int, int]], torch.Tensor, float]:
    pairs = NTU_PAIRS if pairs is None else pairs
    pairs_idx: List[Tuple[int, int]] = []
    lengths: List[float] = []
    J = canonical_joints.shape[-2]
    for a, b in pairs:
        a_idx = a - 1
        b_idx = b - 1
        if 0 <= a_idx < J and 0 <= b_idx < J:
            L = (canonical_joints[a_idx] - canonical_joints[b_idx]).norm().item()
            pairs_idx.append((a_idx, b_idx))
            lengths.append(L)
    if len(lengths) == 0:
        lengths_t = torch.tensor([1.0], dtype=canonical_joints.dtype, device=canonical_joints.device)
        Lmax = 1.0
    else:
        lengths_t = torch.tensor(lengths, dtype=canonical_joints.dtype, device=canonical_joints.device)
        Lmax = float(torch.clamp(lengths_t.max(), min=1e-6).item())
    return pairs_idx, lengths_t, Lmax


def _build_line_samples_adaptive(
    canonical_joints: torch.Tensor,
    base_num_line_samples: int,
    min_samples: int = 1,
    max_samples: Optional[int] = None,
    beta: float = 1.0,
    pairs: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[torch.Tensor, List[Tuple[int, int, float]]]:
    samples = []
    sample_defs: List[Tuple[int, int, float]] = []
    pairs_idx, lengths_t, Lmax = _compute_bone_lengths(canonical_joints, pairs=pairs)
    if max_samples is None:
        max_samples = base_num_line_samples
    for (a_idx, b_idx), L in zip(pairs_idx, lengths_t):
        r = float(L) / Lmax
        n = int(round(base_num_line_samples * (r ** beta)))
        n = max(min_samples, min(max_samples, n))
        for s in range(1, n + 1):
            alpha = s / (n + 1)
            p = (1.0 - alpha) * canonical_joints[a_idx] + alpha * canonical_joints[b_idx]
            samples.append(p)
            sample_defs.append((a_idx, b_idx, alpha))
    samples_tensor = torch.stack(samples, dim=0) if len(samples) > 0 else torch.empty(0, 3, device=canonical_joints.device)
    return samples_tensor, sample_defs


def _compute_adaptive_scales(
    canonical_joints: torch.Tensor,
    sample_defs: List[Tuple[int, int, float]],
    num_joints: int,
    base_joint_scale: float = 0.030,
    base_line_scale: float = 0.020,
    min_joint_scale: float = 0.010,
    max_joint_scale: float = 0.035,
    min_line_scale: float = 0.006,
    max_line_scale: float = 0.030,
    gamma: float = 1.0,
    pairs: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = canonical_joints.device
    dtype = canonical_joints.dtype

    # Bone lengths and adjacency
    pairs_idx, lengths_t, Lmax = _compute_bone_lengths(canonical_joints, pairs=pairs)
    length_by_pair = {pair: float(L) for pair, L in zip(pairs_idx, lengths_t)}
    joints_adj: List[List[float]] = [[] for _ in range(num_joints)]
    for (a_idx, b_idx), L in zip(pairs_idx, lengths_t):
        joints_adj[a_idx].append(float(L))
        joints_adj[b_idx].append(float(L))

    # Joint scales
    joint_scales = []
    for j in range(num_joints):
        if len(joints_adj[j]) == 0:
            rj = 0.5
        else:
            med = float(torch.tensor(joints_adj[j]).median().item())
            rj = med / Lmax
        sj = base_joint_scale * (rj ** gamma)
        sj = float(max(min_joint_scale, min(max_joint_scale, sj)))
        joint_scales.append([sj, sj, sj])
    joint_scales_t = torch.tensor(joint_scales, dtype=dtype, device=device)

    # Line sample scales per sample_def
    line_scales = []
    for (a_idx, b_idx, _alpha) in sample_defs:
        L = length_by_pair.get((a_idx, b_idx), Lmax)
        r = float(L) / Lmax
        sl = base_line_scale * (r ** gamma)
        sl = float(max(min_line_scale, min(max_line_scale, sl)))
        line_scales.append([sl, sl, sl])
    line_scales_t = torch.tensor(line_scales, dtype=dtype, device=device) if len(line_scales) > 0 else torch.empty(0, 3, dtype=dtype, device=device)

    return joint_scales_t, line_scales_t


def _make_lbs_logits_for_samples(num_joints: int, sample_defs: List[Tuple[int, int, float]],
                                 joint_focus_logit: float = 10.0, other_logit: float = -10.0) -> torch.Tensor:
    logits = []
    for a_idx, b_idx, alpha in sample_defs:
        row = torch.full((num_joints,), other_logit)
        row[a_idx] = math.log(max(1e-6, 1.0 - alpha)) + joint_focus_logit
        row[b_idx] = math.log(max(1e-6, alpha)) + joint_focus_logit
        logits.append(row)
    return torch.stack(logits, dim=0) if len(logits) > 0 else torch.empty(0, num_joints)


def _depth_to_rgb(z: torch.Tensor) -> torch.Tensor:
    zmin = z.amin(dim=1, keepdim=True)
    zmax = z.amax(dim=1, keepdim=True)
    d = (z - zmin) / (zmax - zmin + 1e-6)
    r = d
    g = 1.0 - torch.abs(d - 0.5) * 2.0
    b = 1.0 - d
    return torch.stack([r, g, b], dim=-1).clamp(0.0, 1.0)

#随机的
def _sample_indices_uniform(T: int, target_T: int, device: Optional[torch.device] = None, seed: Optional[int] = None) -> torch.Tensor:
    # Returns LongTensor of length target_T with indices in [0, T-1]
    target_T = max(1, int(target_T))
    if T <= 0:
        return torch.zeros(target_T, dtype=torch.long, device=device)
    if target_T == 1:
        return torch.tensor([max(0, min(T - 1, T // 2))], dtype=torch.long, device=device)
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    if T == target_T:
        idx = torch.arange(target_T, dtype=torch.long)
        return idx.to(device) if device is not None else idx
    if T < target_T:
        # Random sample with replacement, then sort to keep temporal order
        idx = torch.randint(low=0, high=T, size=(target_T,), generator=gen)
        idx, _ = torch.sort(idx)
        return idx.to(device) if device is not None else idx
    # T > target_T: divide into target_T segments and sample one per segment
    boundaries = torch.linspace(0, T, steps=target_T + 1)
    indices = []
    for i in range(target_T):
        start = int(boundaries[i].floor().item())
        end = int(boundaries[i + 1].ceil().item()) - 1
        start = max(0, min(start, T - 1))
        end = max(start, min(end, T - 1))
        if end >= start:
            r = torch.randint(low=start, high=end + 1, size=(1,), generator=gen).item()
        else:
            r = start
        indices.append(r)
    idx = torch.tensor(indices, dtype=torch.long)
    return idx.to(device) if device is not None else idx


def _preprocess_poses_for_rendering(
    poses: torch.Tensor,
    root_idx: int = 1,
    target_bone_len: float = 0.3,
    do_scale_unify: bool = False,
    pairs: Optional[List[Tuple[int, int]]] = None,
) -> torch.Tensor:
    # poses: (T, P, J, 3)
    T, P, J, C = poses.shape
    device = poses.device
    pairs = NTU_PAIRS if pairs is None else pairs
    # 1) optional unify scale per frame using median bone length
    if do_scale_unify:
        pairs0 = [(a - 1, b - 1) for a, b in pairs]
        with torch.no_grad():
            scale_factors = []
            for t in range(T):
                lengths = []
                for p in range(P):
                    sk = poses[t, p]
                    if sk.abs().sum() == 0:
                        continue
                    for a, b in pairs0:
                        if 0 <= a < J and 0 <= b < J:
                            lengths.append((sk[a] - sk[b]).norm().item())
                if len(lengths) == 0:
                    scale_factors.append(1.0)
                else:
                    med = float(torch.tensor(lengths).median().item())
                    s = target_bone_len / med if med > 1e-6 else 1.0
                    scale_factors.append(s)
            s = torch.tensor(scale_factors, dtype=poses.dtype, device=device).view(T, 1, 1, 1)
        poses = poses * s

    # Identify valid skeletons to avoid affecting padded data
    valid_mask = poses.abs().sum(dim=(-1, -2)) > 1e-6  # Shape: (T, P)

    if valid_mask.any():
        z_coords_valid = poses[valid_mask][..., 2]
        if z_coords_valid.numel() > 0:
            z_med = z_coords_valid.median()
            if torch.abs(z_med) < 0.5:
                # Apply offset only to the Z coordinate of valid skeletons
                for t in range(T):
                    for p in range(P):
                        if valid_mask[t, p]:
                            poses[t, p, :, 2] += 1.0
    return poses


def _preprocess_humanml3d_poses(poses: torch.Tensor) -> torch.Tensor:
    """Center HumanML3D Y and place the motion in front of the camera."""
    valid_mask = poses.abs().sum(dim=(-1, -2)) > 1e-6
    if not valid_mask.any():
        return poses

    valid_poses = poses[valid_mask]
    y_offset = -valid_poses[..., 1].median()
    z_offset = 2.5 - valid_poses[..., 2].median()
    offsets = poses.new_tensor([0.0, y_offset.item(), z_offset.item()])
    poses = poses.clone()
    poses[valid_mask] = poses[valid_mask] + offsets
    return poses


def render_skeleton_video_3dgs(
    skeleton_path: str,
    H: int = 448,
    W: int = 448,
    num_line_samples: int = 10,
    adaptive_scales: bool = True,
    adaptive_scale_gamma: float = 1.0,
    base_joint_scale: float = 0.030,
    base_line_scale: float = 0.020,
    min_joint_scale: float = 0.010,
    max_joint_scale: float = 0.035,
    min_line_scale: float = 0.006,
    max_line_scale: float = 0.030,
    fovx_deg: float = 60.0,
    fovy_deg: float = 60.0,
    device: Optional[torch.device] = None,
    target_num_frames: int = 16,
    sample_seed: Optional[int] = None,
    use_pose_rotation: bool = False,
    enable_nfm: bool = False,
) -> torch.Tensor:
    """
    Read an NTU .skeleton file and render it into a tensor of shape (1, T, H, W, 3)
    using a differentiable 3D Gaussian splatter with joints and sampled bone points.
    """
    # Prefer cleaned directory when requested

    poses, metas, orients, skeleton_type = parse_skeleton_file(skeleton_path)
    bone_pairs = get_bone_pairs(skeleton_type)
    T_len, P, J = poses.shape[0], poses.shape[1], poses.shape[2]
    device = device or torch.device('cpu')
    poses = poses.to(device)
    metas = metas.to(device)
    orients = orients.to(device)  # Use orientation data for pose-dependent rotations

    # Temporal sampling to exactly target_num_frames (supports any N>=1)
    if target_num_frames is not None:
        idx = _sample_indices_uniform(T_len, target_num_frames, device=device, seed=sample_seed)
        poses = poses.index_select(0, idx)
        metas = metas.index_select(0, idx)
        orients = orients.index_select(0, idx)
        T_len = idx.numel()

    # Preprocess for stability: centerize XY (keep Z), optional smooth; no scale unify by default
    if skeleton_type == 'humanml3d':
        poses = _preprocess_humanml3d_poses(poses)
    else:
        poses = _preprocess_poses_for_rendering(
            poses,
            root_idx=1,
            target_bone_len=0.3,
            do_scale_unify=False,
            pairs=bone_pairs,
        )

    # Canonical pose as frame 0
    canonical = poses[0, 0]
    line_samples, sample_defs = _build_line_samples(
        canonical,
        num_line_samples=num_line_samples,
        pairs=bone_pairs,
    )
    K_line = line_samples.shape[0]
    K_total = J + K_line

    renderer = DifferentiableSkeletonRenderer(
        num_gaussians=K_total,
        num_joints=J,
        feature_dim=3,
        metadata_dim=metas.shape[-1],
        H=H, W=W,
        use_gsplat=True,
        temporal_stride=4,
        use_temporal_gru=False,
        use_nn_modulation=True,
        enable_nfm=enable_nfm,
        bone_pairs=bone_pairs,
    ).to(device)

    renderer.set_canonical_joints(canonical)

    canonical_means = torch.zeros(K_total, 3, device=device)
    canonical_means[:J] = canonical
    if K_line > 0:
        canonical_means[J:] = line_samples
    renderer.set_canonical_means(canonical_means)

    if adaptive_scales:
        joint_scales_t, line_scales_t = _compute_adaptive_scales(
            canonical,
            sample_defs,
            num_joints=J,
            base_joint_scale=base_joint_scale,
            base_line_scale=base_line_scale,
            min_joint_scale=min_joint_scale,
            max_joint_scale=max_joint_scale,
            min_line_scale=min_line_scale,
            max_line_scale=max_line_scale,
            gamma=adaptive_scale_gamma,
            pairs=bone_pairs,
        )
        renderer.canonical_scales[:J] = joint_scales_t
        if K_line > 0 and line_scales_t.numel() > 0:
            renderer.canonical_scales[J:] = line_scales_t
    else:
        renderer.canonical_scales[:J] = base_joint_scale
        if K_line > 0:
            renderer.canonical_scales[J:] = base_line_scale

    logits = torch.full((K_total, J), -10.0, device=device)
    for j in range(J):
        logits[j, j] = 10.0
    if K_line > 0:
        line_logits = _make_lbs_logits_for_samples(J, sample_defs).to(device)
        logits[J:] = line_logits
    renderer.set_lbs_weights_logits(logits)

    # Build camera intrinsics from FoV
    fovx = math.radians(fovx_deg)
    fovy = math.radians(fovy_deg)
    fx = W / (2.0 * math.tan(fovx * 0.5))
    fy = H / (2.0 * math.tan(fovy * 0.5))
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=torch.float32, device=device)
    K = K.expand(T_len, 3, 3).contiguous()
    w2c = torch.eye(4, device=device, dtype=torch.float32).expand(T_len, 4, 4).contiguous()

    # Compute motion before rendering. The standalone path intentionally calls
    # the same renderer.forward implementation used by model training/inference.
    stride = 4
    idx_fut = torch.clamp(torch.arange(T_len, device=device) + stride, max=T_len - 1)
    vels = poses.index_select(0, idx_fut) - poses
    return renderer(
        poses=poses,
        metas=metas,
        K=K,
        w2c=w2c,
        vels=vels,
        orients=orients if use_pose_rotation else None,
        num_line_samples=num_line_samples,
        adaptive_scales=adaptive_scales,
        adaptive_scale_gamma=adaptive_scale_gamma,
        base_joint_scale=base_joint_scale,
        base_line_scale=base_line_scale,
        min_joint_scale=min_joint_scale,
        max_joint_scale=max_joint_scale,
        min_line_scale=min_line_scale,
        max_line_scale=max_line_scale,
    )


def skeleton_to_pixel_values(
    skeleton_path: str,
    H: int = 448,
    W: int = 448,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convenience wrapper returning pixel_values for chat/batch_chat.

    Returns (1, T, H, W, 3) in [0,1].
    """
    return render_skeleton_video_3dgs(skeleton_path=skeleton_path, H=H, W=W, device=device)
