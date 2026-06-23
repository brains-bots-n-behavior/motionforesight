"""Losses and metrics for future-scene-flow training.

Training objective (default): latent-space regression of the predicted xyz
query latent toward the VAE encoding of the GT normalized residual track maps —
this is exactly the signal TrackCraft3r itself regresses, and it avoids
back-propagating through the (frozen) VAE decoder, which keeps a 1.3B DiT over
32 frames trainable on a single 24 GB GPU.

Eval metrics are computed in metric (meters) space by decoding the predicted
latent, reconstructing the 3D track, sampling at the object-mask points, and
comparing to the GT track — matching the from-scratch baseline's ADE/FDE.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def latent_loss(
    pred_latent: torch.Tensor,   # B, z, T, Hl, Wl
    target_latent: torch.Tensor,  # B, z, T, Hl, Wl
    obs_frames: int,
    future_weight: float = 1.0,
    obs_weight: float = 0.25,
) -> torch.Tensor:
    """Frame-weighted MSE between predicted and target xyz latents."""
    pred_latent = pred_latent.float()
    target_latent = target_latent.float()
    err = (pred_latent - target_latent).pow(2).mean(dim=(0, 1, 3, 4))  # per frame T
    obs = err[:obs_frames].mean() if obs_frames > 0 else err.new_zeros(())
    fut = err[obs_frames:].mean() if err.shape[0] > obs_frames else err.new_zeros(())
    return obs_weight * obs + future_weight * fut


def decoded_loss(
    model,
    query_latent: torch.Tensor,
    batch: dict,
    obs_frames: int,
    future_weight: float = 1.0,
    obs_weight: float = 0.25,
    scale: float = 10.0,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    """TrackCraft3r-faithful loss: decode the latent to a 3D pointmap and take
    MSE in coordinate space (validity-masked, ×scale), frame-weighted.

    Mirrors ``WanVideoPipeline.training_loss`` (decode -> masked MSE ×10), but
    masks by track validity instead of GT visibility (our dense clips have no
    visibility channel) and weights observed vs future frames.
    """
    xyz, _ = model.split_latent(query_latent)
    delta = model.decode_xyz_grad(xyz, use_checkpoint=use_checkpoint)   # B,3,T,h,w
    p0 = batch["p0_t0_norm"].to(delta.device)
    pred = model.reconstruct(delta, p0).float()                        # B,3,T,h,w
    gt = batch["track_norm"].to(delta.device).float().permute(0, 2, 1, 3, 4)  # B,3,T,h,w
    mask = batch["valid"].to(delta.device).unsqueeze(1).float()        # B,1,T,h,w

    diff = (pred - gt).pow(2) * mask                                   # B,3,T,h,w
    denom = mask.expand(-1, 3, -1, -1, -1).sum(dim=(0, 1, 3, 4)) + 1e-6  # T
    per_frame = diff.sum(dim=(0, 1, 3, 4)) / denom                     # T
    obs = per_frame[:obs_frames].mean() if obs_frames > 0 else per_frame.new_zeros(())
    fut = per_frame[obs_frames:].mean() if per_frame.shape[0] > obs_frames else per_frame.new_zeros(())
    return (obs_weight * obs + future_weight * fut) * scale


@torch.no_grad()
def decoded_metrics(
    model,
    query_latent: torch.Tensor,
    batch: dict,
    obs_frames: int,
    max_points: int = 256,
) -> dict[str, float]:
    """ADE/FDE in meters over object-mask points, for future and all frames."""
    xyz_lat, _ = model.split_latent(query_latent)
    delta = model.decode_xyz(xyz_lat).float()                       # B,3,T,h,w
    p0 = batch["p0_t0_norm"].to(delta.device).float()               # B,3,h,w
    pred_norm = model.reconstruct(delta, p0)                        # B,3,T,h,w
    pj_mean = batch["pj_mean"].to(delta.device).float()
    pj_scale = batch["pj_scale"].to(delta.device).float()
    pred_m = model.denormalize(pred_norm, pj_mean, pj_scale)        # B,3,T,h,w
    # track_norm is stored frame-major (B,T,3,h,w) -> channel-major (B,3,T,h,w).
    gt_norm = batch["track_norm"].to(delta.device).float().permute(0, 2, 1, 3, 4)
    gt_m = model.denormalize(gt_norm, pj_mean, pj_scale)

    valid = batch["valid"].to(delta.device)                         # B,T,h,w
    obj = batch["obj_mask"].to(delta.device)                        # B,h,w
    B, _, T, h, w = pred_m.shape

    ade_all, fde_all, ade_fut, fde_fut = [], [], [], []
    for b in range(B):
        pix_valid = valid[b].all(dim=0)                            # h,w  valid all frames
        cand = (obj[b] & pix_valid)
        if cand.sum() < 8:
            cand = pix_valid
        ys, xs = torch.where(cand)
        if ys.numel() == 0:
            continue
        if ys.numel() > max_points:
            sel = torch.randperm(ys.numel(), device=ys.device)[:max_points]
            ys, xs = ys[sel], xs[sel]
        pred_pts = pred_m[b, :, :, ys, xs]                         # 3,T,N
        gt_pts = gt_m[b, :, :, ys, xs]
        d = (pred_pts - gt_pts).pow(2).sum(0).sqrt()              # T,N
        ade_all.append(d.mean().item())
        fde_all.append(d[-1].mean().item())
        ade_fut.append(d[obs_frames:].mean().item())
        fde_fut.append(d[-1].mean().item())

    def _m(x):
        return float(sum(x) / len(x)) if x else float("nan")

    return {
        "ade_m": _m(ade_all),
        "fde_m": _m(fde_all),
        "ade_future_m": _m(ade_fut),
        "fde_future_m": _m(fde_fut),
    }
