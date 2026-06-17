"""Losses and metrics for future 3D point-track prediction."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def trajectory_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    observed_tracks: torch.Tensor,
    velocity_weight: float = 0.2,
    first_step_weight: float = 0.1,
) -> torch.Tensor:
    pos_loss = F.smooth_l1_loss(pred, target)
    vel_loss = F.smooth_l1_loss(pred[:, :, 1:] - pred[:, :, :-1], target[:, :, 1:] - target[:, :, :-1])
    first_vel_pred = pred[:, :, 0] - observed_tracks[:, :, -1]
    first_vel_target = target[:, :, 0] - observed_tracks[:, :, -1]
    first_loss = F.smooth_l1_loss(first_vel_pred, first_vel_target)
    return pos_loss + velocity_weight * vel_loss + first_step_weight * first_loss


@torch.no_grad()
def trajectory_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    track_scale: torch.Tensor,
) -> dict[str, float]:
    err_norm = torch.linalg.norm(pred - target, dim=-1)
    scale = track_scale.reshape(-1, 1, 1)
    err_m = err_norm * scale
    return {
        "ade_norm": float(err_norm.mean().item()),
        "fde_norm": float(err_norm[:, :, -1].mean().item()),
        "ade_m": float(err_m.mean().item()),
        "fde_m": float(err_m[:, :, -1].mean().item()),
    }

