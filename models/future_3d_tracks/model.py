"""Future 3D point-track predictor.

This is a repo-local, trainable architecture inspired by the TrackCraft3r data
representation.  It does not edit the TrackCraft3r submodule.  TrackCraft3r
dense maps provide pseudo-labels; this model only consumes observed frames and
observed point histories.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class FutureTrackPredictorConfig:
    obs_frames: int = 10
    future_frames: int = 22
    text_dim: int = 256
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1


class ConvFrameEncoder(nn.Module):
    def __init__(self, obs_frames: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.obs_frames = obs_frames
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.frame_pos = nn.Parameter(torch.randn(obs_frames, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        bsz, frames_t, channels, height, width = frames.shape
        if frames_t != self.obs_frames:
            raise ValueError(f"expected {self.obs_frames} frames, got {frames_t}")
        x = frames.reshape(bsz * frames_t, channels, height, width)
        feat = self.cnn(x).flatten(1).reshape(bsz, frames_t, -1)
        feat = feat + self.frame_pos[None]
        feat = self.temporal(feat)
        return self.norm(feat.mean(dim=1))


class FutureTrackPredictor(nn.Module):
    """Predicts future 3D positions for sampled points.

    Inputs:
        frames: B,Tobs,3,H,W, only observed video frames.
        observed_tracks: B,N,Tobs,3, normalized observed 3D point history.
        point_uv: B,N,2, frame-0 point coordinates normalized to [-1, 1].
        text_bow: B,text_dim, hashed text prompt vector.

    Output:
        B,N,Tfuture,3 normalized future 3D point positions.
    """

    def __init__(self, config: FutureTrackPredictorConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = FutureTrackPredictorConfig(**kwargs)
        self.config = config
        d = config.embed_dim

        self.frame_encoder = ConvFrameEncoder(config.obs_frames, d, config.dropout)
        point_in_dim = config.obs_frames * 3 + max(0, config.obs_frames - 1) * 3 + 2
        self.point_encoder = nn.Sequential(
            nn.Linear(point_in_dim, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(config.dropout),
            nn.Linear(d, d),
            nn.GELU(),
        )
        self.text_encoder = nn.Sequential(
            nn.Linear(config.text_dim, d),
            nn.GELU(),
            nn.LayerNorm(d),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=config.num_heads,
            dim_feedforward=d * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.point_context = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.future_pos = nn.Parameter(torch.randn(config.future_frames, d) * 0.02)
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, 3),
        )

    @property
    def config_dict(self) -> dict:
        return asdict(self.config)

    def forward(
        self,
        frames: torch.Tensor,
        observed_tracks: torch.Tensor,
        point_uv: torch.Tensor,
        text_bow: torch.Tensor,
    ) -> torch.Tensor:
        bsz, num_points, obs_frames, xyz = observed_tracks.shape
        if obs_frames != self.config.obs_frames or xyz != 3:
            raise ValueError(
                f"expected observed_tracks B,N,{self.config.obs_frames},3; "
                f"got {tuple(observed_tracks.shape)}"
            )
        video_token = self.frame_encoder(frames)
        text_token = self.text_encoder(text_bow)

        velocity = observed_tracks[:, :, 1:] - observed_tracks[:, :, :-1]
        point_input = torch.cat(
            [
                observed_tracks.reshape(bsz, num_points, -1),
                velocity.reshape(bsz, num_points, -1),
                point_uv,
            ],
            dim=-1,
        )
        point_token = self.point_encoder(point_input)
        point_token = point_token + video_token[:, None] + text_token[:, None]
        point_token = self.point_context(point_token)

        future_token = point_token[:, :, None, :] + self.future_pos[None, None]
        delta = self.head(future_token)
        last_observed = observed_tracks[:, :, -1:, :]
        return last_observed + delta

