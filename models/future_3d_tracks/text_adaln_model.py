"""Text-conditioned future 3D track predictor with adaLN-Zero blocks.

This variant uses a trainable text encoder over Something-Something labels and
injects the resulting condition through adaptive layer norm, following the
conditioning style used by Diffusion Transformers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from .model import ConvFrameEncoder


@dataclass
class TextAdaLNFutureTrackPredictorConfig:
    obs_frames: int = 10
    future_frames: int = 22
    vocab_size: int = 4096
    max_text_tokens: int = 16
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    text_layers: int = 2
    dropout: float = 0.1


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None, :]) + shift[:, None, :]


class TextTokenEncoder(nn.Module):
    """Small trainable Transformer text encoder for short action labels."""

    def __init__(
        self,
        vocab_size: int,
        max_text_tokens: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.max_text_tokens = max_text_tokens
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(max_text_tokens, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, text_tokens: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        if text_tokens.shape[1] != self.max_text_tokens:
            raise ValueError(f"expected {self.max_text_tokens} text tokens, got {text_tokens.shape[1]}")
        x = self.token_embed(text_tokens) + self.pos_embed[None]
        key_padding_mask = ~text_mask.bool()
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x[:, 0]


class AdaLNZeroBlock(nn.Module):
    """Transformer block whose residual paths are modulated by a condition vector."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim * 6),
        )
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self.ada(cond).chunk(6, dim=-1)
        attn_in = _modulate(self.norm1(x), shift_attn, scale_attn)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + gate_attn[:, None, :] * attn_out
        mlp_in = _modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :] * self.mlp(mlp_in)
        return x


class TextAdaLNFutureTrackPredictor(nn.Module):
    """Predict future 3D tracks with explicit text-token conditioning."""

    def __init__(self, config: TextAdaLNFutureTrackPredictorConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = TextAdaLNFutureTrackPredictorConfig(**kwargs)
        self.config = config
        d = config.embed_dim

        self.frame_encoder = ConvFrameEncoder(config.obs_frames, d, config.dropout)
        self.text_encoder = TextTokenEncoder(
            config.vocab_size,
            config.max_text_tokens,
            d,
            config.num_heads,
            config.text_layers,
            config.dropout,
        )
        self.condition = nn.Sequential(
            nn.Linear(d * 2, d * 4),
            nn.SiLU(),
            nn.Linear(d * 4, d),
        )

        point_in_dim = config.obs_frames * 3 + max(0, config.obs_frames - 1) * 3 + 2
        self.point_encoder = nn.Sequential(
            nn.Linear(point_in_dim, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Dropout(config.dropout),
            nn.Linear(d, d),
            nn.GELU(),
        )
        self.video_to_point = nn.Linear(d, d)
        self.blocks = nn.ModuleList(
            [AdaLNZeroBlock(d, config.num_heads, config.dropout) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(d)
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
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, num_points, obs_frames, xyz = observed_tracks.shape
        if obs_frames != self.config.obs_frames or xyz != 3:
            raise ValueError(
                f"expected observed_tracks B,N,{self.config.obs_frames},3; "
                f"got {tuple(observed_tracks.shape)}"
            )

        video_token = self.frame_encoder(frames)
        text_token = self.text_encoder(text_tokens, text_mask)
        cond = self.condition(torch.cat([video_token, text_token], dim=-1))

        velocity = observed_tracks[:, :, 1:] - observed_tracks[:, :, :-1]
        point_input = torch.cat(
            [
                observed_tracks.reshape(bsz, num_points, -1),
                velocity.reshape(bsz, num_points, -1),
                point_uv,
            ],
            dim=-1,
        )
        point_token = self.point_encoder(point_input) + self.video_to_point(video_token)[:, None]
        for block in self.blocks:
            point_token = block(point_token, cond)
        point_token = self.final_norm(point_token)

        future_token = point_token[:, :, None, :] + self.future_pos[None, None]
        delta = self.head(future_token)
        return observed_tracks[:, :, -1:, :] + delta
