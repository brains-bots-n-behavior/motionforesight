"""Future 3D scene-flow predictor that reuses the pretrained TrackCraft3r model.

Design (minimal change to TrackCraft3r)
---------------------------------------
TrackCraft3r maps a *fully observed* T-frame clip to dense per-pixel 3D tracks
in frame-0 camera space.  Internally (``model_fn_wan_video``) it builds two
streams over T latent frames and predicts the "row" stream:

    diagonal[t] = [ RGB_latent(t) | Pj_latent(t) ]      # appearance + geometry @ t
    row[t]      = [ RGB_latent(0) | Pj_latent(0) ]      # frame-0 anchor (query)
    track(t)    = DiT( concat_T(diagonal, row) )        # where frame-0 point goes @ t

For *future* prediction we observe only frames ``0 .. obs-1``.  We keep the
entire pretrained pipeline and make exactly one structural change: the diagonal
entries for the unobserved future frames ``obs .. T-1`` are replaced by two
small **learnable mask latents** (one for the RGB half, one for the Pj half).
The frozen DiT then attends from the future query rows (still the frame-0
anchor) to the observed diagonal entries and the mask tokens, and regresses the
future tracks.  RoPE already gives every future frame a distinct temporal
position, so a single shared mask latent per stream is enough.

Because TrackCraft3r's own dense outputs (``*_dense.npz``) already store the
exact tensors the pipeline consumes — ``rgb`` and ``recon_map`` (= Pj in cam-0
space) — and the supervision target ``track_map`` (= GT P0(t)), no depth /
camera recomputation is needed: we feed the observed ``rgb``/``recon_map`` and
supervise the future ``track_map``.

Nothing in ``external/TrackCraft3r`` is modified; this uses the vendored copy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

import models_pretrained  # noqa: F401  (vendored-import isolation)
from evaluation.wan_scene_flow_predictor import WanSceneFlowPredictor


# Groups of parameters that may be made trainable.  The base DiT, the loaded
# LoRA and all VAEs are frozen unless explicitly enabled.
TRAINABLE_GROUPS = ("mask", "io", "head", "lora", "vae")


@dataclass
class FutureSceneFlowConfig:
    checkpoint_path: str
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    lora_rank: int = 1024
    lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2"
    height: int = 224
    width: int = 384
    obs_frames: int = 10
    total_frames: int = 32
    regression_timestep: int = -1
    # Which parameter groups to fine-tune.  "mask" (the future mask latents) is
    # always trained.  Default also trains the I/O projections + head, which is
    # light enough for a single 24 GB GPU.  Add "lora" for full adaptation on a
    # larger GPU.
    trainable: tuple[str, ...] = ("mask", "io", "head")
    # Which LoRA params to train when "lora" is in ``trainable``:
    #   "all"  -> every LoRA tensor (~1.0B params; needs a big GPU / 8-bit optim)
    #   "attn" -> self-attention q,k,v,o LoRA only (~0.38B; fits a 24 GB GPU).
    #             Cross-attention is skipped because it attends to the (null)
    #             text prompt and carries no temporal signal here.
    lora_scope: str = "attn"
    lora_last_n: int = 0  # if >0, restrict LoRA training to the last N DiT blocks
    predict_vis: bool = True

    @property
    def future_frames(self) -> int:
        return self.total_frames - self.obs_frames


class FutureSceneFlowModel(nn.Module):
    """Pretrained TrackCraft3r + learnable future mask latents.

    Forward inputs (all on the model device):
        rgb_obs : (B, obs, 3, H, W) in [-1, 1]   observed RGB frames
        pj_obs  : (B, obs, 3, H, W)              observed normalized Pj(t) maps
    Output:
        query_latent : (B, 2*z, T, Hl, Wl)  predicted [xyz | vis] track latents
                                             for all ``total_frames``.
    Use :meth:`split_latent`, :meth:`decode_xyz` and :meth:`reconstruct` to turn
    the xyz half into metric 3D tracks, exactly as the eval predictor does.
    """

    def __init__(self, config: FutureSceneFlowConfig) -> None:
        super().__init__()
        self.config = config

        # 1. Build the full TrackCraft3r pipeline and load the release weights.
        #    WanSceneFlowPredictor does all the heavy lifting: LoRA injection,
        #    patch_embedding 16->32 expansion, scene_flow flags, head vis
        #    expansion, deep-copied vae_pj / vae_vis, checkpoint loading, and
        #    moving everything to the device in bf16.
        predictor = WanSceneFlowPredictor(
            checkpoint_path=config.checkpoint_path,
            model_id=config.model_id,
            lora_rank=config.lora_rank,
            lora_target_modules=config.lora_target_modules,
            height=config.height,
            width=config.width,
            device="cuda" if torch.cuda.is_available() else "cpu",
            regression_timestep=config.regression_timestep,
            predict_vis=config.predict_vis,
            parallel_vae_decode=False,  # single-stream is friendlier for training
        )
        self.predictor = predictor
        self.pipe = predictor.pipe
        self.device = torch.device(predictor.device)
        self.z_dim = int(self.pipe.vae.z_dim)

        # Register the pretrained submodules so .to()/.train()/.state_dict() see
        # them, but they are frozen by default (see _configure_trainable).
        self.dit = self.pipe.dit
        self.vae = self.pipe.vae
        self.vae_pj = self.pipe.vae_pj if self.pipe.vae_pj is not None else self.pipe.vae
        self.vae_vis = self.pipe.vae_vis

        # 2. Learnable future mask latents (one per input stream), broadcast over
        #    space and over all future frames.  Small; this is the only genuinely
        #    new structural parameter.
        self.rgb_mask_latent = nn.Parameter(torch.zeros(1, self.z_dim, 1, 1, 1))
        self.pj_mask_latent = nn.Parameter(torch.zeros(1, self.z_dim, 1, 1, 1))
        nn.init.normal_(self.rgb_mask_latent, std=0.02)
        nn.init.normal_(self.pj_mask_latent, std=0.02)

        # 3. Cached null text context + regression timestep (from the predictor).
        self._null_context = predictor._null_context
        # We only ever use the cached null-text context, so drop the (CPU-resident)
        # T5 text encoder: keeps it out of .parameters() so the model is on a single
        # device (required by DistributedDataParallel) and frees host RAM.
        if getattr(self.pipe, "text_encoder", None) is not None:
            self.pipe.text_encoder = None
        if getattr(getattr(self.pipe, "prompter", None), "text_encoder", None) is not None:
            self.pipe.prompter.text_encoder = None
        self.register_buffer(
            "regression_timestep_value",
            self.pipe.scheduler.timesteps[config.regression_timestep].clone(),
            persistent=False,
        )

        # 4. Freeze everything, then unfreeze the selected groups.
        self._configure_trainable(config.trainable)

        # 5. Move the newly-created params/buffers to the pipeline device (the
        #    pretrained pipe modules are already there; these are not). Moving ALL
        #    buffers to the device keeps the module single-device for DDP.
        self.rgb_mask_latent.data = self.rgb_mask_latent.data.to(self.device)
        self.pj_mask_latent.data = self.pj_mask_latent.data.to(self.device)
        for _b in self.buffers():
            _b.data = _b.data.to(self.device)

    # ------------------------------------------------------------------ setup
    def _configure_trainable(self, groups) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        # mask latents are always trained
        self.rgb_mask_latent.requires_grad_(True)
        self.pj_mask_latent.requires_grad_(True)

        groups = set(groups)
        if "io" in groups:
            for p in self.dit.patch_embedding.parameters():
                p.requires_grad_(True)
        if "head" in groups or "io" in groups:
            for p in self.dit.head.parameters():
                p.requires_grad_(True)
        if "lora" in groups:
            for name, p in self.dit.named_parameters():
                if self._lora_selected(name):
                    p.requires_grad_(True)
        if "vae" in groups:
            for p in self.vae.parameters():
                p.requires_grad_(True)
            for p in self.vae_pj.parameters():
                p.requires_grad_(True)

        # Keep small fp32 master copies for the always-light groups (mask + I/O
        # + head) for stable AdamW updates.  LoRA / VAE stay bf16 to save memory.
        for name, p in self.named_parameters():
            if p.requires_grad and "lora_" not in name and not name.startswith("vae"):
                p.data = p.data.float()

    def _lora_selected(self, name: str) -> bool:
        """Whether a DiT parameter ``name`` is a LoRA tensor we want to train."""
        if "lora_" not in name:
            return False
        if self.config.lora_scope == "attn" and ".self_attn." not in name:
            return False
        if self.config.lora_last_n > 0:
            # names look like "blocks.<idx>.self_attn.q.lora_A.default.weight"
            parts = name.split(".")
            try:
                idx = int(parts[parts.index("blocks") + 1])
            except (ValueError, IndexError):
                return True  # non-block LoRA (none expected) -> keep
            n_layers = len(self.dit.blocks)
            if idx < n_layers - self.config.lora_last_n:
                return False
        return True

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    @property
    def config_dict(self) -> dict:
        d = asdict(self.config)
        d["trainable"] = list(self.config.trainable)
        return d

    def train(self, mode: bool = True):  # noqa: D401
        """Set train mode but keep frozen VAEs (and frozen DiT if applicable)
        in eval so their norm/dropout stats stay fixed."""
        super().train(mode)
        if "vae" not in set(self.config.trainable):
            self.vae.eval()
            self.vae_pj.eval()
            if self.vae_vis is not None:
                self.vae_vis.eval()
        return self

    # --------------------------------------------------------------- encoding
    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor, vae) -> torch.Tensor:
        """(B, Tf, 3, H, W) -> (B, z, Tf, Hl, Wl) per-frame VAE latents."""
        b, tf, c, h, w = frames.shape
        flat = frames.reshape(b * tf, c, 1, h, w).to(self.device, torch.bfloat16)
        lat = vae.encode(flat, device=self.device, tiled=False)
        lat = lat.to(torch.bfloat16)
        _, cz, _, hl, wl = lat.shape
        return lat.reshape(b, tf, cz, hl, wl).transpose(1, 2)  # B, z, Tf, Hl, Wl

    @torch.no_grad()
    def encode_target_delta(self, target_delta: torch.Tensor) -> torch.Tensor:
        """Encode the normalized residual track maps (B, T, 3, H, W) -> latents.

        The xyz head decodes to the normalized residual ``track_norm - P0(0)``;
        the latent-space target is the VAE encoding of that residual via the
        primary (xyz) VAE.
        """
        return self.encode_frames(target_delta, self.vae)

    # ---------------------------------------------------------------- forward
    def _build_input_streams(self, rgb_obs, pj_obs):
        """Encode observed frames and pad the future with learnable mask latents.

        Returns rgb_latents, pj_latents each (B, z, total_frames, Hl, Wl).
        """
        rgb_obs_lat = self.encode_frames(rgb_obs, self.vae)          # B,z,obs,Hl,Wl
        pj_obs_lat = self.encode_frames(pj_obs, self.vae_pj)         # B,z,obs,Hl,Wl
        b, z, _, hl, wl = rgb_obs_lat.shape
        fut = self.config.future_frames

        rgb_mask = self.rgb_mask_latent.to(rgb_obs_lat.dtype).expand(b, z, fut, hl, wl)
        pj_mask = self.pj_mask_latent.to(pj_obs_lat.dtype).expand(b, z, fut, hl, wl)

        rgb_latents = torch.cat([rgb_obs_lat, rgb_mask], dim=2)
        pj_latents = torch.cat([pj_obs_lat, pj_mask], dim=2)
        return rgb_latents, pj_latents

    def forward(self, rgb_obs, pj_obs, use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False):
        rgb_latents, pj_latents = self._build_input_streams(rgb_obs, pj_obs)
        b = rgb_latents.shape[0]
        context = self._null_context.expand(b, -1, -1)
        timestep = self.regression_timestep_value.reshape(1).to(
            dtype=torch.bfloat16, device=self.device
        )
        query_latent = self.pipe.model_fn(
            dit=self.dit,
            latents=rgb_latents,
            timestep=timestep,
            context=context,
            cfg_merge=False,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            pj_latents=pj_latents,
        )
        return query_latent  # B, 2z (or z), T, Hl, Wl

    # ------------------------------------------------------------ decode utils
    def split_latent(self, query_latent):
        """Return (xyz_latent, vis_latent_or_None)."""
        if self.config.predict_vis and query_latent.shape[1] == 2 * self.z_dim:
            xyz, vis = query_latent.chunk(2, dim=1)
            return xyz, vis
        return query_latent, None

    def decode_xyz(self, xyz_latent, mini_batch: int = 12) -> torch.Tensor:
        """(B, z, T, Hl, Wl) -> (B, 3, T, H, W) normalized residual delta (no grad)."""
        return self.predictor._decode_latents(xyz_latent, mini_batch=mini_batch)

    def decode_xyz_grad(self, xyz_latent, use_checkpoint: bool = True) -> torch.Tensor:
        """Grad-enabled per-frame VAE decode for TrackCraft3r-style decoded loss.

        Decodes one latent frame at a time; with ``use_checkpoint`` each frame's
        decoder activations are recomputed in backward instead of stored, so the
        32-frame decode fits in the autograd graph on a 24 GB GPU.

        (B, z, T, Hl, Wl) -> (B, 3, T, H, W).
        """
        b, c, t, hl, wl = xyz_latent.shape

        def _dec(lat1):  # lat1: (B, z, Hl, Wl)
            out = self.vae.decode(lat1.unsqueeze(2), device=self.device, tiled=False)
            return out.squeeze(2)  # (B, 3, H, W)

        frames = []
        for ti in range(t):
            lat1 = xyz_latent[:, :, ti]
            if use_checkpoint and torch.is_grad_enabled():
                out = torch.utils.checkpoint.checkpoint(_dec, lat1, use_reentrant=False)
            else:
                out = _dec(lat1)
            frames.append(out)
        return torch.stack(frames, dim=2)  # (B, 3, T, H, W)

    @staticmethod
    def reconstruct(delta, p0_t0_norm):
        """Normalized residual delta + frame-0 anchor -> normalized track.

        delta       : (B, 3, T, H, W)
        p0_t0_norm  : (B, 3, H, W)   normalized Pj at frame 0
        returns       (B, 3, T, H, W) normalized track in cam-0 space.
        """
        return delta + p0_t0_norm[:, :, None]

    @staticmethod
    def denormalize(track_norm, pj_mean, pj_scale):
        """(B,3,T,H,W) normalized -> metric cam-0 coordinates."""
        return track_norm * pj_scale.view(-1, 1, 1, 1, 1) + pj_mean.view(-1, 3, 1, 1, 1)
