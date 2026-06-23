"""Fresh-LoRA variant of the future scene-flow predictor.

This is a clean, separate alternative to training the checkpoint's existing
rank-1024 LoRA (see ``model.py`` / ``--trainable lora``).  Instead of touching
that large adapter at all, it **stacks a small fresh LoRA adapter** (default
rank 32, ~30 M params) on top of the *frozen* pretrained model and trains only
that.  This is the lightest-weight way to "augment slightly":

    track(t) = DiT_frozen + LoRA_default(frozen, rank 1024) + LoRA_future(trained, rank 32)

Both adapters are active in the forward pass (their deltas sum); only the fresh
``future`` adapter (plus the mask latents and, by default, the I/O projections /
head) receives gradients.  Fits comfortably on a single 24 GB GPU (<1 GB of
extra optimizer/grad memory for the fresh adapter).

Nothing here modifies ``model.py``; it subclasses :class:`FutureSceneFlowModel`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.lora import LoraLayer

from .model import FutureSceneFlowConfig, FutureSceneFlowModel

FRESH_ADAPTER_NAME = "future"


@dataclass
class FreshLoRAConfig(FutureSceneFlowConfig):
    # The fresh adapter that is actually trained.
    fresh_lora_rank: int = 32
    fresh_lora_alpha: int = 32
    # "attn" -> self-attention q,k,v,o only (temporal reasoning; recommended).
    # "all"  -> self-attn q,k,v,o + ffn.0/ffn.2.
    fresh_lora_scope: str = "attn"


class FreshLoRAFutureSceneFlowModel(FutureSceneFlowModel):
    """Pretrained TrackCraft3r + a small trainable fresh LoRA adapter."""

    def __init__(self, config: FreshLoRAConfig) -> None:
        # Base build: loads the checkpoint (incl. the frozen rank-1024 "default"
        # LoRA), creates the mask latents, and configures mask/io/head training.
        # We force the base NOT to train the large LoRA regardless of config.
        base_trainable = tuple(g for g in config.trainable if g != "lora")
        object.__setattr__(config, "trainable", base_trainable)
        super().__init__(config)

        # Inject + wire the fresh adapter, then (re)select trainables.
        self._inject_fresh_adapter(config)
        self._configure_trainable_fresh()

    # ------------------------------------------------------------------
    def _fresh_targets(self, scope: str) -> list[str]:
        if scope == "attn":
            return ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o"]
        return ["self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
                "ffn.0", "ffn.2"]

    def _inject_fresh_adapter(self, config: FreshLoRAConfig) -> None:
        lora_cfg = LoraConfig(
            r=config.fresh_lora_rank,
            lora_alpha=config.fresh_lora_alpha,
            target_modules=self._fresh_targets(config.fresh_lora_scope),
        )
        inject_adapter_in_model(lora_cfg, self.dit, adapter_name=FRESH_ADAPTER_NAME)
        # Activate BOTH adapters on every LoRA layer so their deltas sum.
        n = 0
        for mod in self.dit.modules():
            if isinstance(mod, LoraLayer):
                active = [a for a in ("default", FRESH_ADAPTER_NAME) if a in mod.lora_A]
                mod.set_adapter(active)
                n += 1
        print(f"  fresh LoRA injected: rank={config.fresh_lora_rank}, "
              f"scope={config.fresh_lora_scope}, layers={n}")

    def _configure_trainable_fresh(self) -> None:
        # NOTE: peft's adapter injection re-freezes all non-LoRA params and
        # enables *every* LoRA adapter, so we re-assert the selection here.
        groups = set(self.config.trainable)
        # 1. LoRA: train only the fresh adapter; keep the rank-1024 default frozen.
        for name, p in self.dit.named_parameters():
            if "lora_" in name:
                p.requires_grad_(FRESH_ADAPTER_NAME in name)
        # 2. Re-enable the light I/O / head groups that peft just froze.
        if "io" in groups:
            for p in self.dit.patch_embedding.parameters():
                p.requires_grad_(True)
        if "head" in groups or "io" in groups:
            for p in self.dit.head.parameters():
                p.requires_grad_(True)
        # 3. Mask latents live outside self.dit; peft never touched them.
        self.rgb_mask_latent.requires_grad_(True)
        self.pj_mask_latent.requires_grad_(True)
        # 4. fp32 master copies for the (small) trainable groups for stable AdamW.
        for name, p in self.named_parameters():
            if p.requires_grad and not name.startswith("vae"):
                p.data = p.data.float().to(self.device)
