#!/usr/bin/env python3
"""Minimal model loader for MotionForesight inference."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import torch

import models_pretrained  # noqa: F401  (activates vendored import isolation)
from models_pretrained.future_scene_flow import FutureSceneFlowConfig, FutureSceneFlowModel
from models_pretrained.future_scene_flow.model_fresh_lora import (
    FreshLoRAConfig,
    FreshLoRAFutureSceneFlowModel,
)


def build_model_from_ckpt(checkpoint: Path, base_ckpt: Path):
    """Load a fine-tuned MotionForesight checkpoint and its TrackCraft3r base."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    base_ckpt = Path(base_ckpt).expanduser().resolve()

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        shutil.copy(checkpoint, tmp.name)
        tmp_path = tmp.name
    state = torch.load(tmp_path, map_location="cpu", weights_only=False)
    Path(tmp_path).unlink(missing_ok=True)

    c = state["config"]
    common = dict(
        checkpoint_path=str(base_ckpt),
        lora_rank=c.get("lora_rank", 1024),
        height=c["height"],
        width=c["width"],
        obs_frames=c["obs_frames"],
        total_frames=c["total_frames"],
        trainable=tuple(c.get("trainable", ["mask", "io", "head"])),
        predict_vis=c.get("predict_vis", True),
    )
    if "fresh_lora_rank" in c:
        cfg = FreshLoRAConfig(
            **common,
            fresh_lora_rank=c["fresh_lora_rank"],
            fresh_lora_alpha=c.get("fresh_lora_alpha", c["fresh_lora_rank"]),
            fresh_lora_scope=c.get("fresh_lora_scope", "attn"),
        )
        model = FreshLoRAFutureSceneFlowModel(cfg)
        variant = "fresh"
    else:
        cfg = FutureSceneFlowConfig(
            **common,
            lora_scope=c.get("lora_scope", "attn"),
            lora_last_n=c.get("lora_last_n", 0),
        )
        model = FutureSceneFlowModel(cfg)
        variant = "lora"

    missing, unexpected = model.load_state_dict(state["trainable_state"], strict=False)
    print(
        f"loaded {len(state['trainable_state'])} fine-tuned tensors "
        f"(variant={variant}, unexpected={len(unexpected)}, epoch={state.get('epoch')})"
    )
    if missing:
        print(f"  missing keys: {len(missing)}")
    model.eval()
    return model, state
