#!/usr/bin/env python3
"""Run SAM3 on the hand-interacted object in each processed OOD clip (3dflow env).

Segments the object (by a per-clip text prompt) on frame 0 of each processed
`_user.npz`, saves the best mask + an overlay for inspection.

Usage:
    python scripts/comparison/sam3_ood_masks.py \
      --proc-dir zero-shot-eval/processed --out-dir zero-shot-eval/masks \
      --prompt 1000055976:cup --prompt 1000055977:dumbbell \
      --prompt 1000055978:"remote control" --prompt 1000055979:handle
"""
from __future__ import annotations
import argparse, glob, os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

REPO = Path(__file__).resolve().parents[2]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-dir", default="zero-shot-eval/processed")
    ap.add_argument("--out-dir", default="zero-shot-eval/masks")
    ap.add_argument("--prompt", action="append", default=[], help="videoid:text (repeatable)")
    ap.add_argument("--sam3-root", default="external/sam3")
    ap.add_argument("--frame", type=int, default=0, help="which observed frame to segment")
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--min-area", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def sam_masks(processor, frame_rgb, prompt):
    with torch.inference_mode():
        state = processor.set_image(Image.fromarray(frame_rgb))
        state = processor.set_text_prompt(prompt=prompt, state=state)
    masks = state["masks"].detach().cpu().numpy()
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    return masks.astype(bool), state["scores"].detach().cpu().numpy()


def main():
    a = parse_args()
    prompts = {}
    for kv in a.prompt:
        vid, txt = kv.split(":", 1)
        prompts[vid] = txt
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    bpe = Path(a.sam3_root) / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    print("loading SAM3 ...", flush=True)
    model = build_sam3_image_model(bpe_path=str(bpe) if bpe.exists() else None,
                                   checkpoint_path=None, load_from_HF=True,
                                   device=a.device, eval_mode=True)
    processor = Sam3Processor(model=model, device=a.device, confidence_threshold=a.conf)

    for f in sorted(glob.glob(os.path.join(a.proc_dir, "*_user.npz"))):
        vid = os.path.basename(f).replace("_user.npz", "")
        prompt = prompts.get(vid)
        if prompt is None:
            print(f"  {vid}: no prompt given, skipping"); continue
        rgb = np.load(f)["rgb"][a.frame]                                # H,W,3
        masks, scores = sam_masks(processor, rgb, prompt)
        # keep best-scoring instance above area threshold
        order = np.argsort(-scores)
        chosen = None
        for i in order:
            if int(masks[i].sum()) >= a.min_area:
                chosen = i; break
        if chosen is None:
            print(f"  {vid}: '{prompt}' -> no mask >= {a.min_area}px (best area "
                  f"{int(masks[order[0]].sum()) if len(order) else 0})"); continue
        m = masks[chosen]
        cv2.imwrite(str(out / f"{vid}_mask.png"), m.astype(np.uint8) * 255)
        ov = rgb.copy(); ov[m] = (0.5 * ov[m] + 0.5 * np.array([40, 230, 90])).astype(np.uint8)
        cv2.imwrite(str(out / f"{vid}_overlay.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
        print(f"  {vid}: '{prompt}' -> mask {int(m.sum())}px score {scores[chosen]:.2f}", flush=True)
    print(f"wrote masks to {out}")


if __name__ == "__main__":
    main()
