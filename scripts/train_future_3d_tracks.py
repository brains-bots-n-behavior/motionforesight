#!/usr/bin/env python3
"""Train a future 3D point-track predictor on curated TrackCraft3r outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.future_3d_tracks.dataset import (  # noqa: E402
    DenseTrackDataset,
    build_track_index,
    split_items,
)
from models.future_3d_tracks.losses import trajectory_loss, trajectory_metrics  # noqa: E402
from models.future_3d_tracks.model import FutureTrackPredictor, FutureTrackPredictorConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/something_something"))
    parser.add_argument("--tracks-name", default="anchor_tracks32_500")
    parser.add_argument("--manifest", type=Path, default=Path("sam3_anchor_masks/manifest_500.json"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--obs-frames", type=int, default=10)
    parser.add_argument("--total-frames", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--image-size", type=int, nargs=2, default=(128, 224), metavar=("H", "W"))
    parser.add_argument("--text-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=200)
    parser.add_argument("--val-steps", type=int, default=25)
    parser.add_argument("--samples-per-clip", type=int, default=4)
    parser.add_argument("--limit-clips", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--velocity-weight", type=float, default=0.2)
    parser.add_argument("--first-step-weight", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use bf16 autocast on CUDA.")
    parser.add_argument("--save-every", type=int, default=1)
    return parser.parse_args()


def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def _mean_dict(dicts: list[dict[str, float]]) -> dict[str, float]:
    keys = dicts[0].keys()
    return {key: float(np.mean([d[key] for d in dicts])) for key in keys}


def evaluate(
    model: FutureTrackPredictor,
    loader: DataLoader,
    device: torch.device,
    max_steps: int,
    amp: bool,
) -> dict[str, float]:
    model.eval()
    rows: list[dict[str, float]] = []
    device_type = "cuda" if device.type == "cuda" else "cpu"
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= max_steps:
                break
            batch = _to_device(batch, device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
                pred = model(
                    batch["frames"],
                    batch["observed_tracks"],
                    batch["point_uv"],
                    batch["text_bow"],
                )
                loss = trajectory_loss(pred, batch["future_tracks"], batch["observed_tracks"])
            metrics = trajectory_metrics(pred.float(), batch["future_tracks"], batch["track_scale"])
            metrics["loss"] = float(loss.item())
            rows.append(metrics)
    return _mean_dict(rows) if rows else {"loss": float("nan"), "ade_norm": float("nan"), "fde_norm": float("nan"), "ade_m": float("nan"), "fde_m": float("nan")}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)

    root = args.root.expanduser().resolve()
    output_dir = args.output_dir
    if output_dir is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = root / "future_track_training" / stamp
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    items = build_track_index(
        root=root,
        tracks_name=args.tracks_name,
        manifest=args.manifest,
        require_masks=True,
        limit_clips=args.limit_clips,
    )
    if len(items) < 2:
        raise SystemExit(f"need at least 2 dense clips with masks; found {len(items)}")
    train_items, val_items = split_items(items, args.val_fraction, args.seed)
    print(f"indexed {len(items)} clips: train={len(train_items)} val={len(val_items)}")

    train_ds = DenseTrackDataset(
        train_items,
        obs_frames=args.obs_frames,
        total_frames=args.total_frames,
        num_points=args.num_points,
        image_size=tuple(args.image_size),
        text_dim=args.text_dim,
        samples_per_clip=args.samples_per_clip,
        seed=args.seed,
    )
    val_ds = DenseTrackDataset(
        val_items,
        obs_frames=args.obs_frames,
        total_frames=args.total_frames,
        num_points=args.num_points,
        image_size=tuple(args.image_size),
        text_dim=args.text_dim,
        samples_per_clip=1,
        seed=args.seed + 100_000,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    config = FutureTrackPredictorConfig(
        obs_frames=args.obs_frames,
        future_frames=args.total_frames - args.obs_frames,
        text_dim=args.text_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = FutureTrackPredictor(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_config = vars(args).copy()
    run_config["root"] = str(root)
    run_config["manifest"] = str(args.manifest)
    run_config["output_dir"] = str(output_dir)
    run_config["model"] = model.config_dict
    (output_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    best_ade = float("inf")
    global_step = 0
    device_type = "cuda" if device.type == "cuda" else "cpu"
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        start = time.time()
        for step, batch in enumerate(train_loader, start=1):
            if step > args.steps_per_epoch:
                break
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
                pred = model(
                    batch["frames"],
                    batch["observed_tracks"],
                    batch["point_uv"],
                    batch["text_bow"],
                )
                loss = trajectory_loss(
                    pred,
                    batch["future_tracks"],
                    batch["observed_tracks"],
                    velocity_weight=args.velocity_weight,
                    first_step_weight=args.first_step_weight,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1
            losses.append(float(loss.item()))

            if step == 1 or step % 20 == 0:
                metrics = trajectory_metrics(pred.detach().float(), batch["future_tracks"], batch["track_scale"])
                print(
                    f"epoch {epoch:03d} step {step:04d} "
                    f"loss={loss.item():.4f} ade_m={metrics['ade_m']:.4f} fde_m={metrics['fde_m']:.4f}"
                )

        val_metrics = evaluate(model, val_loader, device, args.val_steps, args.amp)
        train_loss = float(np.mean(losses)) if losses else float("nan")
        elapsed = time.time() - start
        print(
            f"epoch {epoch:03d} done train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_ade_m={val_metrics['ade_m']:.4f} "
            f"val_fde_m={val_metrics['fde_m']:.4f} time={elapsed:.1f}s"
        )

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": model.config_dict,
            "epoch": epoch,
            "global_step": global_step,
            "val_metrics": val_metrics,
        }
        torch.save(state, output_dir / "last.pt")
        if val_metrics["ade_m"] < best_ade:
            best_ade = val_metrics["ade_m"]
            torch.save(state, output_dir / "best.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(state, output_dir / f"epoch_{epoch:03d}.pt")

    print(f"wrote checkpoints to {output_dir}")


if __name__ == "__main__":
    main()

