#!/usr/bin/env python3
"""Log GT|pred val videos SIDE BY SIDE to wandb as a browsable Table.

Reads the sharded render_unified_viewer outputs (shard*/manifest.json + webm),
hconcats each clip's GT and pred trail videos into one frame, and logs a sample
spanning the ADE distribution (best / median / worst tiers) to a wandb run.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import imageio.v2 as iio
import wandb


def frames(path):
    r = iio.get_reader(str(path))
    fs = [f[..., :3] for f in r]
    r.close()
    return fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="viewer_val_all dir with shard*/")
    ap.add_argument("--project", default="future-scene-flow")
    ap.add_argument("--run-name", default="val_eval_best_ep24")
    ap.add_argument("--sample", type=int, default=90, help="total clips (split across 3 tiers)")
    ap.add_argument("--fps", type=int, default=8)
    a = ap.parse_args()
    base = Path(a.base).resolve()

    items = []
    for shard in sorted(base.glob("shard*")):
        mf = shard / "manifest.json"
        if mf.exists():
            for it in json.loads(mf.read_text()).get("items", []):
                it["shard"] = shard.name
                items.append(it)
    items.sort(key=lambda x: x["ade"])
    n = len(items)
    ade = np.array([it["ade"] for it in items]) * 100
    fde = np.array([it["fde"] for it in items]) * 100
    print(f"loaded {n} val clips | ADE mean {ade.mean():.2f}cm median {np.median(ade):.2f} | "
          f"FDE mean {fde.mean():.2f}cm")

    k = max(1, a.sample // 3)
    picks = ([(it, "best") for it in items[:k]]
             + [(it, "median") for it in items[n // 2 - k // 2: n // 2 - k // 2 + k]]
             + [(it, "worst") for it in items[-k:]])

    media = base / "wandb_sidebyside"; media.mkdir(exist_ok=True)
    run = wandb.init(project=a.project, name=a.run_name, job_type="eval",
                     config={"n_val": n, "ade_cm_mean": float(ade.mean()),
                             "ade_cm_median": float(np.median(ade)),
                             "fde_cm_mean": float(fde.mean())})
    tbl = wandb.Table(columns=["tier", "video_id", "label", "ade_cm", "fde_cm", "GT | prediction"])
    for it, tier in picks:
        try:
            gt = frames(base / it["shard"] / it["gtVideo"])
            pr = frames(base / it["shard"] / it["predVideo"])
            T = min(len(gt), len(pr))
            comb = [np.concatenate([gt[t], pr[t]], axis=1) for t in range(T)]
            mp4 = media / f"{it['videoId']}.mp4"
            iio.mimwrite(str(mp4), comb, fps=a.fps, quality=8, macro_block_size=None)
            tbl.add_data(tier, it["videoId"], it.get("title", ""),
                         round(it["ade"] * 100, 1), round(it["fde"] * 100, 1),
                         wandb.Video(str(mp4), fps=a.fps, format="mp4"))
        except Exception as e:
            print(f"  skip {it['videoId']}: {e}")
    # summary scalars for the full val set
    run.summary["full_val/ade_cm_mean"] = float(ade.mean())
    run.summary["full_val/ade_cm_median"] = float(np.median(ade))
    run.summary["full_val/fde_cm_mean"] = float(fde.mean())
    run.summary["full_val/n"] = n
    wandb.log({"val_predictions_side_by_side": tbl})
    wandb.finish()
    print(f"logged {len(picks)} side-by-side clips to wandb run '{a.run_name}'")


if __name__ == "__main__":
    main()
