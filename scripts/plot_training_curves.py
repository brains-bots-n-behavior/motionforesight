#!/usr/bin/env python3
"""Plot train-vs-val ADE and loss curves from one or more training logs.

Parses the per-epoch "epoch NNN done train_loss=.. val_loss=.. val_ade_fut_m=.."
lines and the per-step "epoch NNN step .. loss=.. ade_fut_m=.." lines.
"""
import argparse, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DONE = re.compile(r"epoch (\d+) done train_loss=([\d.]+) val_loss=([\d.]+) val_ade_fut_m=([\d.]+)")
STEP = re.compile(r"epoch (\d+) step \d+ loss=([\d.]+) ade_fut_m=([\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="training curves")
    args = ap.parse_args()
    done, step_ade = {}, defaultdict(list)
    for lg in args.logs:
        try:
            txt = open(lg).read()
        except FileNotFoundError:
            continue
        for m in DONE.finditer(txt):
            done[int(m.group(1))] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
        for m in STEP.finditer(txt):
            step_ade[int(m.group(1))].append(float(m.group(3)))
    if not done:
        raise SystemExit("no epoch-done lines found")
    eps = sorted(done)
    tl = [done[e][0] for e in eps]; vl = [done[e][1] for e in eps]
    vade = [done[e][2] * 100 for e in eps]
    tade = [np.mean(step_ade[e]) * 100 if step_ade.get(e) else np.nan for e in eps]
    best_e = min(eps, key=lambda e: done[e][2])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    ax1.plot(eps, tade, 'o-', color="#2f9e44", label="train ADE (mean of step batches)", ms=4)
    ax1.plot(eps, vade, 's-', color="#c0504d", label="val ADE (held-out)", ms=4)
    ax1.axvline(best_e, ls="--", color="#888", lw=1)
    ax1.scatter([best_e], [done[best_e][2] * 100], c="#c0504d", s=90, zorder=5, edgecolor="k")
    ax1.annotate(f"best val\nepoch {best_e}\n{done[best_e][2]*100:.2f}cm",
                 (best_e, done[best_e][2] * 100), textcoords="offset points", xytext=(8, 12), fontsize=9)
    ax1.set_xlabel("epoch"); ax1.set_ylabel("future-frame ADE (cm)")
    ax1.set_title("ADE: train vs val"); ax1.legend(fontsize=9); ax1.grid(alpha=.3)
    ax2.plot(eps, tl, 'o-', color="#2f9e44", label="train loss", ms=4)
    ax2.plot(eps, vl, 's-', color="#c0504d", label="val loss", ms=4)
    ax2.axvline(best_e, ls="--", color="#888", lw=1)
    ax2.set_xlabel("epoch"); ax2.set_ylabel("decoded loss (MSE x10)")
    ax2.set_title("Loss: train vs val"); ax2.legend(fontsize=9); ax2.grid(alpha=.3)
    fig.suptitle(args.title, fontweight="bold"); fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"epochs {eps[0]}-{eps[-1]} | best val epoch {best_e} {done[best_e][2]*100:.2f}cm | wrote {args.out}")


if __name__ == "__main__":
    main()
