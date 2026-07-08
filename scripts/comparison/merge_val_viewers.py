#!/usr/bin/env python3
"""Merge sharded render_unified_viewer outputs into ONE full-val viewer.

Each shard dir has manifest.json ({"items":[...]}) + videos/. This concatenates
all items (rewriting video paths to shardN/videos/...), recomputes the aggregate
ADE/FDE over the full val set, and writes a single index.html. Videos use
preload="none" so 2000+ clips stay loadable in a browser."""
from __future__ import annotations
import argparse, html, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="dir containing shard0..shardN")
    ap.add_argument("--out", default=None, help="output dir (default: --base)")
    ap.add_argument("--sort", choices=["worst", "best", "id"], default="worst")
    a = ap.parse_args()
    base = Path(a.base).resolve()
    out = Path(a.out).resolve() if a.out else base

    items = []
    for shard in sorted(base.glob("shard*")):
        mf = shard / "manifest.json"
        if not mf.exists():
            print(f"  skip {shard.name}: no manifest.json")
            continue
        data = json.loads(mf.read_text()).get("items", [])
        for it in data:
            it = dict(it)
            it["gtVideo"] = f"{shard.name}/{it['gtVideo']}"
            it["predVideo"] = f"{shard.name}/{it['predVideo']}"
            items.append(it)
        print(f"  {shard.name}: {len(data)} clips")

    if not items:
        raise SystemExit("no items found across shards")
    ade = np.array([it["ade"] for it in items]) * 100
    fde = np.array([it["fde"] for it in items]) * 100
    n = len(items)
    header = (f"FULL VAL — {n} clips | ADE {ade.mean():.2f}cm (median {np.median(ade):.2f}) "
              f"· FDE {fde.mean():.2f}cm (median {np.median(fde):.2f})")
    print("\n" + header)

    if a.sort == "worst":
        items.sort(key=lambda x: -x["ade"])
    elif a.sort == "best":
        items.sort(key=lambda x: x["ade"])
    else:
        items.sort(key=lambda x: x["videoId"])

    css = """:root{color-scheme:dark}body{margin:0;background:#101214;color:#f3f5f7;font-family:system-ui,sans-serif}
    header{position:sticky;top:0;background:rgba(16,18,20,.97);border-bottom:1px solid #2a3037;padding:14px 20px;z-index:9}
    h1{font-size:16px;margin:0 0 6px}header p{margin:0;color:#a6afb8;font-size:13px}main{padding:16px 20px;display:grid;grid-template-columns:repeat(auto-fill,minmax(460px,1fr));gap:14px}
    .card{border:1px solid #2a3037;background:#181b1f;border-radius:8px;overflow:hidden}
    .meta{padding:9px 12px;border-bottom:1px solid #2a3037}.meta h2{font-size:13px;margin:0}.meta p{font-size:12px;color:#a6afb8;margin:3px 0 0}
    .v{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#2a3037}.pane{background:#090a0b;padding:6px}
    .pane h3{font-size:10px;text-transform:uppercase;color:#a6afb8;margin:0 0 4px}video{width:100%;background:#000;border-radius:4px}"""
    cards = []
    for it in items:
        cards.append(
            f'<article class="card"><div class="meta"><h2>{html.escape(it["title"])}</h2>'
            f'<p>{html.escape(it["videoId"])} · {it["numPoints"]} pts · ADE {it["ade"]*100:.1f}cm · FDE {it["fde"]*100:.1f}cm</p></div>'
            f'<div class="v"><section class="pane"><h3>Ground Truth</h3>'
            f'<video src="{it["gtVideo"]}" controls muted loop preload="none"></video></section>'
            f'<section class="pane"><h3>Prediction</h3>'
            f'<video src="{it["predVideo"]}" controls muted loop preload="none"></video></section></div></article>')
    doc = (f'<!doctype html><html><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1">'
           f'<title>Full val — future 3D tracks</title><style>{css}</style></head><body>'
           f'<header><h1>Future 3D tracks — full validation set (best.pt, epoch 24)</h1>'
           f'<p>{html.escape(header)} · sorted {a.sort}-first · GT vs prediction (camera-subtracted)</p></header>'
           f'<main>{chr(10).join(cards)}</main></body></html>')
    (out / "index.html").write_text(doc)
    (out / "full_val_metrics.json").write_text(json.dumps(
        {"n": n, "ade_cm_mean": float(ade.mean()), "ade_cm_median": float(np.median(ade)),
         "fde_cm_mean": float(fde.mean()), "fde_cm_median": float(np.median(fde))}, indent=2))
    print(f"wrote {out/'index.html'} ({n} clips)")


if __name__ == "__main__":
    main()
