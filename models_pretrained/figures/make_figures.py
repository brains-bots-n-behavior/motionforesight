"""Generate schematic figures for the future-scene-flow modification of TrackCraft3r.

Mirrors the notation in fig0_trackcraft3r_original.jpg:
  E^rgb / E^pm  : RGB / pointmap VAE encoders
  z^rgb / z^pm  : RGB / pointmap latents
  Track Latents : frame-0 query stream (repeated)
  Geometry Lat. : per-frame stream (the "diagonal")
  Video DiT     : Wan2.1 DiT
  D^track/D^vis : residual-track / visibility decoders

Outputs:
  fig1_future_modification.png
  fig2_training_mechanism.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# palette
FROZEN = "#cfe0f5"; FROZEN_E = "#5b8fc9"      # frozen pretrained (blue)
PM = "#9bbce8"
TRAIN = "#bfe3c0"; TRAIN_E = "#2f9e44"        # trained / new (green)
DIT = "#dfe7d3"; DIT_E = "#8aa86a"
DEC = "#efe3c8"; DEC_E = "#c9a84a"
GREY = "#e9ecef"; GREY_E = "#adb5bd"
TXT = "#1d2b3a"


def box(ax, x, y, w, h, fc, ec, txt="", fs=9, lw=1.4, style="round,pad=0.02,rounding_size=0.04",
        weight="normal", dashed=False, tc=TXT):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw,
                       ls="--" if dashed else "-")
    ax.add_patch(p)
    if txt:
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                color=tc, weight=weight, zorder=5)


def arrow(ax, x1, y1, x2, y2, color="#444", lw=1.6, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 lw=lw, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=4))


def framestack(ax, x, y, w, h, n=3, fc="#c9c9c9", ec="#888"):
    for i in range(n):
        ax.add_patch(Rectangle((x + i * 0.06, y - i * 0.06), w, h, fc=fc, ec=ec, lw=1.0, zorder=2 + i))


# ----------------------------------------------------------------------------- FIG 1
def fig1():
    fig, ax = plt.subplots(figsize=(16, 6.7))
    ax.set_xlim(0, 16); ax.set_ylim(0, 6.7); ax.axis("off")
    ax.text(8, 6.5, "Future 3D scene-flow — our modification of TrackCraft3r",
            ha="center", fontsize=14, weight="bold", color=TXT)

    yT, yG = 4.35, 2.05   # Track-latent row, Geometry-latent row
    hbox = 0.62; wbox = 0.7

    # --- inputs ---
    framestack(ax, 0.35, 4.25, 1.05, 0.85, fc="#bcae9a")
    ax.text(0.95, 5.35, "Observed video\n$\\{I_j\\}_{j=0}^{9}$", ha="center", fontsize=9, color=TXT)
    framestack(ax, 0.35, 1.95, 1.05, 0.85, fc="#7fb6a8")
    ax.text(0.95, 1.55, "Observed pointmap\n$\\{P_j(t_j)\\}_{j=0}^{9}$", ha="center", fontsize=9, color=TXT)
    ax.text(0.95, 0.95, "future frames 10–31:\nNOT observed", ha="center", fontsize=8.5,
            color=TRAIN_E, weight="bold")

    # --- encoders (frozen) ---
    box(ax, 1.75, 4.2, 0.62, 0.95, GREY, FROZEN_E, "$\\mathcal{E}^{rgb}$", 11)
    box(ax, 1.75, 1.9, 0.62, 0.95, GREY, FROZEN_E, "$\\mathcal{E}^{pm}$", 11)
    arrow(ax, 1.4, 4.65, 1.75, 4.65, FROZEN_E); arrow(ax, 1.4, 2.35, 1.75, 2.35, FROZEN_E)

    # --- latent grid ---
    x0 = 2.95; dx = 0.92
    # column labels: observed 0,1,..,9  |  future 10,..,31
    cols = [("0", False), ("1", False), ("9", False), ("10", True), ("31", True)]
    ell_after = {1, 3}   # draw ellipsis after these indices
    ax.text(x0 - 0.35, yT + hbox + 0.95, "Track Latents (frame-0 query, repeated)",
            ha="left", fontsize=9.5, color=FROZEN_E, weight="bold")
    ax.text(x0 - 0.35, yG - 0.55, "Geometry Latents (per-frame)  —  future = learned MASK latents",
            ha="left", fontsize=9.5, color=TRAIN_E, weight="bold")

    for i, (lab, fut) in enumerate(cols):
        cx = x0 + i * dx
        # Track latents (query): always frame-0 z (frozen) — both rgb & pm rows
        box(ax, cx, yT + hbox + 0.08, wbox, hbox, FROZEN, FROZEN_E, "$z_0^{rgb}$", 9, dashed=True)
        box(ax, cx, yT, wbox, hbox, PM, FROZEN_E, "$z_0^{pm}$", 9)
        # Geometry latents (diagonal): observed=real z_j ; future=mask
        if not fut:
            box(ax, cx, yG + hbox + 0.08, wbox, hbox, FROZEN, FROZEN_E, f"$z_{{{lab}}}^{{rgb}}$", 9, dashed=True)
            box(ax, cx, yG, wbox, hbox, PM, FROZEN_E, f"$z_{{{lab}}}^{{pm}}$", 9)
        else:
            box(ax, cx, yG + hbox + 0.08, wbox, hbox, TRAIN, TRAIN_E, "$z_{*}^{rgb}$", 9, weight="bold", dashed=True)
            box(ax, cx, yG, wbox, hbox, TRAIN, TRAIN_E, "$z_{*}^{pm}$", 9, weight="bold")
        # RoPE timestep
        box(ax, cx, yG + hbox + 0.7, wbox, 0.34, "#fdf2d0", DEC_E, f"$t_{{{lab}}}$", 8.5)
        if i in ell_after:
            ax.text(cx + wbox + (dx - wbox) / 2, yT + hbox * 1.5, "$\\cdots$", ha="center", fontsize=13)
            ax.text(cx + wbox + (dx - wbox) / 2, yG + hbox * 0.5, "$\\cdots$", ha="center", fontsize=13)
            ax.text(cx + wbox + (dx - wbox) / 2, yG + hbox + 0.87, "$\\cdots$", ha="center", fontsize=11, color=DEC_E)

    ax.text(x0 + 4 * dx + wbox + 0.15, yG + hbox + 0.87, "Temporal\nRoPE", ha="left", va="center",
            fontsize=8.5, color=DEC_E)
    # bracket: future block
    fx = x0 + 3 * dx - 0.1
    ax.add_patch(Rectangle((fx, yG - 0.12), 2 * dx - 0.12, hbox * 2 + 0.2, fill=False,
                           ec=TRAIN_E, lw=2.0, ls="--", zorder=6))
    ax.text(fx + dx - 0.05, yG - 0.34, "future = mask latents (the only structural change)",
            ha="center", fontsize=8.5, color=TRAIN_E, weight="bold")

    # --- DiT ---
    dxd = x0 + 5 * dx + 0.15
    box(ax, dxd, 1.9, 1.25, 3.4, DIT, DIT_E, "", 11)
    ax.text(dxd + 0.62, 4.7, "Video DiT", ha="center", fontsize=11, weight="bold", color="#3f5a2a")
    ax.text(dxd + 0.62, 4.2, "(frozen)", ha="center", fontsize=9, color="#3f5a2a")
    box(ax, dxd + 0.05, 2.05, 1.15, 0.95, TRAIN, TRAIN_E, "fresh\nLoRA\n(trained)", 8.5, weight="bold")
    for yy in (yT + 0.3, yG + 0.3):
        arrow(ax, x0 + 4 * dx + wbox + 0.02, yy, dxd, yy, "#666")

    # --- output decoder + tracks ---
    box(ax, dxd + 1.55, 3.3, 0.95, 1.0, DEC, DEC_E, "$\\hat r^{\\Delta}_{0..31}$", 9.5)
    box(ax, dxd + 2.7, 3.35, 0.6, 0.9, GREY, DEC_E, "$\\mathcal{D}^{track}$", 9.5)
    arrow(ax, dxd + 1.25, 3.8, dxd + 1.55, 3.8, "#666")
    arrow(ax, dxd + 2.5, 3.8, dxd + 2.7, 3.8, "#666")
    framestack(ax, dxd + 3.55, 3.3, 1.15, 0.95, fc="#2a2a2a", ec="#555")
    ax.text(dxd + 4.1, 4.45, "Future residual\ntracks $\\hat\\Delta_{10..31}$", ha="center", fontsize=8.5)
    arrow(ax, dxd + 3.3, 3.8, dxd + 3.55, 3.8, "#666")
    framestack(ax, dxd + 1.9, 1.0, 1.7, 1.3, fc="#6c8a3a", ec="#456")
    ax.text(dxd + 2.75, 0.65, "Predicted future\nTracking Pointmap $\\hat P_0(t_j)$, $j{=}10..31$",
            ha="center", fontsize=9, weight="bold", color="#2f5d0e")
    arrow(ax, dxd + 4.1, 3.25, dxd + 2.9, 2.35, "#456", rad=-0.2)
    ax.text(dxd + 3.9, 2.75, "$+\\,P_0(t_0)$", fontsize=8.5, color="#456")

    # legend
    lx, ly = 0.4, 0.2
    box(ax, lx, ly, 0.34, 0.26, FROZEN, FROZEN_E)
    ax.text(lx + 0.45, ly + 0.13, "frozen pretrained TrackCraft3r (DiT, rank-1024 LoRA, VAEs, T5)",
            va="center", fontsize=8.5)
    box(ax, lx + 6.6, ly, 0.34, 0.26, TRAIN, TRAIN_E)
    ax.text(lx + 7.05, ly + 0.13, "trained (12.2M): mask latents + fresh rank-32 LoRA + patch-embed + head",
            va="center", fontsize=8.5)

    fig.savefig("fig1_future_modification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- FIG 2
def fig2():
    fig, ax = plt.subplots(figsize=(14, 5.6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5.6); ax.axis("off")
    ax.text(7, 5.35, "Training mechanism — two loss spaces", ha="center", fontsize=14, weight="bold", color=TXT)

    # shared front-end
    box(ax, 0.4, 2.55, 1.7, 1.0, DIT, DIT_E, "frozen DiT\n+ fresh LoRA", 9.5, weight="bold")
    ax.text(1.25, 3.75, "observed RGB+Pj (0–9)\n+ mask latents (10–31)", ha="center", fontsize=8)
    box(ax, 2.5, 2.7, 1.15, 0.72, DEC, DEC_E, "$\\hat r^{\\Delta}$\n(pred latent)", 8.5)
    arrow(ax, 2.1, 3.05, 2.5, 3.05, "#666")

    # ---- decoded (faithful) path: top ----
    ax.text(4.0, 5.0, "A.  decoded loss  (TrackCraft3r-faithful)", fontsize=11, weight="bold", color=TRAIN_E)
    box(ax, 4.05, 4.05, 0.7, 0.8, GREY, FROZEN_E, "$\\mathcal{D}^{track}$\n(frozen)", 8)
    box(ax, 5.1, 4.05, 0.95, 0.8, "#2a2a2a", "#555", "$\\hat\\Delta$", 10, tc="white")
    box(ax, 6.4, 4.05, 1.05, 0.8, "#6c8a3a", "#456", "$\\hat P_0(t_j)$", 9, tc="white")
    box(ax, 7.9, 4.05, 1.7, 0.8, "#f6d4d4", "#c0504d",
        "MSE $\\times 10$\n(valid-masked)", 8.5, weight="bold")
    box(ax, 10.0, 4.05, 1.2, 0.8, FROZEN, FROZEN_E, "GT track\n$P_0(t_j)$", 8.5)
    arrow(ax, 3.65, 3.35, 4.4, 4.05, "#666", rad=0.2)
    arrow(ax, 4.75, 4.45, 5.1, 4.45, "#666")
    arrow(ax, 6.05, 4.45, 6.4, 4.45, "#666"); ax.text(6.22, 4.0, "$+P_0(t_0)$", fontsize=7.5, ha="center")
    arrow(ax, 7.45, 4.45, 7.9, 4.45, "#666")
    arrow(ax, 10.0, 4.45, 9.6, 4.45, "#666")
    # grad-through-frozen-VAE note
    arrow(ax, 7.9, 4.1, 4.75, 4.0, "#c0504d", lw=1.4, ls="--", rad=0.35)
    ax.text(6.3, 3.62, "gradients flow back through the frozen VAE decoder (weights not updated)",
            ha="center", fontsize=8, color="#c0504d", style="italic")

    # ---- latent (proxy) path: bottom ----
    ax.text(4.0, 1.95, "B.  latent loss  (cheap proxy — no decode)", fontsize=11, weight="bold", color=FROZEN_E)
    box(ax, 6.4, 1.0, 1.7, 0.8, "#f6d4d4", "#c0504d", "MSE\n(latent space)", 8.5, weight="bold")
    box(ax, 4.05, 1.0, 0.7, 0.8, GREY, FROZEN_E, "$\\mathcal{E}^{pm}$\n(frozen)", 8)
    box(ax, 5.0, 1.0, 1.2, 0.8, FROZEN, FROZEN_E, "enc(GT\n$\\Delta$)", 8.5)
    arrow(ax, 3.05, 2.7, 7.25, 1.8, "#666", rad=-0.15); ax.text(4.7, 2.45, "$\\hat r^{\\Delta}$", fontsize=9)
    arrow(ax, 4.75, 1.4, 5.0, 1.4, "#666")
    arrow(ax, 6.2, 1.4, 6.4, 1.4, "#666")
    box(ax, 1.9, 1.0, 1.5, 0.8, FROZEN, FROZEN_E, "GT $\\Delta$\n(normalized)", 8)
    arrow(ax, 3.4, 1.4, 4.05, 1.4, "#666")

    ax.text(11.45, 4.45, "→ loss", fontsize=10, weight="bold", color="#c0504d")
    ax.text(8.15, 1.4, "→ loss", fontsize=10, weight="bold", color="#c0504d")
    ax.text(7, 0.35, "Default used for the best run: A (decoded). B is the cheaper proxy; "
                     "both keep the VAE/DiT frozen.", ha="center", fontsize=9, color=TXT)

    fig.savefig("fig2_training_mechanism.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2()
    print("wrote fig1_future_modification.png, fig2_training_mechanism.png")
