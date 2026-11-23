"""Figures for the lab, in the same ledger style as plots.py: monospace, square
frames, grayscale fields, status told by texture and a written code before hue.
"""

from __future__ import annotations

import numpy as np

from . import plots
from .plots import FAIL, GROUND, INK, MUTE, PANEL, RULE, SLOTS, TRUTH, _chip, _field, _line, plt

from matplotlib.patches import Rectangle  # noqa: E402

DPI = 120
STALE = "#DDAA33"
UNK = "#004488"
# status -> (face colour, hatch, edge colour, text colour)
STATUS_STYLE = {
    "OK": (PANEL, None, MUTE, INK),
    "DRIFT": (PANEL, "////", STALE, INK),
    "WRONG": (PANEL, "xxxx", FAIL, INK),
    "DIVERGED": (FAIL, None, FAIL, "#FFFFFF"),
    "SKIP": (PANEL, "....", UNK, INK),
}


def _finish(fig, save):
    if save:
        fig.savefig(save)
    return None


# written code chips: outlined when normal, filled when something is off
CHIP_STYLE = {"OK": (PANEL, MUTE, INK), "DRIFT": (STALE, STALE, "#000000"), "WRONG": (FAIL, FAIL, "#FFFFFF"), "DIVERGED": (FAIL, FAIL, "#FFFFFF"), "SKIP": (PANEL, UNK, INK)}


def _status_chip(ax, x, y, status, **kw):
    face, edge, text = CHIP_STYLE[status]
    ax.text(x, y, status, color=text, fontsize=6.5, fontweight="bold", bbox=dict(boxstyle="square,pad=0.3", fc=face, ec=edge, lw=1.0), **kw)


def _label(ax, title, lines):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0, 0.9, title.upper(), fontsize=8 if len(title) < 20 else 6.8, fontweight="bold", va="center")
    step = 0.2 if len(lines) <= 3 else 0.16
    for j, line in enumerate(lines):
        ax.text(0, 0.7 - step * j, line, fontsize=6.5, color=MUTE, va="center")


def show_data(data, names, frames=6, save=None):
    """One row per trajectory: B at a few frames of its active part, then mean B
    over time with the named parts marked."""
    plots._style()
    n = len(names)
    fig = plt.figure(figsize=(8.6, 1.22 * n + 0.75), dpi=DPI)
    gs = fig.add_gridspec(n, frames + 3, width_ratios=[1.45] + [1] * frames + [0.4, 2.6], left=0.015, right=0.985, top=1 - 0.5 / (1.22 * n + 0.75), bottom=0.42 / (1.22 * n + 0.75), wspace=0.08, hspace=0.3)
    fig.text(0.015, 0.995, "TRAJECTORIES", fontsize=11, fontweight="bold", va="top")
    fig.text(0.2, 0.992, f"B field (white 0, black {plots.B_MAX}); right: spatial mean of B, hatched = collapse part, dotted line = nothing changes after", fontsize=6.5, color=MUTE, va="top")
    for r, name in enumerate(names):
        i = data.index(name)
        u, parts = data._u[i], data.parts(name)
        _label(fig.add_subplot(gs[r, 0]), name, [f"F {data.F[i]:.3f}  K {data.k[i]:.3f}", f"AMP {data.amp[i]:.2f}"])
        _chip(fig.axes[-1], 0.02, 0.1, data.outcome(name).upper(), va="center", ha="left")
        a, b = parts["active"]
        picks = np.unique(np.round(np.linspace(a, b - 1, frames)).astype(int))
        for c, f in enumerate(picks):
            ax = fig.add_subplot(gs[r, 1 + c])
            _field(ax, u[f, 1])
            ax.set_title(f"FRAME {f}", fontsize=5.8, color=MUTE, pad=2)
        ax = fig.add_subplot(gs[r, frames + 2])
        mean_b = u[:, 1].mean(axis=(1, 2))
        top = max(float(mean_b.max()), 1e-3)
        if "collapse" in parts:
            ca, cb = parts["collapse"]
            ax.add_patch(Rectangle((ca, -0.05 * top), cb - 1 - ca, 1.2 * top, fill=False, hatch="////", ec=RULE, lw=0))
            ax.set_title(f"COLLAPSE {ca}-{cb}", loc="right", fontsize=5.8, color=MUTE, fontweight="bold", pad=2)
        if "steady" in parts:
            ax.axvline(parts["steady"][0], color=MUTE, lw=0.7, ls=(0, (1, 2)))
        _line(ax, np.arange(len(mean_b)), mean_b, 0, lw=1.2, every=max(1, len(mean_b) // 6))
        ax.set_xlim(0, len(mean_b) + 2)
        ax.set_ylim(-0.05 * top, 1.15 * top)
        ax.yaxis.set_major_locator(plots.MaxNLocator(3))
        ax.yaxis.set_major_formatter(plots.FuncFormatter(lambda v, _: f"{v:.2g}"))
        ax.grid(True, axis="y")
        ax.tick_params(labelbottom=(r == n - 1))
        if r == n - 1:
            ax.set_xlabel("FRAME", fontsize=6.5)
    return _finish(fig, save)


def plot_result(result, selections=None, frames=5, save=None):
    """Per rolled-out segment: simulator and model B at a few predicted frames,
    then mean B and RMSE over the segment."""
    plots._style()
    keys = [k for k in result.rollouts if selections is None or k.split("/", 1)[1] in selections]
    if not keys:
        print("nothing to plot: no segment was rolled out")
        return None
    n = len(keys)
    height = 1.95 * n + 0.8
    fig = plt.figure(figsize=(8.6, height), dpi=DPI)
    outer = fig.add_gridspec(n, 1, left=0.015, right=0.985, top=1 - 0.55 / height, bottom=0.45 / height, hspace=0.32)
    fig.text(0.015, 0.995, result.spec.label().upper(), fontsize=11, fontweight="bold", va="top")
    fig.text(0.015, 1 - 0.27 / height, f"Rollouts: ground truth for the first {result.model.cfg.history} frames of each segment, the rest predicted. Top row simulator, bottom row model, B field.", fontsize=6.5, color=MUTE, va="top")
    rows = {f"{r['split']}/{r['selection']}": r for r in result.rows}
    for r, key in enumerate(keys):
        ro, row = result.rollouts[key], rows[key]
        gs = outer[r].subgridspec(2, frames + 5, width_ratios=[2.15] + [1] * frames + [0.55, 2.0, 0.6, 2.0], wspace=0.08, hspace=0.06)
        h = result.model.cfg.history
        b_max = max(float(ro["truth"][h:, 1].max()), 0.02)  # one grey scale per segment, shared by both rows
        ax = fig.add_subplot(gs[:, 0])
        _label(ax, row["selection"].replace(":", " : "), [f"{row['split'].upper()}  FRAMES {row['frames']}", f"REL RMSE {row['rel_rmse']:.3g}", f"ONE-STEP {row['one_step_err']:.3g}", f"BLACK = B {b_max:.2g}"])
        _status_chip(ax, 0.03, -0.02, row["status"], va="center", ha="left")
        n_pred = len(ro["pred"])
        picks = np.unique(np.round(np.linspace(0, n_pred - 1, frames)).astype(int))
        for c, j in enumerate(picks):
            top, bot = fig.add_subplot(gs[0, 1 + c]), fig.add_subplot(gs[1, 1 + c])
            for ax_im, img in ((top, ro["truth"][h + j, 1]), (bot, ro["pred"][j, 1])):
                ax_im.imshow(np.clip(img, 0, b_max), cmap=plots.FIELD, vmin=0, vmax=b_max, interpolation="nearest")
                ax_im.set_xticks([])
                ax_im.set_yticks([])
            top.set_title(f"FRAME {ro['start'] + j}", fontsize=5.8, color=MUTE, pad=2)
        x = ro["start"] + np.arange(n_pred)
        ax_b, ax_e = fig.add_subplot(gs[:, frames + 2]), fig.add_subplot(gs[:, frames + 4])
        truth_b = ro["truth"][h:, 1].mean(axis=(1, 2))
        pred_b = np.nan_to_num(ro["pred"][:, 1].mean(axis=(1, 2)), nan=1e6)
        top_b = max(float(truth_b.max()), 1e-4)
        ax_b.plot(x, truth_b, color=TRUTH, lw=3.0, solid_capstyle="butt")
        _line(ax_b, x, pred_b, 0, every=max(1, n_pred // 5), top=2.5 * top_b)
        ax_b.set_ylim(-0.08 * top_b, min(max(top_b, float(pred_b.max())), 2.5 * top_b) * 1.15)
        ax_b.yaxis.set_major_locator(plots.MaxNLocator(3))
        ax_b.yaxis.set_major_formatter(plots.FuncFormatter(lambda v, _: f"{v:.2g}"))
        ax_b.set_title("MEAN B  (GREY = SIMULATOR)", fontsize=5.8, color=MUTE, pad=2, loc="left")
        ax_e.plot(x, np.clip(ro["persistence"], 1e-7, 1e3), color=TRUTH, lw=3.0, solid_capstyle="butt")
        _line(ax_e, x, np.clip(ro["rmse"], 1e-7, 1e3), 0, every=max(1, n_pred // 5), top=2.0)
        ax_e.set_yscale("log")
        ax_e.set_ylim(1e-6, 2.0)
        ax_e.set_yticks([1e-5, 1e-3, 1e-1])
        ax_e.set_title("RMSE  (GREY = PERSISTENCE)", fontsize=5.8, color=MUTE, pad=2, loc="left")
        for ax in (ax_b, ax_e):
            ax.grid(True, axis="y")
            ax.tick_params(labelsize=6)
            if r == n - 1:
                ax.set_xlabel("FRAME", fontsize=6.5)
    return _finish(fig, save)


def plot_compare(results, save=None):
    """Status of every segment for every experiment, then the loss curves."""
    plots._style()
    segments = []
    for res in results:
        for row in res.rows:
            key = (row["split"], row["selection"])
            if key not in segments:
                segments.append(key)
    segments.sort(key=lambda k: k[0] != "train")
    n, m = len(results), len(segments)
    panels = (n + 3) // 4
    height = 0.4 * n + 0.95 + 1.9
    fig = plt.figure(figsize=(8.6, height), dpi=DPI)
    top_frac = (0.4 * n + 0.95) / height
    gs = fig.add_gridspec(1, 1, left=0.015, right=0.985, top=0.995, bottom=1 - top_frac + 0.02)
    ax = fig.add_subplot(gs[0])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(n + 1.9, 0)
    name_w = 0.27
    cell_w = (1 - name_w) / max(m, 1)
    ax.text(0, 0.45, "SEGMENT STATUS", fontsize=11, fontweight="bold", va="center")
    ax.text(0.2, 0.47, "after rollout; number = rollout RMSE relative to persistence", fontsize=6.5, color=MUTE, va="center")
    for c, (split, sel) in enumerate(segments):
        name, _, part = sel.partition(":")
        xc = name_w + (c + 0.5) * cell_w
        ax.text(xc, 1.0, split.upper(), fontsize=5.8, color=MUTE, ha="center", va="center")
        ax.text(xc, 1.36, name.upper(), fontsize=6.3 if m <= 6 else 5.6, fontweight="bold", ha="center", va="center")
        ax.text(xc, 1.72, (part or "all").upper(), fontsize=5.8, color=INK, ha="center", va="center")
    for r, res in enumerate(results):
        y = r + 2.4
        ok, tot = res.count("train")
        ax.text(0, y, res.spec.label().upper()[:34], fontsize=7, fontweight="bold", va="center")
        rows = {(row["split"], row["selection"]): row for row in res.rows}
        for c, key in enumerate(segments):
            x0 = name_w + c * cell_w
            if key not in rows:
                ax.text(x0 + cell_w / 2, y, "-", fontsize=7, color=MUTE, ha="center", va="center")
                continue
            row = rows[key]
            face, hatch, edge, text = STATUS_STYLE[row["status"]]
            ax.add_patch(Rectangle((x0 + 0.004, y - 0.4), cell_w - 0.008, 0.8, fc=face, ec=edge, lw=1.0, hatch=hatch))
            value = "" if row["status"] in ("DIVERGED", "SKIP") else f"  {row['rel_rmse']:.2g}"
            ax.text(x0 + cell_w / 2, y, row["status"] + value, fontsize=6.5, fontweight="bold", color=text, ha="center", va="center",
                    bbox=None if hatch is None else dict(boxstyle="square,pad=0.15", fc=PANEL, ec="none"))  # fmt: skip
    # loss curves, at most four per panel so that dash + marker stay readable
    lg = fig.add_gridspec(1, panels, left=0.06, right=0.985, top=1 - top_frac - 0.12 / height * 1.9, bottom=0.42 / height, wspace=0.18)
    for pnl in range(panels):
        axl = fig.add_subplot(lg[pnl])
        chunk = results[4 * pnl : 4 * pnl + 4]
        for s, res in enumerate(chunk):
            if not res.log:
                continue
            steps = np.array([e["step"] for e in res.log])
            loss = np.array([e["loss"] for e in res.log])
            _line(axl, steps, np.maximum(loss, 1e-12), s, every=1)
            axl.text(0.985, 0.95 - 0.115 * s, f"{'■●◆□'[s]} {res.spec.label().upper()[:30]}", transform=axl.transAxes, fontsize=5.8, ha="right", va="top", color=SLOTS[s][0], fontweight="bold",
                     bbox=dict(boxstyle="square,pad=0.15", fc=PANEL, ec="none", alpha=0.9), zorder=6)  # fmt: skip
        axl.set_yscale("log")
        axl.grid(True, axis="y")
        axl.set_xlabel("STEP", fontsize=6.5)
        axl.set_title("TRAINING LOSS (NORMALISED UNITS, NOT COMPARABLE ACROSS NORMALISATIONS)" if pnl == 0 else "", fontsize=5.8, color=MUTE, pad=3, loc="left")
    return _finish(fig, save)
