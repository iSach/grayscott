"""Figures, in the dense "ledger" style: monospace, square frames, grayscale
fields, and series told apart by dash pattern and marker shape before colour.
"""

from __future__ import annotations

from pathlib import Path

import sys

import matplotlib

if "ipykernel" not in sys.modules:  # keep the inline backend inside notebooks
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator

GROUND, PANEL, FRAME, INK, MUTE = "#EEEEEE", "#FFFFFF", "#111111", "#111111", "#4A4A4A"
RULE, RULE2, FAIL, TRUTH = "#D5D5D5", "#E6E6E6", "#BB5566", "#9A9A9A"
# compare slots: (colour, dash, marker, filled)
SLOTS = [
    ("#111111", "-", "s", True),
    ("#004488", (0, (7, 3)), "o", True),
    ("#BB5566", (0, (1.5, 3)), "D", True),
    ("#997700", (0, (9, 3, 2, 3)), "s", False),
]
FIELD = LinearSegmentedColormap.from_list("ledger_seq", [PANEL, INK])
B_MAX = 0.6
DPI = 170


def _style():
    have = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ("Berkeley Mono", "JetBrains Mono", "DejaVu Sans Mono") if f in have), "monospace")
    plt.rcParams.update(
        {
            "font.family": font,
            "font.size": 7.5,
            "text.color": INK,
            "axes.edgecolor": FRAME,
            "axes.linewidth": 0.8,
            "axes.facecolor": PANEL,
            "axes.labelcolor": MUTE,
            "figure.facecolor": GROUND,
            "savefig.facecolor": GROUND,
            "xtick.color": MUTE,
            "ytick.color": MUTE,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "grid.color": RULE,
            "grid.linestyle": ":",
            "grid.linewidth": 0.6,
            "hatch.linewidth": 0.6,
        }
    )


def _line(ax, x, y, slot, lw=1.4, every=20, top=None):
    """Line with periodic markers and an end marker; no end marker if it ends above `top`."""
    colour, dash, marker, filled = SLOTS[slot]
    cap = "round" if slot == 2 else "butt"
    kw = dict(color=colour, markerfacecolor=colour if filled else PANEL, markeredgecolor=colour, markeredgewidth=1.0)
    ax.plot(x, y, linestyle=dash, lw=lw, dash_capstyle=cap, marker=marker, ms=3.6, markevery=every, **kw)
    if top is None or y[-1] <= top:
        ax.plot(x[-1:], y[-1:], linestyle="none", marker=marker, ms=4.6, clip_on=False, zorder=5, **kw)


def _legend(fig, entries, y, x0=0.02, width=0.96):
    """A one-line key above the plots: marker + dash sample + UPPERCASE label."""
    ax = fig.add_axes([x0, y, width, 0.018])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    step = 1.0 / max(len(entries), 1)
    for i, (label, slot) in enumerate(entries):
        xa = i * step
        if slot is None:
            ax.plot([xa, xa + 0.05], [0.5, 0.5], color=TRUTH, lw=3.0, solid_capstyle="butt")
        else:
            colour, dash, marker, filled = SLOTS[slot]
            ax.plot([xa, xa + 0.05], [0.5, 0.5], color=colour, linestyle=dash, lw=1.4)
            ax.plot([xa + 0.025], [0.5], marker=marker, ms=4.6, color=colour, mfc=colour if filled else PANEL, mec=colour)
        ax.text(xa + 0.06, 0.5, label.upper(), va="center", ha="left", fontsize=7, fontweight="bold")


def _chip(ax, x, y, text, bad=False, **kw):
    box = dict(boxstyle="square,pad=0.25", lw=0.8, fc=FAIL if bad else PANEL, ec=FAIL if bad else MUTE)
    ax.text(x, y, text, color="#FFFFFF" if bad else MUTE, fontsize=6.3, fontweight="bold", bbox=box, **kw)


def _row_label(ax, name, F, k, outcome):
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0, 0.80, name.upper(), fontsize=8, fontweight="bold", va="center")
    ax.text(0, 0.55, f"F {F:.3f}  K {k:.3f}", fontsize=6.5, color=MUTE, va="center")
    _chip(ax, 0.03, 0.26, outcome.upper(), va="center", ha="left")


def _field(ax, img):
    ax.imshow(np.clip(img, 0, B_MAX), cmap=FIELD, vmin=0, vmax=B_MAX, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def _title(fig, title, sub):
    fig.text(0.02, 0.985, title.upper(), fontsize=11, fontweight="bold", va="top")
    fig.text(0.02, 0.965, sub, fontsize=7, color=MUTE, va="top")


def fig_dataset(d, out):
    u, names = d["u"], [str(n) for n in d["names"]]
    n, n_frames = u.shape[:2]
    dt = float(d["frame_dt"])
    frames = [0, 4, 16, 48, n_frames - 1]
    fig = plt.figure(figsize=(7.2, 1.18 * n + 1.0), dpi=DPI)
    gs = fig.add_gridspec(n, 8, width_ratios=[1.25, 1, 1, 1, 1, 1, 0.42, 2.3], left=0.02, right=0.985, top=0.905, bottom=0.05, wspace=0.07, hspace=0.12)
    _title(fig, "Gray-Scott trajectories", f"B field (white 0, black {B_MAX}) at five times; right: RMS of the frame-to-frame change, log scale")
    _legend(fig, [("rms dA", 0), ("rms dB", 1)], 0.925, x0=0.70, width=0.28)
    floor = 1e-9
    for i in range(n):
        _row_label(fig.add_subplot(gs[i, 0]), names[i], d["F"][i], d["k"][i], str(d["outcomes"][i]))
        for j, f in enumerate(frames):
            ax = fig.add_subplot(gs[i, 1 + j])
            _field(ax, u[i, f, 1])
            if i == 0:
                ax.set_title(f"T={f * dt:.0f}", fontsize=6.5, color=MUTE, pad=3)
        ax = fig.add_subplot(gs[i, 7])
        rms = np.sqrt(((u[i, 1:] - u[i, :-1]).astype(np.float64) ** 2).mean(axis=(2, 3)))  # [T-1, 2]
        x = np.arange(1, n_frames)
        zero = np.where((rms == 0).all(axis=1))[0]
        for c in (0, 1):
            keep = rms[:, c] > floor
            _line(ax, x[keep], np.maximum(rms[keep, c], floor), c, lw=1.2, every=24)
        if len(zero):
            ax.add_patch(Rectangle((x[zero[0]], floor), n_frames - x[zero[0]], 1, fill=False, hatch="////", ec=RULE, lw=0))
            ax.text(n_frames - 3, 3e-8, "CHANGE = 0", ha="right", va="center", fontsize=6, color=MUTE, fontweight="bold")
        ax.axhline(1e-5, color=MUTE, lw=0.7, ls=(0, (1, 2)))
        if i == 0:
            ax.text(n_frames - 3, 2.2e-5, "WINDOW EPS 1E-5", ha="right", va="bottom", fontsize=5.8, color=MUTE)
        ax.set_yscale("log")
        ax.set_ylim(floor, 0.5)
        ax.set_xlim(0, n_frames + 2)
        ax.set_yticks([1e-8, 1e-5, 1e-2])
        ax.grid(True, axis="y")
        ax.tick_params(labelbottom=(i == n - 1))
        if i == n - 1:
            ax.set_xlabel("FRAME", fontsize=6.5)
    path = Path(out) / "dataset.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# what the figures show, in this order; everything else is in metrics.json.
# Warm start: the dataset-agnostic variants, then the Gray-Scott-specific
# reference. Cold start: the Markov (Gray-Scott-specific) variants.
SHOWN = {
    "warm": ("global+clip", "global", "window", "window-aware", "fixed-abs+clip"),
    "cold": ("fixed-abs+clip", "fixed-pd", "fixed+clip", "fixed-linear+clip"),
}
CURVES = {
    "warm": ("global+clip", "window", "window-aware", "fixed-abs+clip"),
    "cold": ("fixed-abs+clip", "fixed-pd", "fixed+clip", "fixed-linear+clip"),
}


def fig_rollouts(d, results, finals, out, protocol):
    names = [str(n) for n in d["names"]]
    keys = [f"{protocol}/{lab}" for lab in SHOWN[protocol] if f"{protocol}/{lab}" in results]
    if not keys:
        return None
    n, m = len(names), len(keys)
    fig = plt.figure(figsize=(7.2, 1.45 * n + 1.0), dpi=DPI)
    gs = fig.add_gridspec(n, m + 2, width_ratios=[1.25] + [1] * (m + 1), left=0.02, right=0.985, top=0.9, bottom=0.045, wspace=0.07, hspace=0.46)
    start = results[keys[0]]["start_frame"]
    _title(
        fig,
        f"Final frame after rollout, {protocol} start",
        f"B at the last frame; simulator frames 0..{start - 1} given, rest predicted. Hatched strip + filled chip: wrong outcome",
    )
    for i, name in enumerate(names):
        _row_label(fig.add_subplot(gs[i, 0]), name, d["F"][i], d["k"][i], str(d["outcomes"][i]))
        ax = fig.add_subplot(gs[i, 1])
        _field(ax, d["u"][i, -1, 1])
        if i == 0:
            ax.set_title("SIMULATOR", fontsize=6.5, color=INK, fontweight="bold", pad=3)
        for j, key in enumerate(keys):
            ax = fig.add_subplot(gs[i, 2 + j])
            r = results[key]["trajectories"][name]
            _field(ax, finals[f"{key}/{name}"][1])
            if i == 0:
                ax.set_title(key.split("/", 1)[1].upper(), fontsize=6.5, color=INK, fontweight="bold", pad=3)
            bad = r["outcome"] != r["outcome_true"]
            if bad:  # texture + word + hue: hatched strip, WRONG chip
                ax.add_patch(Rectangle((0, 0), 0.09, 1, transform=ax.transAxes, fc=PANEL, ec=FAIL, hatch="xxxxx", lw=0.8, zorder=4))
                for s in ax.spines.values():
                    s.set_edgecolor(FAIL)
                    s.set_linewidth(1.6)
            label = ("WRONG: " if bad else "") + r["outcome"].upper()
            _chip(ax, 0.0, -0.13, label, bad=bad, transform=ax.transAxes, va="center", ha="left")
            ax.text(0.0, -0.33, f"RMSE {r['rmse_mean']:.3g}", transform=ax.transAxes, va="center", ha="left", fontsize=6.3, color=MUTE)
    path = Path(out) / f"rollouts_{protocol}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_curves(d, results, curves, meanB, out, protocol="warm"):
    names = [str(n) for n in d["names"]]
    keys = [f"{protocol}/{lab}" for lab in CURVES[protocol] if f"{protocol}/{lab}" in results]
    if not keys:
        return None
    n = len(names)
    start = results[keys[0]]["start_frame"]
    n_frames = d["u"].shape[1]
    x = np.arange(start, n_frames)
    fig = plt.figure(figsize=(7.2, 1.2 * n + 1.1), dpi=DPI)
    gs = fig.add_gridspec(n, 3, width_ratios=[1.0, 2.5, 2.5], left=0.02, right=0.975, top=0.89, bottom=0.05, wspace=0.3, hspace=0.22)
    _title(fig, f"Rollout error and mean B over time, {protocol} start", "Left: RMSE against the simulator (both fields, log scale; lines leave the top when they diverge). Right: mean of B")
    _legend(fig, [(k.split("/", 1)[1], s) for s, k in enumerate(keys)] + [("simulator", None)], 0.912)
    for i, name in enumerate(names):
        _row_label(fig.add_subplot(gs[i, 0]), name, d["F"][i], d["k"][i], str(d["outcomes"][i]))
        ax_e, ax_b = fig.add_subplot(gs[i, 1]), fig.add_subplot(gs[i, 2])
        truth = d["u"][i, start:, 1].mean(axis=(1, 2))
        ax_b.plot(x, truth, color=TRUTH, lw=3.0, solid_capstyle="butt")
        # y range from the simulator; a series that goes far beyond it leaves the
        # panel and is named in an OFF SCALE note instead of flattening the rest
        cap = 2.5 * max(float(truth.max()), 0.004)
        series = [np.nan_to_num(meanB[f"{key}/{name}"], nan=1e6) for key in keys]
        top = max([float(truth.max())] + [float(mb.max()) for mb in series if mb.max() <= cap and mb.min() >= -cap])
        off = [key.split("/", 1)[1] for key, mb in zip(keys, series) if mb.max() > cap or mb.min() < -cap]
        e_top = 2.0
        for s, key in enumerate(keys):
            _line(ax_e, x, np.clip(curves[f"{key}/{name}"], 1e-7, 1e3), s, top=e_top)
            _line(ax_b, x, series[s], s, top=1.12 * top)
        if off:
            ax_b.set_title("OFF SCALE: " + ", ".join(off).upper(), loc="right", fontsize=5.8, fontweight="bold", color=MUTE, pad=2)
        ax_e.set_yscale("log")
        ax_e.set_ylim(1e-7, e_top)
        ax_e.set_yticks([1e-6, 1e-4, 1e-2, 1])
        ax_b.set_ylim(-0.04 * top, 1.12 * top)
        ax_b.yaxis.set_major_locator(MaxNLocator(4))
        ax_b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2g}"))
        for ax in (ax_e, ax_b):
            ax.set_xlim(0, n_frames + 3)
            ax.grid(True, axis="y")
            ax.tick_params(labelbottom=(i == n - 1))
            if i == n - 1:
                ax.set_xlabel("FRAME", fontsize=6.5)
        if i == 0:
            ax_e.set_title("RMSE", fontsize=6.5, color=MUTE, pad=3, loc="left")
            ax_b.set_title("MEAN B", fontsize=6.5, color=MUTE, pad=3, loc="left")
    path = Path(out) / f"curves_{protocol}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def make_all(d, results, curves, finals, meanB, out):
    _style()
    paths = [fig_dataset(d, out)]
    for protocol in ("warm", "cold"):
        if any(k.startswith(protocol + "/") for k in results):
            for p in (fig_rollouts(d, results, finals, out, protocol), fig_curves(d, results, curves, meanB, out, protocol)):
                if p:
                    paths.append(p)
    return paths
