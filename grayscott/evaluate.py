"""Roll trained emulators out on the training trajectories and score them.

    python -m grayscott.evaluate            # every runs/<variant> that exists

Two protocols:
  warm   ground-truth frames 0..W-1 are given (W = 4), frames W.. are predicted.
         Every model can run this; it is the common comparison.
  cold   only frame 0 is given. Markov (history = 1) models only, i.e. the
         Gray-Scott-specific reference variants.

Models with a linear or sinh head are rolled out both with and without the
projection on their bounds. The scores themselves use the exact Gray-Scott box
A in [0,1], B >= 0, whatever the model was told. A rollout whose unprojected
prediction exceeds 10 in absolute value anywhere is scored as "diverged".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import data, sim
from .train import GENERAL, VARIANTS, load_run

WARM = 4
BOX_TOL = 1e-3  # excursions smaller than this are not counted as leaving the box


def one_step_error(model, u, params):
    """Teacher-forced error relative to the size of the true update.

    sqrt( sum |pred - target|^2 / sum |target - prev|^2 ) over every frame the
    model can predict, per trajectory.
    """
    h = model.cfg.history
    out = []
    with torch.no_grad():
        for i in range(u.shape[0]):
            hist = torch.stack([u[i, t - h + 1 : t + 1] for t in range(h - 1, u.shape[1] - 1)])
            target = u[i, h:]
            pred, _ = model(hist, params[i : i + 1].expand(len(hist), -1))
            num = (pred - target).square().sum()
            den = (target - u[i, h - 1 : -1]).square().sum()
            out.append(float((num / den).sqrt()))
    return out


def _finite(t):
    return [x if np.isfinite(x) else None for x in t.flatten().tolist()]


def score(states, raw, truth):
    """states/raw/truth: [T, 2, H, W] numpy. Returns dict of metrics."""
    err = states.astype(np.float64) - truth
    rmse_t = np.sqrt((err**2).mean(axis=(1, 2, 3)))
    A, B = raw[:, 0], raw[:, 1]
    outside = (A < -BOX_TOL) | (A > 1 + BOX_TOL) | (B < -BOX_TOL)
    violation = np.maximum(np.maximum(-A, A - 1.0), np.maximum(-B, 0.0))
    blew_up = bool(np.abs(raw).max() > 10.0)
    return {
        "rmse_mean": float(rmse_t.mean()),
        "rmse_final": float(rmse_t[-1]),
        "outcome": "diverged" if blew_up else sim.classify(states[-1]),
        "outcome_true": sim.classify(truth[-1]),
        "meanB_final": float(states[-1, 1].mean()),
        "meanB_final_true": float(truth[-1, 1].mean()),
        "raw_min_A": float(A.min()),
        "raw_max_A": float(A.max()),
        "raw_min_B": float(B.min()),
        "frac_outside_box": float(outside.mean()),
        "max_violation": float(violation.max()),
        "blew_up": blew_up,
    }, rmse_t


def evaluate(run_dirs, data_path=data.DEFAULT_PATH, device="cpu"):
    d = data.load(data_path)
    u = torch.from_numpy(d["u"]).to(device)
    params = torch.from_numpy(d["params"]).to(device)
    names = [str(n) for n in d["names"]]
    n_frames = u.shape[1]

    results, curves, finals, meanB = {}, {}, {}, {}
    for run in run_dirs:
        model, meta = load_run(run)
        model.to(device)
        variant = meta["variant"]
        one_step = one_step_error(model, u, params)
        modes = [("warm", model.cfg.clip)]
        if model.cfg.head != "pd":
            modes.append(("warm", not model.cfg.clip))
        if model.cfg.history == 1:
            modes.append(("cold", model.cfg.clip))
        for protocol, clip in modes:
            start = WARM if protocol == "warm" else 1
            label = variant + ("+clip" if clip and model.cfg.head != "pd" else "")
            key = f"{protocol}/{label}"
            states, raw = model.rollout(u[:, :start], params, n_frames - start, clip=clip)
            states, raw = states.cpu().numpy(), raw.cpu().numpy()
            per_traj = {}
            for i, name in enumerate(names):
                m, rmse_t = score(states[i], raw[i], d["u"][i, start:])
                m["one_step_rel_err"] = one_step[i]
                per_traj[name] = m
                curves[f"{key}/{name}"] = rmse_t
                finals[f"{key}/{name}"] = states[i, -1]
                meanB[f"{key}/{name}"] = states[i, :, 1].mean(axis=(1, 2))
            results[key] = {
                "variant": variant,
                "protocol": protocol,
                "clip": bool(clip),
                "general": variant in GENERAL,
                "bounds": {"lo": _finite(model.lo), "hi": _finite(model.hi)} if clip else None,
                "start_frame": start,
                "params": meta["params"],
                "final_train_loss": meta["log"][-1]["loss"],
                "outcomes_correct": int(sum(m["outcome"] == m["outcome_true"] for m in per_traj.values())),
            "one_step_rel_err": dict(zip(names, one_step)),
                "rmse_mean_avg": float(np.mean([m["rmse_mean"] for m in per_traj.values()])),
                "trajectories": per_traj,
            }
    return d, results, curves, finals, meanB


def print_table(results, names):
    for protocol in ("warm", "cold"):
        keys = [k for k in results if k.startswith(protocol + "/")]
        if not keys:
            continue
        print(f"\n== {protocol} start: mean rollout RMSE / predicted outcome (ok = matches simulator) ==")
        print(f"{'':20s}" + "".join(f"{n:>24s}" for n in names))
        for k in keys:
            row = f"{k.split('/', 1)[1]:20s}"
            for n in names:
                m = results[k]["trajectories"][n]
                ok = "ok" if m["outcome"] == m["outcome_true"] else "WRONG"
                row += f"{m['rmse_mean']:9.3g} {m['outcome'][:8]:>8s} {ok:>5s}"
            print(row)
        print(f"\n   raw predictions: fraction of pixel-steps more than {BOX_TOL:g} outside the box / largest excursion")
        for k in keys:
            row = f"{k.split('/', 1)[1]:20s}"
            for n in names:
                m = results[k]["trajectories"][n]
                row += f"{m['frac_outside_box']:11.4f} {m['max_violation']:10.1e}"
            print(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=None, help="run directories (default: every runs/<variant> that exists)")
    ap.add_argument("--data", default=str(data.DEFAULT_PATH))
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)

    runs = args.runs or [f"runs/{v}" for v in VARIANTS if (Path("runs") / v / "model.pt").exists()]
    d, results, curves, finals, meanB = evaluate(runs, args.data, args.device)
    names = [str(n) for n in d["names"]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    np.savez_compressed(out / "curves.npz", **{k.replace("/", "|"): v for k, v in curves.items()})
    print_table(results, names)
    print(f"\nwrote {out / 'metrics.json'}")
    if not args.no_plots:
        from . import plots

        for path in plots.make_all(d, results, curves, finals, meanB, out):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
