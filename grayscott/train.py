"""Train an emulator on the generated trajectories.

    python -m grayscott.train --variant global         # dataset-agnostic
    python -m grayscott.train --variant window
    python -m grayscott.train --variant window-aware
    python -m grayscott.train --variant fixed-abs      # Gray-Scott-specific reference
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import data
from .model import Config, Emulator

# Dataset-agnostic variants: no knowledge of the fields, their bounds or the
# PDE parameters. Everything they use is estimated from the training array or
# from the input window.
GENERAL = {
    # training-set statistics in, training-set delta units out, detected bounds
    "global": dict(scheme="global-meanstd", history=4, cond="none", clip=True, bounds="data"),
    # Walrus-style per-window statistics
    "window": dict(scheme="window-rms", history=4, cond="none", clip=False, bounds="data"),
    # same, with the window statistics also fed to the network
    "window-aware": dict(scheme="window-rms", history=4, cond="scales", clip=False, bounds="data"),
}

# Gray-Scott-specific reference variants: (1 - A, B) inputs, the exact box,
# and (F, k) as an input.
_GS = dict(scheme="fixed", history=1, cond="params", bounds="exact")
SPECIFIC = {
    "fixed-abs": dict(_GS, head="linear", clip=True, loss_space="absolute"),
    "fixed": dict(_GS, head="sinh", clip=True, loss_space="asinh"),
    "fixed-pd": dict(_GS, head="pd", clip=False, loss_space="asinh"),
    "fixed-linear": dict(_GS, head="linear", clip=True, loss_space="relative"),
    "samplewise": dict(scheme="window-rms", history=4, cond="params", clip=False, bounds="exact"),
}
VARIANTS = {**GENERAL, **SPECIFIC}


def build(cfg: Config, d, segments=None):
    """Model with its statistics, taken from `segments` (default: everything).

    Only the `fixed` scheme reads anything Gray-Scott-specific.
    """
    stats = data.field_stats(d["u"], segments=segments)
    stats["param_mean"] = d["params"].mean(0)
    stats["param_std"] = d["params"].std(0) + 1e-8
    if cfg.scheme == "fixed":
        stats["in_scale"] = data.depletion_scale(d["u"])
    return Emulator(cfg, stats)


def save_run(path, model, meta):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": model.cfg.to_dict(), "state": model.state_dict()}, path / "model.pt")
    (path / "train_log.json").write_text(json.dumps(meta, indent=1))


def load_run(path):
    path = Path(path)
    ckpt = torch.load(path / "model.pt", map_location="cpu")
    cfg = Config.from_dict(ckpt["cfg"])
    model = Emulator(cfg)
    legacy = {"cond_mean": "param_mean", "cond_std": "param_std"}
    state = {legacy.get(k, k): v for k, v in ckpt["state"].items()}
    if "delta_std" not in state:  # written before the four Walrus strategies were separate
        state["delta_std"] = state["delta_scale"].clone()
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = {"mean", "std", "rms", "delta_mean", "lo", "hi", "param_mean", "param_std"}
    if unexpected or set(missing) - allowed:
        raise RuntimeError(f"{path}: missing {missing}, unexpected {unexpected}")
    model.eval()
    return model, json.loads((path / "train_log.json").read_text())


def make_pairs(u, segments, history, unroll=1, steady_weight=0.1):
    """Training samples inside `segments` and their sampling probabilities.

    A sample is (trajectory, index of the last input frame); its whole input
    window and its targets lie inside one segment. Frames after which nothing
    changes any more are kept but drawn `steady_weight` times as often.
    """
    change = (u[:, 1:] - u[:, :-1]).square().mean(dim=(2, 3, 4)).sqrt().cpu().numpy()  # [N, T-1]
    pairs = np.array([(i, t) for i, a, b in segments for t in range(a + history - 1, b - unroll)], dtype=int).reshape(-1, 2)
    if len(pairs) == 0:
        raise ValueError(f"no training sample: every segment is shorter than history + unroll = {history + unroll} frames")
    prob = np.where(change[pairs[:, 0], pairs[:, 1]] > 1e-6, 1.0, steady_weight)
    return pairs, prob / prob.sum()


def fit(model, u, params, segments=None, steps=4000, batch=8, lr=1e-3, unroll=1, seed=0, steady_weight=0.1, log_every=200, on_log=None, on_save=None, save_every=1000):
    """Train `model` in place on `segments` of u [N, T, C, H, W]; returns the loss log."""
    cfg = model.cfg
    rng = np.random.default_rng(seed)
    if segments is None:
        segments = [(i, 0, u.shape[1]) for i in range(u.shape[0])]
    h = cfg.history
    pairs, prob = make_pairs(u, segments, h, unroll, steady_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log, t0, running = [], time.time(), []
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        i, t = pairs[rng.choice(len(pairs), batch, p=prob)].T.tolist()
        hist = torch.stack([u[a, b - h + 1 : b + 1] for a, b in zip(i, t)])
        p = params[i]
        if unroll == 1:
            loss = model.training_loss(hist, p, torch.stack([u[a, b + 1] for a, b in zip(i, t)]))
        else:
            loss = 0.0
            prev_true = hist[:, -1]
            for j in range(unroll):
                target = torch.stack([u[a, b + 1 + j] for a, b in zip(i, t)])
                pred, scale = model(hist, p)
                loss = loss + model.loss(pred, scale, hist[:, -1], prev_true, target) / unroll
                prev_true = target
                if j + 1 < unroll:
                    nxt = model.project(pred) if cfg.clip else pred
                    hist = torch.cat([hist[:, 1:], nxt[:, None]], dim=1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running.append(float(loss.detach()))
        if (step + 1) % log_every == 0 or step + 1 == steps:
            entry = {"step": step + 1, "loss": float(np.mean(running)), "minutes": (time.time() - t0) / 60}
            log.append(entry)
            running = []
            if on_log:
                on_log(entry)
        if on_save and (step + 1) % save_every == 0 and step + 1 < steps:
            on_save(log)
    model.eval()
    return log


def lr_at(step, steps, lr, warmup=100, final=0.01):
    if step < warmup:
        return lr * (step + 1) / warmup
    x = (step - warmup) / max(1, steps - warmup)
    return lr * (final + (1 - final) * 0.5 * (1 + math.cos(math.pi * x)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=list(VARIANTS), default="global")
    ap.add_argument("--data", default=str(data.DEFAULT_PATH))
    ap.add_argument("--out", default=None, help="run directory (default: runs/<variant>)")
    ap.add_argument("--init", default=None, help="run directory to start from (same variant); used for unrolled fine-tuning")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--unroll", type=int, default=1, help="number of autoregressive steps in the loss")
    ap.add_argument("--loss", choices=["l1", "l2"], default="l2")
    ap.add_argument("--floor", type=float, default=0.1, help="fixed scheme: e = floor * dataset delta RMS")
    ap.add_argument("--steady-weight", type=float, default=0.1, help="sampling weight of frames with no change")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.set_flush_denormal(True)
    torch.manual_seed(args.seed)

    d = data.load(args.data)
    dev = torch.device(args.device)
    u = torch.from_numpy(d["u"]).to(dev)  # [N, T, C, H, W]
    params_all = torch.from_numpy(d["params"]).to(dev)  # [N, P], read only when cond = params
    n_traj, n_frames, n_fields = u.shape[:3]

    cfg = Config(
        n_fields=n_fields,
        n_params=params_all.shape[1],
        width=args.width,
        blocks=args.blocks,
        loss=args.loss,
        floor=args.floor,
        **VARIANTS[args.variant],
    )
    model = build(cfg, d)
    history = []
    if args.init:
        prev, prev_meta = load_run(args.init)
        if prev_meta["variant"] != args.variant:
            raise ValueError(f"--init is a {prev_meta['variant']} run, not {args.variant}")
        model.load_state_dict(prev.state_dict())
        history = prev_meta.get("history", []) + [{"args": prev_meta["args"], "log": prev_meta["log"]}]
    model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    out = Path(args.out or f"runs/{args.variant}")
    print(f"variant={args.variant} cfg={cfg.to_dict()} params={n_params} out={out}", flush=True)
    stats = {k: getattr(model, k).flatten().tolist() for k in ("mean", "std", "delta_scale", "lo", "hi")}
    stats = {k: [x if math.isfinite(x) else None for x in v] for k, v in stats.items()}
    print(f"stats={stats}", flush=True)

    def on_log(entry):
        print(f"step {entry['step']:6d}  loss {entry['loss']:.4f}  {entry['minutes']:.1f} min", flush=True)

    def on_save(log):
        save_run(out, model, {"variant": args.variant, "args": vars(args), "params": n_params, "stats": stats, "log": log})

    log = fit(
        model, u, params_all, steps=args.steps, batch=args.batch, lr=args.lr, unroll=args.unroll, seed=args.seed,
        steady_weight=args.steady_weight, log_every=args.log_every, on_log=on_log, on_save=on_save, save_every=args.save_every,
    )  # fmt: skip

    meta = {"variant": args.variant, "args": vars(args), "params": n_params, "stats": stats, "log": log, "history": history}
    save_run(out, model, meta)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
