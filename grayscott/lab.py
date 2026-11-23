"""Small experiments: pick trajectories or parts of them, pick a model, train,
and see whether it learned them.

    from grayscott import lab

    data = lab.Data.standard()                       # the six preset trajectories
    data.add("weak", F=0.030, k=0.062, amp=0.4)       # simulate another one
    base = lab.Spec(arch="resnet", norm="global-meanstd",
                    train=["subthreshold:collapse", "extinct:collapse"])
    res = lab.run(base, data)
    res.table()                                       # one row per segment, with a status
    res.plot()

    lab.compare(lab.sweep(base, data, arch=["resnet", "unet", "fno"]))
    lab.compare(lab.ablate(base, data, {"window-rms": dict(norm="window-rms"),
                                        "+ spots": dict(train=base.train + ["spots"])}))

Selections are strings: "name" (whole trajectory), "name:part" with part in
all / active / collapse / steady, or "name:a-b" for frames a..b-1.

Status of a rolled-out segment (ground truth for the first `history` frames,
the rest predicted). The error is measured against persistence, i.e. against
holding the last given frame, so that a segment in which little happens is not
"learned" by predicting that nothing happens:

    rel_rmse = rollout RMSE / RMSE of the persistence forecast, over the segment

    OK         right outcome and rel_rmse below `tol` (default 0.1)
    DRIFT      right outcome, rel_rmse above `tol`
    WRONG      the last frame is in another class (pattern / uniform / extinct)
    DIVERGED   a prediction exceeded 10 in absolute value

A segment whose truth does not move (persistence RMSE below 1e-4) is OK when
the rollout RMSE is below 1e-4 as well.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from . import data as gsdata
from . import sim
from .model import Config, Emulator
from .train import build, fit

STATUSES = ("OK", "DRIFT", "WRONG", "DIVERGED")
MARGIN = 4  # steady frames kept after a collapse, so that staying collapsed is part of the task
STILL = 1e-4  # a segment whose persistence RMSE is below this does not move


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------


class Data:
    """Trajectories [N, T, 2, H, W] with their parameters, and named parts."""

    def __init__(self, size=64, n_frames=128, frame_dt=20.0):
        """`frame_dt` is the time between frames (the solver step is 0.5).

        20 is what the earlier experiments used: patterns form within 128
        frames, but a collapse is over in a handful of them. 5 resolves the
        collapses.
        """
        substeps = round(frame_dt / sim.DT)
        if substeps < 1 or abs(substeps * sim.DT - frame_dt) > 1e-9:
            raise ValueError(f"frame_dt must be a multiple of the solver step {sim.DT}")
        self.size, self.n_frames, self.frame_dt, self.substeps = size, n_frames, float(frame_dt), substeps
        self.names, self.F, self.k, self.amp, self._u = [], [], [], [], []

    @classmethod
    def standard(cls, size=64, n_frames=128, frame_dt=20.0, verbose=False):
        """The six presets (three patterns, one uniform, two extinct).

        The number of seeds scales with the area, so a larger grid has the
        same seed density.
        """
        d = cls(size, n_frames, frame_dt)
        for p in gsdata.PRESETS:
            d.add(p.name, p.F, p.k, amp=p.amp, seed=p.seed, n_seeds=max(1, round(p.n_seeds * (size / gsdata.SIZE) ** 2)))
            if verbose:
                print(d.describe(p.name))
        return d

    def add(self, name, F, k, amp=1.0, seed=0, n_seeds=4, radius=4.0):
        """Simulate one more trajectory and append it."""
        if name in self.names:
            raise ValueError(f"there is already a trajectory called {name!r}")
        rng = np.random.default_rng(seed)
        A0, B0 = sim.seeded_state(self.size, rng, n_seeds=n_seeds, radius=radius, amp=amp)
        u = sim.simulate(A0, B0, F, k, self.n_frames, self.substeps)
        u[u < 1e-30] = 0.0
        self.names.append(name)
        self.F.append(F)
        self.k.append(k)
        self.amp.append(amp)
        self._u.append(u)
        return self

    # arrays -----------------------------------------------------------------
    @property
    def u(self):
        return np.stack(self._u)

    @property
    def params(self):
        return np.stack([self.F, self.k], axis=1).astype(np.float32)

    def as_dict(self):
        return {"u": self.u, "params": self.params, "names": np.array(self.names), "F": np.array(self.F), "k": np.array(self.k)}

    # parts ------------------------------------------------------------------
    def index(self, name):
        if name not in self.names:
            raise KeyError(f"no trajectory called {name!r}; have {self.names}")
        return self.names.index(name)

    def outcome(self, name):
        return sim.classify(self._u[self.index(name)][-1])

    def parts(self, name):
        """Named frame ranges (first frame, one past the last) of one trajectory."""
        u = self._u[self.index(name)]
        n = len(u)
        change = np.sqrt(((u[1:] - u[:-1]).astype(np.float64) ** 2).mean(axis=(1, 2, 3)))
        moving = np.where(change > 1e-6)[0]
        settle = int(moving[-1]) + 1 if len(moving) else 0  # first frame after which nothing changes
        mean_b = u[:, 1].mean(axis=(1, 2))
        out = {"all": (0, n), "active": (0, min(n, settle + 1 + MARGIN))}
        if settle + 1 < n - 1:
            out["steady"] = (settle + 1, n)
        peak = int(mean_b.argmax())
        if mean_b[-1] < 0.05 * mean_b[peak]:  # the B field actually dies out
            out["collapse"] = (peak, min(n, settle + 1 + MARGIN))
        return out

    def select(self, selection):
        """'name', 'name:part' or 'name:a-b'  ->  (trajectory index, a, b)."""
        name, _, part = selection.partition(":")
        i = self.index(name)
        if not part:
            return i, 0, self.n_frames
        parts = self.parts(name)
        if part in parts:
            return (i, *parts[part])
        try:
            a, b = (int(x) for x in part.split("-"))
        except ValueError:
            raise KeyError(f"{name!r} has no part {part!r}; it has {list(parts)} (or use 'name:a-b')") from None
        if not 0 <= a < b <= self.n_frames:
            raise ValueError(f"frames {a}-{b} are outside 0-{self.n_frames}")
        return i, a, b

    def describe(self, name):
        parts = ", ".join(f"{k} {a}-{b}" for k, (a, b) in self.parts(name).items())
        i = self.index(name)
        return f"{name:14s} F={self.F[i]:.3f} k={self.k[i]:.3f} amp={self.amp[i]:.2f}  {self.outcome(name):8s}  {parts}"

    def table(self):
        rows = []
        for name in self.names:
            i, parts = self.index(name), self.parts(name)
            rows.append({"name": name, "F": self.F[i], "k": self.k[i], "amp": self.amp[i], "outcome": self.outcome(name), **{p: f"{a}-{b}" for p, (a, b) in parts.items()}})
        return _frame(rows).fillna("")

    def show(self, names=None, frames=6, save=None):
        from . import labplots

        return labplots.show_data(self, names or self.names, frames, save)


# ----------------------------------------------------------------------------
# experiment description
# ----------------------------------------------------------------------------


@dataclass
class Spec:
    """One experiment. `but(...)` returns a modified copy."""

    name: str = ""
    # model
    arch: str = "resnet"  # resnet | unet | fno | walrus
    norm: str = "global-meanstd"  # window-rms | window-meanstd | global-rms | global-meanstd | none | fixed
    cond: str = "none"  # none | scales | params
    clip: bool = True  # project the state fed back on the bounds
    bounds: str = "data"  # data (detected in the training selection) | exact (Gray-Scott box)
    history: int = 4
    loss: str = "l2"  # l2 | l1
    width: int = 32
    blocks: int = 4
    arch_kwargs: dict = field(default_factory=dict)  # e.g. {"size": "tiny"} for walrus, {"modes": 12} for fno
    extra: dict = field(default_factory=dict)  # any other model.Config field (head, loss_space, ...)
    # data
    train: list = field(default_factory=list)
    test: list = field(default_factory=list)
    # optimisation
    steps: int = 1500
    batch: int = 8
    lr: float = 1e-3
    unroll: int = 1
    seed: int = 0
    steady_weight: float = 0.1

    def but(self, name=None, **changes):
        out = replace(self, **changes)
        out.name = name if name is not None else ", ".join(f"{k}={_short(v)}" for k, v in changes.items()) or self.name
        return out

    def label(self):
        return self.name or f"{self.arch} / {self.norm}"

    def config(self, n_fields, n_params):
        return Config(
            arch=self.arch, arch_kwargs=dict(self.arch_kwargs), scheme=self.norm, cond=self.cond, clip=self.clip, bounds=self.bounds,
            history=self.history, loss=self.loss, width=self.width, blocks=self.blocks, n_fields=n_fields, n_params=n_params, **self.extra,
        )  # fmt: skip


def _short(v):
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(str(x) for x in v) + "]"
    return str(v)


# ----------------------------------------------------------------------------
# running
# ----------------------------------------------------------------------------


@dataclass
class Result:
    spec: Spec
    model: Emulator
    log: list
    rows: list  # one dict per evaluated segment
    rollouts: dict  # selection -> dict(truth, pred, start)
    stats: dict
    seconds: float
    tol: float

    def table(self):
        cols = ["split", "selection", "frames", "status", "rel_rmse", "rmse", "persistence_rmse", "one_step_err", "outcome", "truth"]
        return _frame([{c: r[c] for c in cols} for r in self.rows])

    def count(self, split="train"):
        rows = [r for r in self.rows if r["split"] == split]
        return sum(r["status"] == "OK" for r in rows), len(rows)

    def verdict(self):
        ok, n = self.count("train")
        text = f"{self.spec.label()}: learned {ok}/{n} training segments"
        if any(r["split"] == "test" for r in self.rows):
            ok, n = self.count("test")
            text += f", {ok}/{n} held-out segments"
        return text + f" (right outcome, rollout error below {self.tol:g} x persistence)"

    def summary(self):
        tr = [r for r in self.rows if r["split"] == "train"]
        te = [r for r in self.rows if r["split"] == "test"]
        out = {
            "experiment": self.spec.label(),
            "params": sum(p.numel() for p in self.model.parameters()),
            "steps": self.spec.steps,
            "minutes": round(self.seconds / 60, 1),
            "final_loss": self.log[-1]["loss"] if self.log else float("nan"),
            "train_ok": "%d/%d" % self.count("train"),
            "train_one_step": float(np.mean([r["one_step_err"] for r in tr])) if tr else float("nan"),
            "train_rel_rmse": _mean_rel(tr),
        }
        if te:
            out.update({"test_ok": "%d/%d" % self.count("test"), "test_rel_rmse": _mean_rel(te)})
        return out

    def reevaluate(self, data, name=None, clip=None, test=None, tol=None):
        """Same trained model, different rollout: projection on or off, other held-out segments."""
        spec = replace(self.spec, clip=self.spec.clip if clip is None else clip, test=self.spec.test if test is None else list(test))
        spec.name = name or self.spec.label() + (" (clip %s)" % ("on" if spec.clip else "off") if clip is not None else "")
        rows, rollouts = _evaluate(self.model, spec, data, self.tol if tol is None else tol)
        return Result(spec, self.model, self.log, rows, rollouts, self.stats, self.seconds, self.tol if tol is None else tol)

    def plot(self, selections=None, frames=5, save=None):
        from . import labplots

        return labplots.plot_result(self, selections, frames, save)


def _mean_rel(rows):
    rows = [r for r in rows if r["status"] != "SKIP"]
    if not rows:
        return float("nan")
    return float("inf") if any(r["status"] == "DIVERGED" for r in rows) else float(np.mean([r["rel_rmse"] for r in rows]))


def run(spec: Spec, data: Data, device="cpu", tol=0.1, verbose=True, threads=0):
    """Train `spec` on its training selection and roll it out on train and test."""
    if not spec.train:
        raise ValueError("spec.train is empty: give it selections such as 'extinct:collapse'")
    if threads > 0:
        torch.set_num_threads(threads)
    torch.set_flush_denormal(True)
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    d = data.as_dict()
    segments = [data.select(s) for s in spec.train]
    cfg = spec.config(d["u"].shape[2], d["params"].shape[1])
    model = build(cfg, d, segments).to(device)
    u = torch.from_numpy(d["u"]).to(device)
    params = torch.from_numpy(d["params"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        n_pairs = sum(max(0, b - a - cfg.history - spec.unroll + 1) for _, a, b in segments)
        print(f"{spec.label()}: {n_params:,} parameters, {n_pairs} training samples from {len(segments)} segments, {spec.steps} steps")

    def on_log(e):
        if verbose:
            print(f"  step {e['step']:6d}  loss {e['loss']:.4f}  {e['minutes']:.1f} min", flush=True)

    t0 = time.time()
    log = fit(model, u, params, segments, steps=spec.steps, batch=spec.batch, lr=spec.lr, unroll=spec.unroll, seed=spec.seed,
              steady_weight=spec.steady_weight, log_every=max(1, spec.steps // 5), on_log=on_log)  # fmt: skip
    seconds = time.time() - t0
    rows, rollouts = _evaluate(model, spec, data, tol)
    stats = {k: getattr(model, k).flatten().tolist() for k in ("mean", "std", "rms", "delta_mean", "delta_std", "delta_scale", "lo", "hi")}
    res = Result(spec, model, log, rows, rollouts, stats, seconds, tol)
    if verbose:
        print("  " + res.verdict())
    return res


@torch.no_grad()
def _evaluate(model, spec, data, tol):
    device = next(model.parameters()).device
    d = data.as_dict()
    h = model.cfg.history
    rows, rollouts = [], {}
    model.eval()
    for split, selections in (("train", spec.train), ("test", spec.test)):
        for sel in selections:
            i, a, b = data.select(sel)
            truth = d["u"][i, a:b]
            row = {"split": split, "selection": sel, "frames": f"{a}-{b}"}
            if b - a < h + 1:
                row.update(status="SKIP", rel_rmse=np.nan, rmse=np.nan, persistence_rmse=np.nan, one_step_err=np.nan, outcome="", truth=sim.classify(truth[-1]))
                rows.append(row)
                continue
            u = torch.from_numpy(truth).to(device)
            p = torch.from_numpy(d["params"][i : i + 1]).to(device)
            # teacher-forced one-step error, relative to the size of the true update
            torch.manual_seed(0)
            np.random.seed(0)
            hist = torch.stack([u[t - h + 1 : t + 1] for t in range(h - 1, len(u) - 1)])
            pred = torch.cat([model(hist[j : j + 16], p.expand(len(hist[j : j + 16]), -1))[0] for j in range(0, len(hist), 16)])
            num = (pred - u[h:]).square().sum()
            den = (u[h:] - u[h - 1 : -1]).square().sum()
            one_step = float((num / den).sqrt()) if float(den) > 0 else float("nan")
            # rollout from the first h frames
            states, raw = model.rollout(u[None, :h], p, len(u) - h, clip=spec.clip)
            states, raw = states[0].cpu().numpy(), raw[0].cpu().numpy()
            rmse_t = np.sqrt(((states.astype(np.float64) - truth[h:]) ** 2).mean(axis=(1, 2, 3)))
            persist_t = np.sqrt(((truth[h - 1 : h].astype(np.float64) - truth[h:]) ** 2).mean(axis=(1, 2, 3)))
            rmse, persist = float(np.sqrt((rmse_t**2).mean())), float(np.sqrt((persist_t**2).mean()))
            rel = rmse / persist if persist > 0 else float("inf") if rmse > 0 else 0.0
            diverged = bool(np.abs(raw).max() > 10.0)
            outcome, true_outcome = sim.classify(states[-1]), sim.classify(truth[-1])
            if diverged:
                status, outcome = "DIVERGED", "diverged"
            elif outcome != true_outcome:
                status = "WRONG"
            elif (rel < tol) if persist >= STILL else (rmse < STILL):
                status = "OK"
            else:
                status = "DRIFT"
            row.update(status=status, rel_rmse=rel, rmse=rmse, persistence_rmse=persist, one_step_err=one_step, outcome=outcome, truth=true_outcome)
            rows.append(row)
            rollouts[f"{split}/{sel}"] = {"truth": truth, "pred": states, "start": a + h, "first": a, "rmse": rmse_t, "persistence": persist_t}
    return rows, rollouts


def sweep(base: Spec, data: Data, device="cpu", tol=0.1, verbose=True, **grid):
    """One run per combination, e.g. sweep(base, data, arch=["resnet", "unet", "fno"])."""
    keys = list(grid)
    results = []
    for values in itertools.product(*(grid[k] for k in keys)):
        changes = dict(zip(keys, values))
        results.append(run(base.but(**changes), data, device, tol, verbose))
    return results


def ablate(base: Spec, data: Data, variants: dict, device="cpu", tol=0.1, verbose=True, baseline=None):
    """The base experiment, then one run per entry of `variants` (label -> changes).

    Pass an existing Result as `baseline` to avoid training the base again.
    A variant that only switches `clip` reuses the base model (with a
    single-step loss the projection does not enter training).
    """
    base_res = baseline or run(base.but(name=base.name or "base"), data, device, tol, verbose)
    results = [base_res]
    for label, changes in variants.items():
        if set(changes) == {"clip"} and base.unroll == 1:
            results.append(base_res.reevaluate(data, name=label, clip=changes["clip"]))
            if verbose:
                print(f"{label}: same model as the base, rolled out with clip={changes['clip']}\n  " + results[-1].verdict())
        else:
            results.append(run(base.but(name=label, **changes), data, device, tol, verbose))
    return results


def compare(results, plot=True, save=None):
    """One row per experiment; with `plot`, also the status of every segment."""
    table = _frame([r.summary() for r in results]).fillna("")
    if plot:
        from . import labplots

        labplots.plot_compare(results, save)
    return table


def save(result: Result, path):
    """Checkpoint + spec, loadable with torch.load."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"spec": result.spec, "cfg": result.model.cfg.to_dict(), "state": result.model.state_dict(), "log": result.log, "rows": result.rows}, path)
    return path


def _frame(rows):
    import pandas as pd

    return pd.DataFrame(rows)
