"""Trajectory presets, dataset generation and loading.

    python -m grayscott.data            # writes data/trajectories.npz
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import sim

SIZE = 64
N_FRAMES = 128
SUBSTEPS = 40  # 40 * dt(0.5) = 20 time units between frames
DEFAULT_PATH = Path("data/trajectories.npz")


@dataclass(frozen=True)
class Preset:
    name: str
    F: float
    k: float
    seed: int
    n_seeds: int
    amp: float
    expected: str


# Six trajectories, three outcomes. `spots` and `subthreshold` share (F, k) and
# the seed layout and differ only in the amplitude of the initial bumps: the
# trivial state A=1, B=0 is linearly stable, so the outcome is decided by an
# absolute amplitude threshold.
PRESETS = [
    Preset("spots", 0.030, 0.062, seed=1, n_seeds=4, amp=1.0, expected="pattern"),
    Preset("subthreshold", 0.030, 0.062, seed=1, n_seeds=4, amp=0.6, expected="extinct"),
    Preset("maze", 0.029, 0.057, seed=4, n_seeds=3, amp=1.0, expected="pattern"),
    Preset("solitons", 0.030, 0.070, seed=5, n_seeds=5, amp=1.0, expected="pattern"),
    Preset("uniform", 0.040, 0.045, seed=2, n_seeds=1, amp=1.0, expected="uniform"),
    Preset("extinct", 0.050, 0.068, seed=3, n_seeds=4, amp=1.0, expected="extinct"),
]


def generate(path=DEFAULT_PATH, size=SIZE, n_frames=N_FRAMES, substeps=SUBSTEPS, verbose=True):
    trajs, outcomes = [], []
    for p in PRESETS:
        rng = np.random.default_rng(p.seed)
        A0, B0 = sim.seeded_state(size, rng, n_seeds=p.n_seeds, amp=p.amp)
        u = sim.simulate(A0, B0, p.F, p.k, n_frames, substeps)
        u[u < 1e-30] = 0.0  # no float32 denormals in the stored data
        outcome = sim.classify(u[-1])
        if outcome != p.expected:
            raise RuntimeError(f"{p.name}: expected {p.expected}, simulator gave {outcome}")
        trajs.append(u)
        outcomes.append(outcome)
        if verbose:
            A, B = u[:, 0], u[:, 1]
            print(
                f"{p.name:13s} F={p.F:.3f} k={p.k:.3f} amp={p.amp:.1f} -> {outcome:8s}"
                f"  A in [{A.min():.3f}, {A.max():.3f}]  B in [{B.min():.1e}, {B.max():.3f}]"
                f"  final mean(B)={B[-1].mean():.4f}"
            )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        u=np.stack(trajs),  # [N, T, 2, H, W], channels (A, B)
        F=np.array([p.F for p in PRESETS], dtype=np.float32),
        k=np.array([p.k for p in PRESETS], dtype=np.float32),
        names=np.array([p.name for p in PRESETS]),
        outcomes=np.array(outcomes),
        frame_dt=np.float32(substeps * sim.DT),
    )
    if verbose:
        print(f"wrote {path}  shape={np.stack(trajs).shape}")
    return path


def load(path=DEFAULT_PATH):
    """Dict with `u` [N, T, C, H, W] and `params` [N, P] (here P = 2: F and k)."""
    d = np.load(path)
    d = {key: d[key] for key in d.files}
    d["params"] = np.stack([d["F"], d["k"]], axis=1).astype(np.float32)
    return d


def field_stats(u, atom_fraction=0.01, segments=None):
    """Per-field statistics of any array [N, T, C, H, W]; nothing Gray-Scott-specific.

    mean, std, rms             over the selected frames
    delta_mean, delta_std      of the frame-to-frame change
    delta_scale                RMS of the frame-to-frame change
    lo, hi                     detected hard bounds. The minimum (maximum) of a
                               field is taken as a bound when at least
                               `atom_fraction` of all its values lie within 0.1%
                               of the range from it, i.e. the field visibly
                               saturates there; otherwise -inf (+inf). A bound
                               the data never comes near cannot be found this way.

    `segments` restricts the statistics to parts of trajectories: a list of
    (trajectory index, first frame, one past the last frame). Deltas are taken
    inside each segment only.
    """
    c = u.shape[2]
    if segments is None:
        segments = [(i, 0, u.shape[1]) for i in range(u.shape[0])]
    frames = [u[i, a:b] for i, a, b in segments]
    x = np.moveaxis(np.concatenate(frames), 1, 0).reshape(c, -1).astype(np.float64)
    deltas = [f[1:] - f[:-1] for f in frames if len(f) > 1]
    delta = np.moveaxis(np.concatenate(deltas), 1, 0).reshape(c, -1).astype(np.float64) if deltas else np.zeros((c, 1))
    vmin, vmax = x.min(axis=1), x.max(axis=1)
    tol = 1e-3 * (vmax - vmin)
    at_min = (x <= (vmin + tol)[:, None]).mean(axis=1)
    at_max = (x >= (vmax - tol)[:, None]).mean(axis=1)
    tiny = 1e-12
    return {
        "mean": x.mean(axis=1),
        "std": np.maximum(x.std(axis=1), tiny),
        "rms": np.maximum(np.sqrt((x**2).mean(axis=1)), tiny),
        "delta_mean": delta.mean(axis=1),
        "delta_std": np.maximum(delta.std(axis=1), tiny),
        "delta_scale": np.maximum(np.sqrt((delta**2).mean(axis=1)), tiny),
        "lo": np.where(at_min >= atom_fraction, vmin, -np.inf),
        "hi": np.where(at_max >= atom_fraction, vmax, np.inf),
        "fraction_at_min": at_min,
        "fraction_at_max": at_max,
    }


def depletion_scale(u):
    """Gray-Scott-specific: RMS of (1 - A, B), the input scale of the `fixed` scheme."""
    dep = np.stack([1.0 - u[:, :, 0], u[:, :, 1]], axis=2).astype(np.float64)
    return np.sqrt((dep**2).mean(axis=(0, 1, 3, 4)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the Gray-Scott trajectories.")
    ap.add_argument("--out", default=str(DEFAULT_PATH))
    ap.add_argument("--frames", type=int, default=N_FRAMES)
    ap.add_argument("--substeps", type=int, default=SUBSTEPS, help=f"solver steps (dt = {sim.DT}) between frames")
    args = ap.parse_args()
    generate(args.out, n_frames=args.frames, substeps=args.substeps)
