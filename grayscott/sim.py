"""Gray-Scott reaction-diffusion simulator.

    dA/dt = D_A lap(A) - A B^2 + F (1 - A)
    dB/dt = D_B lap(B) + A B^2 - (F + k) B

Periodic square grid, dx = 1, 5-point Laplacian, explicit Euler.

The scheme is chosen because it is monotone, not because it is accurate: with
dt * (4 D_A + F) <= 1 and dt * (4 D_B + F + k) <= 1 every update is a
non-negative combination of neighbouring values plus non-negative sources, so
the invariant region of the PDE is preserved exactly at the discrete level:

    A in [0, 1],   B >= 0        (pointwise)
    mean(A + B) <= 1             (spatial mean, periodic domain)

(the lower bound on A additionally needs dt * B^2 <= 1 - dt * (4 D_A + F),
which holds for B <~ 1). Pointwise A + B <= 1 is *not* an invariant when
D_A != D_B, and is not enforced anywhere in this project.
"""

from __future__ import annotations

import numpy as np

D_A = 0.2
D_B = 0.1
DT = 0.5


def laplacian(x: np.ndarray) -> np.ndarray:
    return (
        np.roll(x, 1, -1) + np.roll(x, -1, -1) + np.roll(x, 1, -2) + np.roll(x, -1, -2) - 4.0 * x
    )


def step(A, B, F, k, dt=DT, d_a=D_A, d_b=D_B):
    """One explicit Euler step."""
    r = A * B * B
    A_new = A + dt * (d_a * laplacian(A) - r + F * (1.0 - A))
    B_new = B + dt * (d_b * laplacian(B) + r - (F + k) * B)
    return A_new, B_new


def check_stability(F, k, dt=DT, d_a=D_A, d_b=D_B):
    if dt * (4 * d_a + F) > 1 or dt * (4 * d_b + F + k) > 1:
        raise ValueError("time step too large for the monotone (invariant-preserving) regime")


def seeded_state(size, rng, n_seeds=6, radius=4.0, amp=1.0, noise=0.0):
    """A = 1, B = 0 background with Gaussian bumps of reacted material.

    `amp` scales the bump: at the centre A = 1 - 0.5 amp and B = 0.25 amp.
    The trivial state is linearly stable, so small `amp` dies out and large
    `amp` nucleates a pattern.
    """
    yy, xx = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    bump = np.zeros((size, size))
    for _ in range(n_seeds):
        cy, cx = rng.uniform(0, size, 2)
        dy = np.minimum(np.abs(yy - cy), size - np.abs(yy - cy))
        dx = np.minimum(np.abs(xx - cx), size - np.abs(xx - cx))
        bump = np.maximum(bump, np.exp(-(dx**2 + dy**2) / (2 * radius**2)))
    if noise > 0:
        bump = bump * (1.0 + noise * rng.standard_normal(bump.shape))
    bump = np.clip(amp * bump, 0.0, 1.0)
    A = 1.0 - 0.5 * bump
    B = 0.25 * bump
    return A, B


def simulate(A0, B0, F, k, n_frames, substeps, dt=DT, d_a=D_A, d_b=D_B):
    """Returns float32 array [n_frames, 2, H, W]; frame 0 is the initial state.

    Consecutive frames are `substeps * dt` time units apart.
    """
    check_stability(F, k, dt, d_a, d_b)
    A, B = A0.astype(np.float64), B0.astype(np.float64)
    out = np.empty((n_frames, 2) + A.shape, dtype=np.float32)
    out[0, 0], out[0, 1] = A, B
    for t in range(1, n_frames):
        for _ in range(substeps):
            A, B = step(A, B, F, k, dt, d_a, d_b)
        out[t, 0], out[t, 1] = A, B
    return out


def classify(frame, tol=1e-2):
    """Outcome of a (near-)final state: 'extinct', 'uniform' or 'pattern'.

    pattern if std B >= tol; otherwise extinct if mean B < tol, else uniform.
    tol is about 2.5% of a typical B peak, far below the amplitude needed to
    nucleate a spot, so a faint residue still counts as extinct.
    """
    B = frame[1]
    if B.std() >= tol:
        return "pattern"
    return "extinct" if B.mean() < tol else "uniform"
