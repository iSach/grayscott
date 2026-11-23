import numpy as np
import pytest

from grayscott import data, sim


@pytest.mark.parametrize("preset", data.PRESETS, ids=lambda p: p.name)
def test_invariant_region(preset):
    rng = np.random.default_rng(preset.seed)
    A0, B0 = sim.seeded_state(32, rng, n_seeds=preset.n_seeds, amp=preset.amp)
    u = sim.simulate(A0, B0, preset.F, preset.k, n_frames=20, substeps=40)
    A, B = u[:, 0].astype(np.float64), u[:, 1].astype(np.float64)
    assert A.min() >= 0.0 and A.max() <= 1.0
    assert B.min() >= 0.0
    assert (A + B).mean(axis=(1, 2)).max() <= 1.0 + 1e-6


def test_amplitude_threshold():
    """Same (F, k), same seed layout: only the bump amplitude decides the fate."""
    finals = {}
    for amp in (0.6, 1.0):
        rng = np.random.default_rng(1)
        A0, B0 = sim.seeded_state(64, rng, n_seeds=4, amp=amp)
        finals[amp] = sim.simulate(A0, B0, 0.030, 0.062, n_frames=40, substeps=40)[-1]
    assert sim.classify(finals[0.6]) == "extinct"
    assert sim.classify(finals[1.0]) == "pattern"


def test_time_step_guard():
    with pytest.raises(ValueError):
        sim.check_stability(0.03, 0.06, dt=1.5)
