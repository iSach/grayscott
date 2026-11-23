import math

import numpy as np
import torch

from grayscott import data, sim
from grayscott.model import Config, Emulator

STATS = {
    "mean": [0.76, 0.10],
    "std": [0.30, 0.14],
    "in_scale": [0.4, 0.2],
    "delta_scale": [0.02, 0.014],
    "lo": [-math.inf, 0.0],
    "hi": [1.0, math.inf],
    "param_mean": [0.03, 0.06],
    "param_std": [0.01, 0.01],
}
GS = dict(scheme="fixed", history=1, cond="params", bounds="exact")


def make(**kw):
    torch.manual_seed(0)
    return Emulator(Config(width=8, blocks=1, **kw), STATS)


def states(b=3, h=4, size=16):
    torch.manual_seed(1)
    A = torch.rand(b, h, 1, size, size)
    B = torch.rand(b, h, 1, size, size) * (1 - A)
    return torch.cat([A, B], dim=2), torch.tensor([[0.03, 0.06]]).expand(b, -1)


def test_general_variants_need_no_parameters():
    hist, _ = states()
    for kw in (dict(scheme="global"), dict(scheme="window"), dict(scheme="window", cond="scales")):
        pred, _ = make(**kw)(hist)  # no params argument at all
        assert torch.allclose(pred, hist[:, -1])  # zero-initialised head: identity at start


def test_data_bounds_are_detected_from_saturation():
    rng = np.random.default_rng(1)
    A0, B0 = sim.seeded_state(64, rng, n_seeds=4)
    u = sim.simulate(A0, B0, 0.030, 0.062, n_frames=12, substeps=40)[None]
    stats = data.field_stats(u)
    assert abs(stats["hi"][0] - 1.0) < 1e-6 and abs(stats["lo"][1]) < 1e-6  # A <= 1, B >= 0: where the data sits
    assert np.isinf(stats["lo"][0]) and np.isinf(stats["hi"][1])  # nothing saturates there


def test_projection_uses_the_model_bounds():
    u = torch.tensor([[[[-0.2, 0.5, 1.3]], [[-0.1, 0.0, 2.0]]]])
    data_box = make(scheme="global").project(u)
    assert torch.allclose(data_box, torch.tensor([[[[-0.2, 0.5, 1.0]], [[0.0, 0.0, 2.0]]]]))
    exact_box = make(head="linear", **GS).project(u)
    assert torch.allclose(exact_box, torch.tensor([[[[0.0, 0.5, 1.0]], [[0.0, 0.0, 2.0]]]]))


def test_window_input_forgets_amplitude_unless_told():
    """Per-window RMS scaling maps B and 0.6 * B to the same input; cond = scales tells them apart."""
    hist, _ = states()
    scaled = hist.clone()
    scaled[:, :, 1] *= 0.6

    def b_input(x):
        return x[:, :, 1] / x[:, :, 1].square().mean(dim=(1, 2, 3), keepdim=True).sqrt()

    assert torch.allclose(b_input(hist), b_input(scaled), atol=1e-5)
    aware = make(scheme="window", cond="scales")
    with torch.no_grad():
        aware.net.film[-1].weight.normal_()
        aware.net.out.weight.normal_()
    assert not torch.allclose(aware(hist)[0] - hist[:, -1], (aware(scaled)[0] - scaled[:, -1]) / 0.6, atol=1e-4)


def test_pd_head_stays_in_box_for_any_output():
    hist, params = states(h=1)
    model = make(head="pd", **GS)
    with torch.no_grad():
        model.net.out.weight.normal_(0, 5.0)
        model.net.out.bias.normal_(0, 5.0)
    pred, _ = model(hist, params)
    assert pred[:, 0].min() >= 0 and pred[:, 0].max() <= 1 and pred[:, 1].min() >= 0


def test_losses_are_finite_and_rollout_runs():
    cases = (
        dict(scheme="global"),
        dict(scheme="window"),
        dict(scheme="window", cond="scales"),
        dict(head="sinh", loss_space="asinh", **GS),
        dict(head="pd", loss_space="asinh", **GS),
        dict(head="linear", loss_space="relative", **GS),
    )
    for kw in cases:
        model = make(**kw)
        hist, params = states()
        target = hist[:, -1] * 0.99
        pred, w = model(hist, params)
        loss = model.loss(pred, w, hist[:, -1], hist[:, -1], target)
        loss.backward()
        assert torch.isfinite(loss)
        s, raw = model.rollout(hist, params, 3, clip=True)
        assert s.shape == (3, 3, 2, 16, 16) and s[:, :, 1].min() >= 0


def test_legacy_configs_still_load():
    old = dict(scheme="samplewise", head="linear", clip=False, history=4, cond=True, width=8, blocks=1, loss="l2", loss_space="asinh", floor=0.1, eps=1e-5)
    cfg = Config.from_dict(old)
    assert (cfg.scheme, cfg.cond, cfg.bounds, cfg.arch) == ("window-rms", "params", "exact", "resnet")
