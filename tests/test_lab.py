import numpy as np
import pytest

from grayscott import lab


@pytest.fixture(scope="module")
def data():
    d = lab.Data(size=32, n_frames=40, frame_dt=5)
    d.add("dies", F=0.050, k=0.068, amp=1.0, seed=3, n_seeds=2)
    d.add("lives", F=0.029, k=0.057, amp=1.0, seed=4, n_seeds=2)
    return d


def test_parts_and_selections(data):
    assert data.outcome("dies") == "extinct" and data.outcome("lives") == "pattern"
    parts = data.parts("dies")
    assert "collapse" in parts and parts["collapse"][0] == 0 and parts["collapse"][1] <= 40
    assert "collapse" not in data.parts("lives")
    assert data.select("lives") == (1, 0, 40)
    assert data.select("lives:5-20") == (1, 5, 20)
    assert data.select("dies:collapse") == (0, *parts["collapse"])
    with pytest.raises(KeyError):
        data.select("lives:collapse")
    with pytest.raises(KeyError):
        data.select("nobody")


def test_statistics_come_from_the_training_selection(data):
    spec = lab.Spec(train=["dies:collapse"], steps=2, width=8, blocks=1)
    res = lab.run(spec, data, verbose=False)
    a, b = data.parts("dies")["collapse"]
    sel = data.u[0, a:b]
    assert np.allclose(res.stats["mean"], sel.mean(axis=(0, 2, 3)), atol=1e-5)


@pytest.mark.parametrize("arch,kw", [("resnet", {}), ("unet", {}), ("fno", {"modes": 6})])
@pytest.mark.parametrize("norm", ["window-rms", "window-meanstd", "global-rms", "global-meanstd", "none"])
def test_every_model_and_normalisation_runs(data, arch, kw, norm):
    spec = lab.Spec(arch=arch, arch_kwargs=kw, norm=norm, width=8, blocks=2, steps=3, train=["dies:collapse"], test=["lives:0-20"])
    res = lab.run(spec, data, verbose=False)
    assert len(res.rows) == 2 and all(r["status"] in lab.STATUSES for r in res.rows)
    assert np.isfinite(res.log[-1]["loss"])


def test_ablate_reuses_the_base_for_clip_and_compare_tabulates(data):
    base = lab.Spec(width=8, blocks=1, steps=3, train=["dies:collapse"])
    results = lab.ablate(base, data, {"no clip": dict(clip=False), "one more": dict(train=["dies:collapse", "lives:0-20"])}, verbose=False)
    assert results[1].model is results[0].model and results[2].model is not results[0].model
    table = lab.compare(results, plot=False)
    assert list(table["experiment"]) == ["base", "no clip", "one more"]


def test_short_segment_is_skipped_not_crashed(data):
    spec = lab.Spec(width=8, blocks=1, steps=2, train=["dies:collapse"], test=["lives:0-3"])
    res = lab.run(spec, data, verbose=False)
    assert res.rows[-1]["status"] == "SKIP"
