"""Emulator = normalisation scheme + network + output head + optional projection.

Normalisation schemes (`scheme`), dataset-agnostic
--------------------------------------------------
The four strategies of the Walrus code base (walrus/trainer/normalization_strat.py)
plus the identity. With input window U and its frame deltas dU:

    window-rms       U / RMS(U),            update = RMS(dU) * raw        <- released Walrus
    window-meanstd   (U - mean) / std,      update = mean(dU) + std(dU) * raw
    global-rms       U / RMS_train,         update = RMS_train(dU) * raw
    global-meanstd   (U - mean) / std_train, update = mean_train(dU) + std_train(dU) * raw
    none             U,                     update = raw

"window" statistics are taken per sample and per field over (time, space) of
the input window, floored with max(., eps), eps = 1e-5, and not differentiated
through. "global" statistics come from the training selection. The loss is the
error on the next state divided by the update scale, i.e. the error in
normalised space, as in Walrus.

Network (`arch`): resnet | unet | fno | walrus, see nets.py.

Conditioning (`cond`)
---------------------
none        the dynamics have to be read off the input window
scales      window schemes only: log10 of the two window statistics per field
            is fed to the network through FiLM, so the amplitude that the
            normalisation divided out is still available (not for walrus)
params      a vector of known PDE parameters (not dataset-agnostic)

Bounds (`bounds`, used when `clip` is on)
-----------------------------------------
data        per-field hard bounds detected in the training data: an extreme
            value counts as a bound when at least 1% of all values sit on it
            (data.field_stats)
exact       the box written down by hand for Gray-Scott, A in [0,1], B >= 0

`clip` projects the state that is fed back (rollouts, unrolled training). The
loss always sees the unprojected prediction, so there is no dead-gradient
region.

Gray-Scott-specific reference pieces
------------------------------------
fixed       scheme: inputs are (1 - A, B) divided by dataset constants, so the
            trivial state is exactly 0 in both channels.
sinh        head: u+ = u + e sinh(raw), e = floor * dataset delta RMS.
pd          head: production-destruction update, in the box by construction:
                A+ = A (1 - d_A) + (1 - A) p_A
                B+ = B (1 - d_B) + s_B
            with p, d = 1 - exp(-softplus(raw)) in [0, 1) and s_B >= 0.
loss_space  fixed scheme: "absolute" (error / dataset delta RMS), "asinh"
            (error between asinh(delta / e)), or "relative" (error / (RMS of
            the true delta + e), per sample and field).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nets import ARCHS, Backbone, Block, build_net  # noqa: F401  (Backbone, Block: old import path)

SCHEMES = ("window-rms", "window-meanstd", "global-rms", "global-meanstd", "none", "fixed")
SCHEME_ALIASES = {"window": "window-rms", "samplewise": "window-rms", "walrus": "window-rms", "global": "global-meanstd"}
HEADS = ("linear", "sinh", "pd")
CONDS = ("none", "scales", "params")


@dataclass
class Config:
    arch: str = "resnet"
    arch_kwargs: dict = field(default_factory=dict)
    scheme: str = "global-meanstd"
    head: str = "linear"
    clip: bool = False  # project the state fed back (rollout, unrolled training)
    bounds: str = "data"  # "data" | "exact"
    history: int = 4
    cond: str = "none"
    n_fields: int = 2
    n_params: int = 2
    width: int = 32
    blocks: int = 4
    loss: str = "l2"  # "l1" | "l2": norm applied to the normalised error
    causal_loss: bool = True  # a causal network (walrus) is trained on every input frame, as Walrus is
    loss_space: str = "absolute"  # fixed scheme only
    floor: float = 0.1  # fixed scheme: e = floor * dataset delta RMS
    eps: float = 1e-5  # window schemes: floor on the window statistics

    def __post_init__(self):
        self.scheme = SCHEME_ALIASES.get(self.scheme, self.scheme)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        """Also reads configs saved by earlier versions of the project."""
        d = dict(d)
        if "bounds" not in d:  # first round of runs: Gray-Scott-specific
            d["bounds"] = "exact"
            d.setdefault("loss_space", "asinh")
        if isinstance(d.get("cond"), bool):
            d["cond"] = "params" if d["cond"] else "none"
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in fields(cls)}})


class Emulator(nn.Module):
    """`stats` holds per-field arrays; every entry is optional and only the ones
    the chosen scheme needs are read:

        mean, std, rms           training-set statistics of the fields   (global schemes)
        delta_mean, delta_std    ... of the frame deltas                 (global-meanstd)
        delta_scale              RMS of the frame deltas                 (global-rms, fixed)
        lo, hi                   hard bounds per field, +-inf where none (clip)
        in_scale                 RMS of (1 - A, B)                       (fixed)
        param_mean, param_std    statistics of the PDE parameters        (cond = params)
    """

    def __init__(self, cfg: Config, stats=None):
        super().__init__()
        stats = stats or {}
        c = cfg.n_fields
        if cfg.scheme not in SCHEMES or cfg.head not in HEADS or cfg.cond not in CONDS or cfg.arch not in ARCHS:
            raise ValueError(f"unknown arch/scheme/head/cond: {cfg.arch}/{cfg.scheme}/{cfg.head}/{cfg.cond}")
        self.window = cfg.scheme.startswith("window")
        if self.window and cfg.history < 2:
            raise ValueError("window schemes need history >= 2 to define the delta statistics")
        if cfg.head != "linear" and cfg.scheme != "fixed":
            raise ValueError("the sinh and pd heads belong to the fixed (Gray-Scott) scheme")
        if cfg.cond == "scales" and not self.window:
            raise ValueError("cond = scales feeds the window statistics, so it needs a window scheme")
        if cfg.scheme == "fixed" and c != 2:
            raise ValueError("the fixed scheme is written for the two Gray-Scott fields")
        self.cfg = cfg

        ones, zeros = [1.0] * c, [0.0] * c
        if cfg.bounds == "exact":
            lo, hi = [0.0, 0.0], [1.0, math.inf]
        else:
            lo, hi = stats.get("lo", [-math.inf] * c), stats.get("hi", [math.inf] * c)
        for name, value, shape in (
            ("mean", stats.get("mean", zeros), (1, 1, c, 1, 1)),
            ("std", stats.get("std", ones), (1, 1, c, 1, 1)),
            ("rms", stats.get("rms", ones), (1, 1, c, 1, 1)),
            ("in_scale", stats.get("in_scale", ones), (1, 1, c, 1, 1)),
            ("delta_mean", stats.get("delta_mean", zeros), (1, c, 1, 1)),
            ("delta_std", stats.get("delta_std", ones), (1, c, 1, 1)),
            ("delta_scale", stats.get("delta_scale", ones), (1, c, 1, 1)),
            ("lo", lo, (1, c, 1, 1)),
            ("hi", hi, (1, c, 1, 1)),
            ("param_mean", stats.get("param_mean", [0.0] * cfg.n_params), (cfg.n_params,)),
            ("param_std", stats.get("param_std", [1.0] * cfg.n_params), (cfg.n_params,)),
        ):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32).reshape(shape).clone())

        cond_dim = {"none": 0, "scales": 2 * c, "params": cfg.n_params}[cfg.cond]
        c_out = 4 if cfg.head == "pd" else c
        self.net = build_net(cfg.arch, c, cfg.history, c_out, cond_dim, cfg.width, cfg.blocks, **cfg.arch_kwargs)
        self.causal = bool(getattr(self.net, "causal", False))
        if cfg.head == "pd":
            # start close to the identity map: softplus(-6) ~ 0.0025
            nn.init.constant_(self.net.out.bias, -6.0)

    def project(self, u):
        """Clamp every field to its hard bounds (no-op where a bound is infinite)."""
        return torch.maximum(torch.minimum(u, self.hi), self.lo)

    def _normalise(self, hist):
        """Input window -> (network input, update shift, update scale, cond)."""
        cfg = self.cfg
        b, _, c = hist.shape[:3]
        cond = None
        if self.window:
            with torch.no_grad():  # statistics are not differentiated through, as in Walrus
                d = hist[:, 1:] - hist[:, :-1]
                if cfg.scheme == "window-rms":
                    shift_in = 0.0
                    s_in = hist.square().mean(dim=(1, 3, 4), keepdim=True).sqrt().clamp_min(cfg.eps)
                    shift_out = 0.0
                    s_out = d.square().mean(dim=(1, 3, 4)).sqrt().clamp_min(cfg.eps).view(b, c, 1, 1)
                else:
                    s_in, shift_in = torch.std_mean(hist, dim=(1, 3, 4), keepdim=True)
                    s_in = s_in.clamp_min(cfg.eps)
                    if d.shape[1] * d.shape[3] * d.shape[4] > 1:
                        s_out, shift_out = torch.std_mean(d, dim=(1, 3, 4))
                    else:
                        s_out, shift_out = torch.ones(b, c, device=hist.device), torch.zeros(b, c, device=hist.device)
                    s_out, shift_out = s_out.clamp_min(cfg.eps).view(b, c, 1, 1), shift_out.view(b, c, 1, 1)
            if cfg.cond == "scales":
                logs = torch.cat([s_in.view(b, c), s_out.view(b, c)], dim=1).log10()
                cond = (logs + 3.0) / 2.0  # eps = 1e-5 -> -1, a scale of 1 -> 1.5
        elif cfg.scheme == "global-rms":
            shift_in, s_in, shift_out, s_out = 0.0, self.rms, 0.0, self.delta_scale
        elif cfg.scheme == "global-meanstd":
            shift_in, s_in, shift_out, s_out = self.mean, self.std, self.delta_mean, self.delta_std
        elif cfg.scheme == "none":
            shift_in, s_in, shift_out, s_out = 0.0, 1.0, 0.0, torch.ones_like(self.delta_scale)
        else:  # fixed: Gray-Scott-specific, the trivial state A = 1, B = 0 maps to 0
            x = torch.stack([1.0 - hist[:, :, 0], hist[:, :, 1]], dim=2) / self.in_scale
            return x, 0.0, self.delta_scale, cond
        return (hist - shift_in) / s_in, shift_out, s_out, cond

    def forward(self, hist, params=None, all_steps=False):
        """hist: [B, H, C, h, w]; params: [B, P] or None.

        Returns (pred, scale): `pred` is the unprojected next state [B, C, h, w]
        and `scale` [B or 1, C, 1, 1] the update scale that normalises the loss.
        With `all_steps` and a causal network, `pred` is [B, H, C, h, w]: the
        prediction of frame j + 1 from frames 0..j, for every j.
        """
        cfg = self.cfg
        hist = hist[:, -cfg.history :]
        u = hist[:, -1]
        x, shift, scale, cond = self._normalise(hist)
        if cfg.cond == "params":
            cond = (params - self.param_mean) / self.param_std

        raw = self.net(x, cond)
        if self.causal:
            if all_steps:
                return hist + (shift + scale * raw.transpose(0, 1)).transpose(0, 1), scale
            raw = raw[:, -1]

        if cfg.head == "linear":
            pred = u + shift + scale * raw
        elif cfg.head == "sinh":
            pred = u + cfg.floor * self.delta_scale * torch.sinh(raw.clamp(-12.0, 12.0))
        else:
            rate = F.softplus(raw)
            p_a, d_a, d_b = (-torch.expm1(-rate[:, i]) for i in (0, 1, 2))
            s_b = 4.0 * self.delta_scale[0, 1] * rate[:, 3]
            A, B = u[:, 0], u[:, 1]
            pred = torch.stack([A * (1.0 - d_a) + (1.0 - A) * p_a, B * (1.0 - d_b) + s_b], dim=1)
        return pred, scale

    def _reduce(self, e):
        return e.abs().mean() if self.cfg.loss == "l1" else e.square().mean()

    def loss(self, pred, scale, u, prev_true, target):
        """Normalised error on the next state.

        u          state the prediction was made from (the model's own state
                   when training is unrolled)
        prev_true  ground-truth state one frame before `target`
        """
        cfg = self.cfg
        if cfg.scheme != "fixed" or cfg.loss_space == "absolute":
            e = (pred - target) / scale
        elif cfg.loss_space == "asinh":
            s = cfg.floor * self.delta_scale
            e = torch.asinh((pred - u) / s) - torch.asinh((target - u) / s)
        else:  # relative
            true_rms = (target - prev_true).square().mean(dim=(2, 3), keepdim=True).sqrt()
            e = (pred - target) / (true_rms + cfg.floor * self.delta_scale)
        return self._reduce(e)

    def training_loss(self, hist, params, target):
        """One-step loss from ground-truth inputs.

        A causal network is trained on its prediction at every input frame
        except the first (no context), as the Walrus trainer does.
        """
        if self.causal and self.cfg.causal_loss and hist.shape[1] > 1:
            pred, scale = self(hist, params, all_steps=True)
            truth = torch.cat([hist[:, -self.cfg.history :][:, 1:], target[:, None]], dim=1)
            e = (pred[:, 1:] - truth[:, 1:]) / (scale[:, None] if scale.ndim == 4 else scale)
            return self._reduce(e)
        pred, scale = self(hist, params)
        return self.loss(pred, scale, hist[:, -1], hist[:, -1], target)

    @torch.no_grad()
    def rollout(self, hist, params, n_steps, clip=None):
        """Autoregressive rollout from a ground-truth history [B, H0, C, h, w].

        Returns (states, raw): `states` [B, n_steps, C, h, w] are what was fed
        back (projected if `clip`; defaults to cfg.clip), `raw` the unprojected
        predictions.
        """
        clip = self.cfg.clip if clip is None else clip
        hist = hist.clone()
        states, raws = [], []
        for _ in range(n_steps):
            pred, _ = self(hist, params)
            pred = torch.nan_to_num(pred, nan=1e6, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
            nxt = self.project(pred) if clip else pred
            raws.append(pred)
            states.append(nxt)
            hist = torch.cat([hist[:, 1:], nxt[:, None]], dim=1) if hist.shape[1] > 1 else nxt[:, None]
        return torch.stack(states, dim=1), torch.stack(raws, dim=1)
