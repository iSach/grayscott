"""Networks behind one interface.

Every network maps a normalised input window to a normalised update:

    net(x, cond=None)     x: [B, T, C, h, w]  ->  [B, C_out, h, w]

A network with `causal = True` returns one update per input frame instead,
[B, T, C_out, h, w]; the last one is the prediction for the next frame.

    resnet   periodic ResNet (the backbone of the earlier experiments)
    unet     periodic U-Net, residual blocks, strided-conv down / up
    fno      Fourier neural operator, truncated spectral convolutions
    walrus   the official Walrus architecture (walrus.models.IsotropicModel),
             any size from a few million parameters to the released 1.3B model

None of them has a spatial normalisation layer by default, except Walrus,
which keeps its own RMSGroupNorm. `unet` and `fno` take `inner_norm="group"`
to put a GroupNorm in every block, to see what that does to amplitude.
"""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(c_in, c_out, k=3, stride=1):
    return nn.Conv2d(c_in, c_out, k, stride=stride, padding=k // 2, padding_mode="circular")


def _flat(x):
    """[B, T, C, h, w] -> [B, T * C, h, w]."""
    return x.flatten(1, 2)


def _zero(module):
    nn.init.zeros_(module.weight)
    nn.init.zeros_(module.bias)
    return module


# ----------------------------------------------------------------------------
# ResNet
# ----------------------------------------------------------------------------


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.conv1 = _conv(width, width)
        self.conv2 = _conv(width, width)
        with torch.no_grad():
            self.conv2.weight.mul_(0.1)
            self.conv2.bias.zero_()

    def forward(self, x, gamma, beta):
        y = self.conv1(F.gelu(x))
        y = y * (1.0 + gamma) + beta
        return x + self.conv2(F.gelu(y))


class Backbone(nn.Module):
    """Periodic ResNet without normalisation layers.

    No GroupNorm/LayerNorm on purpose: a spatial normalisation inside the
    network would divide out the absolute amplitude again.
    """

    causal = False

    def __init__(self, c_in, c_out, width=32, blocks=4, cond_dim=0):
        super().__init__()
        self.width, self.n_blocks = width, blocks
        self.stem = _conv(c_in, width)
        self.blocks = nn.ModuleList(Block(width) for _ in range(blocks))
        self.out = _zero(nn.Conv2d(width, c_out, 1))
        self.film = None
        if cond_dim > 0:
            self.film = nn.Sequential(nn.Linear(cond_dim, 64), nn.GELU(), _zero(nn.Linear(64, 2 * width * blocks)))

    def forward(self, x, cond=None):
        x = _flat(x) if x.ndim == 5 else x
        b = x.shape[0]
        if self.film is not None:
            gb = self.film(cond).view(b, self.n_blocks, 2, self.width, 1, 1)
        else:
            gb = x.new_zeros(b, self.n_blocks, 2, self.width, 1, 1)
        h = self.stem(x)
        for i, blk in enumerate(self.blocks):
            h = blk(h, gb[:, i, 0], gb[:, i, 1])
        return self.out(F.gelu(h))


# ----------------------------------------------------------------------------
# U-Net
# ----------------------------------------------------------------------------


class UBlock(nn.Module):
    """Residual block with optional FiLM and optional GroupNorm."""

    def __init__(self, width, cond_dim=0, inner_norm="none"):
        super().__init__()
        self.conv1, self.conv2 = _conv(width, width), _conv(width, width)
        groups = math.gcd(8, width)
        self.norm1 = nn.GroupNorm(groups, width) if inner_norm == "group" else nn.Identity()
        self.norm2 = nn.GroupNorm(groups, width) if inner_norm == "group" else nn.Identity()
        self.film = _zero(nn.Linear(cond_dim, 2 * width)) if cond_dim > 0 else None
        with torch.no_grad():
            self.conv2.weight.mul_(0.1)
            self.conv2.bias.zero_()

    def forward(self, x, cond=None):
        y = self.conv1(F.gelu(self.norm1(x)))
        if self.film is not None:
            gamma, beta = self.film(cond)[:, :, None, None].chunk(2, dim=1)
            y = y * (1.0 + gamma) + beta
        return x + self.conv2(F.gelu(self.norm2(y)))


class UNet(nn.Module):
    """Periodic U-Net: `levels` downsamplings by 2, width doubling at each."""

    causal = False

    def __init__(self, c_in, c_out, width=32, levels=2, cond_dim=0, inner_norm="none"):
        super().__init__()
        self.levels = levels
        widths = [width * 2**i for i in range(levels + 1)]
        self.stem = _conv(c_in, width)
        self.enc = nn.ModuleList(UBlock(w, cond_dim, inner_norm) for w in widths[:-1])
        self.down = nn.ModuleList(_conv(a, b, 3, stride=2) for a, b in zip(widths[:-1], widths[1:]))
        self.mid = UBlock(widths[-1], cond_dim, inner_norm)
        self.up = nn.ModuleList(nn.ConvTranspose2d(b, a, 2, stride=2) for a, b in zip(widths[:-1], widths[1:]))
        self.merge = nn.ModuleList(nn.Conv2d(2 * w, w, 1) for w in widths[:-1])
        self.dec = nn.ModuleList(UBlock(w, cond_dim, inner_norm) for w in widths[:-1])
        self.out = _zero(nn.Conv2d(width, c_out, 1))

    def forward(self, x, cond=None):
        x = _flat(x) if x.ndim == 5 else x
        if x.shape[-1] % 2**self.levels or x.shape[-2] % 2**self.levels:
            raise ValueError(f"unet with {self.levels} levels needs a grid divisible by {2**self.levels}")
        h = self.stem(x)
        skips = []
        for blk, down in zip(self.enc, self.down):
            h = blk(h, cond)
            skips.append(h)
            h = down(h)
        h = self.mid(h, cond)
        for i in reversed(range(self.levels)):
            h = self.merge[i](torch.cat([self.up[i](h), skips[i]], dim=1))
            h = self.dec[i](h, cond)
        return self.out(F.gelu(h))


# ----------------------------------------------------------------------------
# FNO
# ----------------------------------------------------------------------------


class SpectralConv(nn.Module):
    """Linear map on the lowest `modes` Fourier modes of each axis."""

    def __init__(self, width, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / width
        self.w_pos = nn.Parameter(scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat))
        self.w_neg = nn.Parameter(scale * torch.randn(width, width, modes, modes, dtype=torch.cfloat))

    def forward(self, x):
        b, c, h, w = x.shape
        m = self.modes
        if 2 * m > h or m > w // 2 + 1:
            raise ValueError(f"fno modes={m} is too large for a {h}x{w} grid")
        xf = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros_like(xf)
        out[:, :, :m, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m, :m], self.w_pos)
        out[:, :, -m:, :m] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m:, :m], self.w_neg)
        return torch.fft.irfft2(out, s=(h, w), norm="ortho")


class FNO(nn.Module):
    """Fourier neural operator (Li et al. 2021 layout): lift, spectral layers, project."""

    causal = False

    def __init__(self, c_in, c_out, width=32, layers=4, modes=12, cond_dim=0, inner_norm="none"):
        super().__init__()
        self.lift = nn.Conv2d(c_in, width, 1)
        self.spectral = nn.ModuleList(SpectralConv(width, modes) for _ in range(layers))
        self.local = nn.ModuleList(nn.Conv2d(width, width, 1) for _ in range(layers))
        groups = math.gcd(8, width)
        self.norms = nn.ModuleList(nn.GroupNorm(groups, width) if inner_norm == "group" else nn.Identity() for _ in range(layers))
        self.film = nn.ModuleList(_zero(nn.Linear(cond_dim, 2 * width)) for _ in range(layers)) if cond_dim > 0 else None
        self.proj = nn.Sequential(nn.Conv2d(width, 2 * width, 1), nn.GELU(), _zero(nn.Conv2d(2 * width, c_out, 1)))

    def forward(self, x, cond=None):
        x = _flat(x) if x.ndim == 5 else x
        h = self.lift(x)
        for i, (spec, loc, norm) in enumerate(zip(self.spectral, self.local, self.norms)):
            y = spec(norm(h)) + loc(h)
            if self.film is not None:
                gamma, beta = self.film[i](cond)[:, :, None, None].chunk(2, dim=1)
                y = y * (1.0 + gamma) + beta
            h = h + F.gelu(y)
        return self.proj(h)


# ----------------------------------------------------------------------------
# Walrus
# ----------------------------------------------------------------------------

# Model section of the released checkpoint's extended_config.yaml
# (huggingface.co/polymathic-ai/walrus). The presets below only change sizes.
_WALRUS_KERNELS = {
    "base_kernel_size1d": [[4, 4]],
    "base_kernel_size2d": [[8, 4], [8, 4]],
    "base_kernel_size3d": [[8, 4], [8, 4], [8, 4]],
}
WALRUS_SIZES = {
    #            hidden  intermediate  projection  blocks  groups  heads
    "tiny": dict(hidden_dim=96, intermediate_dim=48, projection_dim=24, processor_blocks=2, groups=4, num_heads=4),
    "small": dict(hidden_dim=192, intermediate_dim=96, projection_dim=48, processor_blocks=4, groups=8, num_heads=8),
    "base": dict(hidden_dim=384, intermediate_dim=192, projection_dim=48, processor_blocks=8, groups=16, num_heads=8),
    "released": dict(hidden_dim=1408, intermediate_dim=352, projection_dim=48, processor_blocks=40, groups=16, num_heads=16),
}
# index of each field in the released model's embedding (data.field_index_map_override)
WALRUS_RELEASED_FIELDS = {"closed_boundary": 0, "open_boundary": 1, "bias_correction": 2, "A": 42, "B": 43}
WALRUS_RELEASED_N_STATES = 67
PERIODIC = 2  # the_well.data.datasets.BoundaryCondition.PERIODIC


def walrus_config(hidden_dim, intermediate_dim, projection_dim, processor_blocks, groups, num_heads, drop_path=0.05, jitter_patches=True):
    act = {"_partial_": True, "_target_": "torch.nn.SiLU"}
    return {
        "_target_": "walrus.models.IsotropicModel",
        "encoder": {
            "_partial_": True,
            "_target_": "walrus.models.encoders.vstride_encoder.SpaceBagAdaptiveDVstrideEncoder",
            "learned_pad": True,
            **_WALRUS_KERNELS,
            "groups": groups,
            "kernel_scales_seq": [[2, 2], [4, 2], [4, 4], [8, 4]],
            "variable_downsample": True,
            "variable_deterministic_ds": True,
            "activation": act,
        },
        "decoder": {
            "_partial_": True,
            "_target_": "walrus.models.decoders.vstride_decoder.AdaptiveDVstrideDecoder",
            "learned_pad": True,
            **_WALRUS_KERNELS,
            "groups": groups,
            "activation": act,
        },
        "processor": {
            "_partial_": True,
            "_target_": "walrus.models.spatiotemporal_blocks.space_time_split.SpaceTimeSplitBlock",
            "space_mixing": {"_partial_": True, "_target_": "walrus.models.spatial_blocks.full_attention.FullAttention", "num_heads": num_heads, "mlp_dim": None},
            "time_mixing": {"_partial_": True, "_target_": "walrus.models.temporal_blocks.axial_time_attention.AxialTimeAttention", "num_heads": num_heads, "bias_type": "rel"},
            "channel_mixing": {"_partial_": True, "_target_": "torch.nn.Identity"},
        },
        "norm_layer": {"_partial_": True, "_target_": "walrus.models.shared_utils.normalization.RMSGroupNorm"},
        "hidden_dim": hidden_dim,
        "projection_dim": projection_dim,
        "intermediate_dim": intermediate_dim,
        "processor_blocks": processor_blocks,
        "drop_path": drop_path,
        "groups": groups,
        "max_d": 3,
        "static_axes": True,
        "weight_tied_axes": False,
        "causal_in_time": True,
        "include_d": [2, 3],
        "override_dimensionality": 0,
        "jitter_patches": jitter_patches,
        "gradient_checkpointing_freq": 0,
        "use_periodic_fixed_jitter": True,
        "input_field_drop": 0.0,
    }


class WalrusNet(nn.Module):
    """The official Walrus model behind the common interface.

    size        a key of WALRUS_SIZES, or a dict of the same six numbers
    weights     path to the released `walrus.pt`; needs size="released" and
                two fields, which are then embedded as Gray-Scott's A and B.
                This loading path follows the official demo notebook and has
                not been run in this project (no GPU, no access to the weights).
    jitter      Walrus's patch jittering. It stays on at inference, as in the
                paper, so rollouts are stochastic unless it is turned off.

    Walrus tokenises to 32 tokens per axis in 2D and only knows patch sizes
    4, 8, 12, 16, 24 and 32, so the grid has to be 128, 256, 384, 512, 768 or
    1024 on each side. The Well's Gray-Scott data is 128 x 128.
    """

    causal = True

    def __init__(self, n_fields, size="tiny", weights=None, jitter=True, drop_path=0.05, field_names=None):
        super().__init__()
        try:
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            from the_well.data.datasets import WellMetadata
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "arch='walrus' needs the official code: pip install the_well einops hydra-core timm, then "
                "pip install --no-deps git+https://github.com/PolymathicAI/walrus.git"
            ) from err
        dims = dict(WALRUS_SIZES[size]) if isinstance(size, str) else dict(size)
        names = list(field_names or [f"field{i}" for i in range(n_fields)])
        if weights is not None:
            if size != "released" or n_fields != 2:
                raise ValueError("released weights need size='released' and the two Gray-Scott fields")
            n_states, labels = WALRUS_RELEASED_N_STATES, [WALRUS_RELEASED_FIELDS["A"], WALRUS_RELEASED_FIELDS["B"]]
        else:
            n_states, labels = 3 + n_fields, list(range(3, 3 + n_fields))  # 0..2 are the boundary / bias slots
        with warnings.catch_warnings():  # the Walrus code calls a deprecated torch initialiser
            warnings.simplefilter("ignore", FutureWarning)
            self.model = instantiate(OmegaConf.create(walrus_config(**dims, drop_path=drop_path, jitter_patches=jitter)), n_states=n_states)
        if weights is not None:
            self.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["app"]["model"])
        self.register_buffer("labels", torch.tensor(labels), persistent=False)
        self._meta = lambda h, w: WellMetadata(
            dataset_name="gray_scott_reaction_diffusion" if weights is not None else "custom",
            n_spatial_dims=2,
            field_names={0: names, 1: [], 2: []},
            spatial_resolution=(h, w),
            scalar_names=[],
            constant_field_names={0: [], 1: [], 2: []},
            constant_scalar_names=[],
            boundary_condition_types=[],
            n_files=[],
            n_trajectories_per_file=[],
            n_steps_per_trajectory=[],
        )

    def forward(self, x, cond=None):
        b, t, c, h, w = x.shape
        if h % 32 or w % 32 or h // 32 not in (4, 8, 12, 16, 24, 32) or w // 32 not in (4, 8, 12, 16, 24, 32):
            raise ValueError(f"Walrus cannot tokenise a {h}x{w} grid: each side must be 128, 256, 384, 512, 768 or 1024")
        bcs = [[[PERIODIC, PERIODIC]] * 3] * b
        out = self.model(x.transpose(0, 1), self.labels, bcs, metadata=self._meta(h, w), train=self.training)
        return out.transpose(0, 1)  # [B, T, C, h, w]


ARCHS = ("resnet", "unet", "fno", "walrus")


def build_net(arch, n_fields, history, c_out, cond_dim=0, width=32, blocks=4, **kw):
    """`width` and `blocks` size the conv nets (blocks = ResNet blocks, FNO
    layers, U-Net levels); everything else goes through `kw`."""
    c_in = n_fields * history
    if arch == "resnet":
        return Backbone(c_in, c_out, width, blocks, cond_dim)
    if arch == "unet":
        return UNet(c_in, c_out, width, levels=kw.pop("levels", min(blocks, 3)), cond_dim=cond_dim, **kw)
    if arch == "fno":
        return FNO(c_in, c_out, width, layers=blocks, cond_dim=cond_dim, **kw)
    if arch == "walrus":
        if cond_dim > 0:
            raise ValueError("the Walrus adapter has no conditioning input (cond must be 'none')")
        if c_out != n_fields:
            raise ValueError("Walrus predicts one channel per field")
        return WalrusNet(n_fields, **kw)
    raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")
