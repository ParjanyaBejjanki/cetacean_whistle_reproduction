"""Latent diffusion baseline (paper Table 5 rows "LDM", Table 9, Appendix D).

[EXPLICIT] U-Net on the 8x8x1 latent of the beta = 1e-4 VAE, predicts the noise; layers of Table 9; batch 128,
           300 epochs, lr 5e-4, linear variance schedule beta_min = 1e-4 .. beta_max = 0.02; T in {250, 500, 1000}.
[CONFLICT] Table 9 lists ConvBlock 3 as "3/2/256" but keeps the 2x2 size -> stride 1 used (sizes are authoritative);
           ConvBlock 5 is "3/1/128" but listed as 8x8x64 -> 128 filters used (closer to the 95.02 MFLOPs/step of the text).
[DECISION] ConvBlock = (time embedding added to the block input, conv, GroupNorm(8), swish) in the order Table 9's
           footnote gives; time embedding = sinusoidal(128) -> Linear 512 -> swish -> Linear 512, projected per block;
           down/upsampling layers without activation; output 1x1 conv; Adam; latents = encoder means of the unrotated
           training images, scaled by one global factor to unit standard deviation as in [17]; DDPM ancestral sampling
           with sigma_t^2 = beta_t; one training per T (the schedule is defined over T).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int = 128) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], 1)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb_dim: int, groups: int = 8):
        super().__init__()
        self.emb = nn.Linear(emb_dim, cin)
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.GroupNorm(groups, cout)

    def forward(self, x, e):
        return F.silu(self.norm(self.conv(x + self.emb(e)[:, :, None, None])))


class UNet(nn.Module):
    def __init__(self, emb_dim: int = 512):
        super().__init__()
        self.temb = nn.Sequential(nn.Linear(128, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.cb1 = ConvBlock(1, 64, emb_dim)
        self.down1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.cb2 = ConvBlock(64, 128, emb_dim)
        self.down2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)
        self.cb3 = ConvBlock(128, 256, emb_dim)
        self.up1 = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1, output_padding=1)
        self.cb4 = ConvBlock(384, 128, emb_dim)
        self.up2 = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1, output_padding=1)
        self.cb5 = ConvBlock(192, 128, emb_dim)
        self.out = nn.Conv2d(128, 1, 1)

    def forward(self, x, t):
        e = self.temb(timestep_embedding(t))
        h1 = self.cb1(x, e)                                     # 8x8x64
        h2 = self.cb2(self.down1(h1), e)                        # 4x4x128
        h3 = self.cb3(self.down2(h2), e)                        # 2x2x256
        h4 = self.cb4(torch.cat([self.up1(h3), h2], 1), e)      # 4x4x128
        h5 = self.cb5(torch.cat([self.up2(h4), h1], 1), e)      # 8x8x128
        return self.out(h5)                                     # 8x8x1


class LinearSchedule:
    def __init__(self, T: int, beta_min: float, beta_max: float, device):
        self.T = T
        self.beta = torch.linspace(beta_min, beta_max, T, dtype=torch.float64, device=device)
        self.alpha = 1.0 - self.beta
        self.abar = torch.cumprod(self.alpha, 0)


def ldm_loss(model: UNet, sched: LinearSchedule, z0: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    """Noise-prediction loss: t ~ U{1..T} (index 0..T-1), z_t = sqrt(abar) z0 + sqrt(1 - abar) eps."""
    b = z0.shape[0]
    t = torch.randint(0, sched.T, (b,), generator=g, device=z0.device)
    eps = torch.randn(z0.shape, generator=g, device=z0.device)
    ab = sched.abar[t].float().view(b, 1, 1, 1)
    zt = ab.sqrt() * z0 + (1 - ab).sqrt() * eps
    return F.mse_loss(model(zt, t), eps)


@torch.no_grad()
def ldm_sample(model: UNet, sched: LinearSchedule, n: int, g: torch.Generator, device) -> torch.Tensor:
    """DDPM ancestral sampling (sigma_t^2 = beta_t). Returns (n, 1, 8, 8) scaled latents."""
    z = torch.randn(n, 1, 8, 8, generator=g, device=device)
    for i in reversed(range(sched.T)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        eps = model(z, t)
        a, ab, b = float(sched.alpha[i]), float(sched.abar[i]), float(sched.beta[i])
        z = (z - b / math.sqrt(1 - ab) * eps) / math.sqrt(a)
        if i > 0:
            z = z + math.sqrt(b) * torch.randn(z.shape, generator=g, device=device)
    return z


def latents_to_grid(mu: np.ndarray, scale: float) -> torch.Tensor:
    """(N, 64) encoder means -> (N, 1, 8, 8), divided by the global scale (unit std as in [17])."""
    return torch.as_tensor(mu / scale, dtype=torch.float32).view(-1, 1, 8, 8)
