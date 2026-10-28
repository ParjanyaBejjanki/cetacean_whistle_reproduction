"""VAE (Table 7) and critic (Table 8) of the paper, plus the Table 6 parameter / FLOP counting convention.

[EXPLICIT] layer types, kernels, strides, filter counts, activations, output sizes.
[IMPLIED]  'same' padding (output sizes), no activation after the encoder FC / decoder projection.
[DECISION] FC split (mu, log sigma^2); Glorot-uniform init with zero bias; critic LayerNorm = GroupNorm(1, C)
           (normalise C x H x W per image, per-channel scale/offset); output layer = 3x3 conv (== stride-1 transposed conv).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IMG = 128


def glorot_init(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GroupNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class Encoder(nn.Module):
    """128x128x1 -> 4 x [conv 3x3 s2 + swish] (32, 32, 64, 64) -> 8x8x64 -> FC 4096 -> 2L."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.convs = nn.ModuleList([nn.Conv2d(ci, co, 3, stride=2, padding=1)
                                    for ci, co in [(1, 32), (32, 32), (32, 64), (64, 64)]])
        self.fc = nn.Linear(4096, 2 * latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for c in self.convs:
            x = F.silu(c(x))
        h = self.fc(x.flatten(1))
        return h[:, :self.latent_dim], h[:, self.latent_dim:]          # mu, log sigma^2


class Decoder(nn.Module):
    """z -> FC L x 4096 -> 8x8x64 -> 4 x [transposed conv 3x3 s2 + swish] (64, 64, 32, 32) -> conv 3x3 s1 1 + sigmoid."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.proj = nn.Linear(latent_dim, 4096)
        self.tconvs = nn.ModuleList([nn.ConvTranspose2d(ci, co, 3, stride=2, padding=1, output_padding=1)
                                     for ci, co in [(64, 64), (64, 64), (64, 32), (32, 32)]])
        self.out = nn.Conv2d(32, 1, 3, stride=1, padding=1)

    def logits(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z).view(-1, 64, 8, 8)
        for t in self.tconvs:
            h = F.silu(t(h))
        return self.out(h)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logits(z))


class VAE(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        glorot_init(self)

    def forward_logits(self, x: torch.Tensor, eps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reparameterised pass returning the decoder LOGITS: z = mu + exp(logvar/2) * eps (eps from the caller)."""
        mu, logvar = self.encoder(x)
        z = mu + torch.exp(0.5 * logvar) * eps
        return self.decoder.logits(z), mu, logvar

    def forward(self, x: torch.Tensor, eps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reparameterised pass returning the reconstruction image sigmoid(logits), mu, log sigma^2."""
        logits, mu, logvar = self.forward_logits(x, eps)
        return torch.sigmoid(logits), mu, logvar


class Critic(nn.Module):
    """Table 8. ``logit()`` returns the pre-sigmoid value a; the critic output of the paper is D = sigmoid(a)."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 5, stride=2, padding=2)
        self.blocks = nn.ModuleList([nn.Conv2d(ci, co, 5, stride=2, padding=2) for ci, co in [(32, 32), (32, 64), (64, 64)]])
        self.norms = nn.ModuleList([nn.GroupNorm(1, c) for c in (32, 64, 64)])
        self.out = nn.Conv2d(64, 1, 8, stride=1, padding=0)
        glorot_init(self)

    def logit(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.c1(x), 0.2)
        for c, n in zip(self.blocks, self.norms):
            h = F.leaky_relu(n(c(h)), 0.2)
        return self.out(h).flatten(1)[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit(x))


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def table6_flops(module: nn.Module, input_shape: tuple[int, ...]) -> int:
    """FLOPs with the convention that reproduces Table 6 (557M for the L = 64 decoder): 2 x multiply-accumulates of
    Linear, Conv2d and ConvTranspose2d layers, where a transposed convolution is counted at its OUTPUT resolution
    (H_out W_out C_in C_out k^2). Activations and normalisation are omitted (negligible)."""
    total = 0
    hooks = []

    def hook(m, inp, out):
        nonlocal total
        if isinstance(m, nn.Linear):
            total += 2 * m.in_features * m.out_features
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            ci = m.in_channels
            co = m.out_channels
            kh, kw = m.kernel_size
            total += 2 * out.shape[-2] * out.shape[-1] * ci * co * kh * kw // m.groups

    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(hook))
    try:
        with torch.no_grad():
            module(torch.zeros(1, *input_shape))
    finally:
        for h in hooks:
            h.remove()
    return total
